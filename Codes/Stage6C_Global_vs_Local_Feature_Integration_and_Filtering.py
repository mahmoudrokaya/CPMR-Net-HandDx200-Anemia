from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

GLOBAL_STATS = Path(
    r"D:\47\472\New-Papers\Anemia_Paper\Outputs\Stage6A_Global_Feature_Statistical_Testing\tables\global_feature_statistical_tests.csv"
)

LOCAL_STATS = Path(
    r"D:\47\472\New-Papers\Anemia_Paper\Outputs\Stage6B_Local_Patch_Statistical_Testing\tables\local_patch_feature_statistical_tests.csv"
)

GLOBAL_FEATURES = Path(
    r"D:\47\472\New-Papers\Anemia_Paper\Outputs\Stage5A_Global_Feature_Extraction\tables\participant_global_features.csv"
)

LOCAL_FEATURES = Path(
    r"D:\47\472\New-Papers\Anemia_Paper\Outputs\Stage5B_Local_Patch_Feature_Extraction\tables\participant_patch_features.csv"
)

OUTPUT_DIR = Path(
    r"D:\47\472\New-Papers\Anemia_Paper\Outputs\Stage6C_Global_vs_Local_Feature_Integration_and_Filtering"
)

TABLES_DIR = OUTPUT_DIR / "tables"
REPORTS_DIR = OUTPUT_DIR / "reports"

for d in [OUTPUT_DIR, TABLES_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIG
# ============================================================

MIN_VARIANCE = 1e-8
MIN_VALID_PER_GROUP = 10
TOP_N = 50


# ============================================================
# HELPERS
# ============================================================

def load_required(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def add_source_type(df, source_type):
    df = df.copy()
    df["feature_source"] = source_type
    return df


def compute_stability_metrics(feature_df, feature_names):
    rows = []

    for feature in feature_names:
        if feature not in feature_df.columns:
            continue

        values = pd.to_numeric(feature_df[feature], errors="coerce")
        valid = values.dropna()

        if "label" in feature_df.columns:
            anemia_values = pd.to_numeric(
                feature_df.loc[feature_df["label"] == 1, feature],
                errors="coerce"
            ).dropna()

            normal_values = pd.to_numeric(
                feature_df.loc[feature_df["label"] == 0, feature],
                errors="coerce"
            ).dropna()
        else:
            anemia_values = pd.Series(dtype=float)
            normal_values = pd.Series(dtype=float)

        rows.append({
            "feature": feature,
            "overall_valid_n": int(valid.shape[0]),
            "overall_missing_n": int(values.isna().sum()),
            "overall_variance": float(valid.var()) if len(valid) > 1 else np.nan,
            "overall_std": float(valid.std()) if len(valid) > 1 else np.nan,
            "anemia_valid_n": int(anemia_values.shape[0]),
            "normal_valid_n": int(normal_values.shape[0]),
            "anemia_variance": float(anemia_values.var()) if len(anemia_values) > 1 else np.nan,
            "normal_variance": float(normal_values.var()) if len(normal_values) > 1 else np.nan,
        })

    return pd.DataFrame(rows)


def classify_quality(row):
    if pd.isna(row.get("overall_variance", np.nan)):
        return "remove_no_variance_info"

    if row["overall_variance"] <= MIN_VARIANCE:
        return "remove_low_variance"

    if row["anemia_valid_n"] < MIN_VALID_PER_GROUP or row["normal_valid_n"] < MIN_VALID_PER_GROUP:
        return "remove_insufficient_group_data"

    if not np.isfinite(row.get("abs_cohens_d", np.nan)):
        return "remove_invalid_effect_size"

    if not np.isfinite(row.get("mannwhitney_p_value", np.nan)):
        return "remove_invalid_p_value"

    return "keep"


def select_top_nonredundant(df, top_n=50):
    """
    This is a simple conservative shortlist:
    - rank by abs_cohens_d
    - keep at most one very similar feature name root when possible
    """
    selected = []
    seen_roots = set()

    sorted_df = df.sort_values(
        by=["abs_cohens_d", "mannwhitney_p_value"],
        ascending=[False, True]
    )

    for _, row in sorted_df.iterrows():
        feature = row["feature"]

        root = feature.lower()

        for token in [
            "_mean",
            "_std",
            "_var",
            "_median",
            "_q25",
            "_q75",
            "_min",
            "_max",
            "_range",
            "_entropy",
        ]:
            root = root.replace(token, "")

        if root in seen_roots:
            continue

        selected.append(row)
        seen_roots.add(root)

        if len(selected) >= top_n:
            break

    if selected:
        return pd.DataFrame(selected)

    return pd.DataFrame(columns=df.columns)


# ============================================================
# LOAD INPUTS
# ============================================================

global_stats = add_source_type(load_required(GLOBAL_STATS), "Global")
local_stats = add_source_type(load_required(LOCAL_STATS), "LocalPatch")

global_features = load_required(GLOBAL_FEATURES)
local_features = load_required(LOCAL_FEATURES)

global_feature_names = global_stats["feature"].tolist()
local_feature_names = local_stats["feature"].tolist()


# ============================================================
# STABILITY CHECKS
# ============================================================

global_stability = compute_stability_metrics(global_features, global_feature_names)
local_stability = compute_stability_metrics(local_features, local_feature_names)

global_stability["feature_source"] = "Global"
local_stability["feature_source"] = "LocalPatch"

stability = pd.concat(
    [global_stability, local_stability],
    ignore_index=True
)

stability.to_csv(
    TABLES_DIR / "feature_stability_metrics.csv",
    index=False
)


# ============================================================
# COMBINE STATISTICAL RESULTS
# ============================================================

combined_stats = pd.concat(
    [global_stats, local_stats],
    ignore_index=True,
    sort=False
)

combined_stats = combined_stats.merge(
    stability,
    on=["feature", "feature_source"],
    how="left"
)

combined_stats["quality_flag"] = combined_stats.apply(
    classify_quality,
    axis=1
)

combined_stats.to_csv(
    TABLES_DIR / "combined_global_local_feature_statistics_all.csv",
    index=False
)


filtered = combined_stats[
    combined_stats["quality_flag"] == "keep"
].copy()

filtered = filtered.sort_values(
    by=["abs_cohens_d", "mannwhitney_p_value"],
    ascending=[False, True]
)

filtered["final_rank"] = range(1, len(filtered) + 1)

filtered.to_csv(
    TABLES_DIR / "combined_global_local_feature_statistics_filtered.csv",
    index=False
)


# ============================================================
# TOP FEATURE TABLES
# ============================================================

top_overall = filtered.head(TOP_N)
top_global = filtered[filtered["feature_source"] == "Global"].head(TOP_N)
top_local = filtered[filtered["feature_source"] == "LocalPatch"].head(TOP_N)
top_rgb = filtered[filtered["modality"] == "RGB"].head(TOP_N)
top_thermal = filtered[filtered["modality"] == "Thermal"].head(TOP_N)

top_overall.to_csv(TABLES_DIR / "top50_features_overall.csv", index=False)
top_global.to_csv(TABLES_DIR / "top50_global_features.csv", index=False)
top_local.to_csv(TABLES_DIR / "top50_local_patch_features.csv", index=False)
top_rgb.to_csv(TABLES_DIR / "top50_rgb_features.csv", index=False)
top_thermal.to_csv(TABLES_DIR / "top50_thermal_features.csv", index=False)


# ============================================================
# NON-REDUNDANT SHORTLIST
# ============================================================

shortlist = select_top_nonredundant(filtered, top_n=50)
shortlist.to_csv(
    TABLES_DIR / "final_nonredundant_feature_shortlist_top50.csv",
    index=False
)


# ============================================================
# SUMMARY TABLES
# ============================================================

source_summary = (
    filtered
    .groupby("feature_source")
    .agg(
        feature_count=("feature", "count"),
        mean_abs_cohens_d=("abs_cohens_d", "mean"),
        max_abs_cohens_d=("abs_cohens_d", "max"),
        min_mannwhitney_p=("mannwhitney_p_value", "min")
    )
    .reset_index()
    .sort_values(by="max_abs_cohens_d", ascending=False)
)

source_summary.to_csv(TABLES_DIR / "summary_by_feature_source.csv", index=False)

modality_summary = (
    filtered
    .groupby(["feature_source", "modality"])
    .agg(
        feature_count=("feature", "count"),
        mean_abs_cohens_d=("abs_cohens_d", "mean"),
        max_abs_cohens_d=("abs_cohens_d", "max"),
        min_mannwhitney_p=("mannwhitney_p_value", "min")
    )
    .reset_index()
    .sort_values(by="max_abs_cohens_d", ascending=False)
)

modality_summary.to_csv(TABLES_DIR / "summary_by_source_and_modality.csv", index=False)

family_summary = (
    filtered
    .groupby(["feature_source", "modality", "feature_family"])
    .agg(
        feature_count=("feature", "count"),
        mean_abs_cohens_d=("abs_cohens_d", "mean"),
        max_abs_cohens_d=("abs_cohens_d", "max"),
        min_mannwhitney_p=("mannwhitney_p_value", "min")
    )
    .reset_index()
    .sort_values(by="max_abs_cohens_d", ascending=False)
)

family_summary.to_csv(TABLES_DIR / "summary_by_source_modality_family.csv", index=False)


quality_summary = (
    combined_stats
    .groupby(["feature_source", "quality_flag"])
    .size()
    .reset_index(name="count")
)

quality_summary.to_csv(TABLES_DIR / "quality_filtering_summary.csv", index=False)


# ============================================================
# MASTER MODELING FEATURE SETS
# ============================================================

# Save selected feature names only, useful for later modeling stages
pd.DataFrame({"feature": top_overall["feature"].tolist()}).to_csv(
    TABLES_DIR / "selected_feature_names_top50_overall.csv",
    index=False
)

pd.DataFrame({"feature": shortlist["feature"].tolist()}).to_csv(
    TABLES_DIR / "selected_feature_names_nonredundant_top50.csv",
    index=False
)


# ============================================================
# REPORT
# ============================================================

best = filtered.iloc[0] if len(filtered) else None

best_global = (
    filtered[filtered["feature_source"] == "Global"].iloc[0]
    if len(filtered[filtered["feature_source"] == "Global"]) else None
)

best_local = (
    filtered[filtered["feature_source"] == "LocalPatch"].iloc[0]
    if len(filtered[filtered["feature_source"] == "LocalPatch"]) else None
)

best_rgb = (
    filtered[filtered["modality"] == "RGB"].iloc[0]
    if len(filtered[filtered["modality"] == "RGB"]) else None
)

best_thermal = (
    filtered[filtered["modality"] == "Thermal"].iloc[0]
    if len(filtered[filtered["modality"] == "Thermal"]) else None
)

report = f"""# Stage 6C Global vs Local Feature Integration and Filtering Report

Stage 6C integrated the statistical results from Stage 6A and Stage 6B, removed unstable or low-information features, and generated final feature rankings for downstream modeling.

## Inputs

Global statistical results:

`{GLOBAL_STATS}`

Local patch statistical results:

`{LOCAL_STATS}`

Global feature matrix:

`{GLOBAL_FEATURES}`

Local patch feature matrix:

`{LOCAL_FEATURES}`

## Feature Counts

Total global statistical features: {len(global_stats)}

Total local patch statistical features: {len(local_stats)}

Total combined features before filtering: {len(combined_stats)}

Total features retained after filtering: {len(filtered)}

## Filtering Rules

Features were removed if they had:

- overall variance ≤ {MIN_VARIANCE}
- fewer than {MIN_VALID_PER_GROUP} valid samples in either class
- invalid Cohen's d
- invalid Mann–Whitney p-value

## Strongest Feature Overall

{f"Feature: `{best['feature']}`" if best is not None else "No retained feature found."}

{f"Source: {best['feature_source']}" if best is not None else ""}

{f"Modality: {best['modality']}" if best is not None else ""}

{f"Feature family: {best['feature_family']}" if best is not None else ""}

{f"Cohen's d: {best['cohens_d']:.4f}" if best is not None else ""}

{f"Mann–Whitney p-value: {best['mannwhitney_p_value']:.6g}" if best is not None else ""}

## Best Global Feature

{f"Feature: `{best_global['feature']}`" if best_global is not None else "No retained global feature found."}

{f"Cohen's d: {best_global['cohens_d']:.4f}" if best_global is not None else ""}

## Best Local Patch Feature

{f"Feature: `{best_local['feature']}`" if best_local is not None else "No retained local feature found."}

{f"Cohen's d: {best_local['cohens_d']:.4f}" if best_local is not None else ""}

## Best RGB Feature

{f"Feature: `{best_rgb['feature']}`" if best_rgb is not None else "No retained RGB feature found."}

{f"Cohen's d: {best_rgb['cohens_d']:.4f}" if best_rgb is not None else ""}

## Best Thermal Feature

{f"Feature: `{best_thermal['feature']}`" if best_thermal is not None else "No retained thermal feature found."}

{f"Cohen's d: {best_thermal['cohens_d']:.4f}" if best_thermal is not None else ""}

## Output Tables

- combined_global_local_feature_statistics_all.csv
- combined_global_local_feature_statistics_filtered.csv
- feature_stability_metrics.csv
- quality_filtering_summary.csv
- top50_features_overall.csv
- top50_global_features.csv
- top50_local_patch_features.csv
- top50_rgb_features.csv
- top50_thermal_features.csv
- final_nonredundant_feature_shortlist_top50.csv
- selected_feature_names_top50_overall.csv
- selected_feature_names_nonredundant_top50.csv
- summary_by_feature_source.csv
- summary_by_source_and_modality.csv
- summary_by_source_modality_family.csv

## Interpretation

This stage produces the final statistically ranked and quality-filtered feature set. The retained features should be used to guide the first modeling experiments and to identify whether global or local features provide stronger evidence for binary anemia classification.
"""

with open(
    REPORTS_DIR / "Stage6C_Global_vs_Local_Feature_Integration_and_Filtering_Report.md",
    "w",
    encoding="utf-8"
) as f:
    f.write(report)


# ============================================================
# CONSOLE SUMMARY
# ============================================================

print("=" * 80)
print("STAGE 6C GLOBAL VS LOCAL FEATURE INTEGRATION COMPLETED")
print("=" * 80)
print(f"Global features: {len(global_stats)}")
print(f"Local patch features: {len(local_stats)}")
print(f"Combined before filtering: {len(combined_stats)}")
print(f"Retained after filtering: {len(filtered)}")

if best is not None:
    print(f"Best overall feature: {best['feature']}")
    print(f"Source: {best['feature_source']}")
    print(f"Modality: {best['modality']}")
    print(f"Cohen's d: {best['cohens_d']:.4f}")
    print(f"Mann–Whitney p-value: {best['mannwhitney_p_value']:.6g}")

print(f"Results saved to: {OUTPUT_DIR}")
print("=" * 80)