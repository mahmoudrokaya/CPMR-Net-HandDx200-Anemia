from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

GLOBAL_FEATURE_FILE = Path(
    r"D:\47\472\New-Papers\Anemia_Paper\Outputs\Stage5A_Global_Feature_Extraction\tables\participant_global_features.csv"
)

PATCH_FEATURE_FILE = Path(
    r"D:\47\472\New-Papers\Anemia_Paper\Outputs\Stage5B_Local_Patch_Feature_Extraction\tables\participant_patch_features.csv"
)

OVERALL_RANKING_FILE = Path(
    r"D:\47\472\New-Papers\Anemia_Paper\Outputs\Stage6C_Global_vs_Local_Feature_Integration_and_Filtering\tables\top50_features_overall.csv"
)

FULL_FILTERED_STATS_FILE = Path(
    r"D:\47\472\New-Papers\Anemia_Paper\Outputs\Stage6C_Global_vs_Local_Feature_Integration_and_Filtering\tables\combined_global_local_feature_statistics_filtered.csv"
)

OUTPUT_DIR = Path(
    r"D:\47\472\New-Papers\Anemia_Paper\Outputs\Stage6D6A_NonredundantFeatureSelection"
)

TABLES_DIR = OUTPUT_DIR / "tables"
REPORTS_DIR = OUTPUT_DIR / "reports"

for d in [OUTPUT_DIR, TABLES_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIG
# ============================================================

CORRELATION_THRESHOLD = 0.90
TARGET_TOP_N = 50


# ============================================================
# HELPERS
# ============================================================

def load_required(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def merge_global_and_patch_features(global_df, patch_df):
    """
    Merge global and patch features by participant_id while keeping metadata
    only from the global feature table.
    """
    key_cols = ["participant_id", "label", "class_name", "sex", "age"]

    patch_drop_cols = [
        c for c in key_cols
        if c in patch_df.columns and c != "participant_id"
    ]

    patch_features_only = patch_df.drop(columns=patch_drop_cols, errors="ignore")

    merged = global_df.merge(
        patch_features_only,
        on="participant_id",
        how="inner"
    )

    return merged


def feature_priority_table(ranking_df, stats_df):
    """
    Build a priority table where larger abs_cohens_d is better,
    then smaller Mann-Whitney p-value is better.
    """
    if "feature" not in ranking_df.columns:
        raise ValueError("Ranking file must contain a 'feature' column.")

    if "feature" not in stats_df.columns:
        raise ValueError("Filtered stats file must contain a 'feature' column.")

    priority = ranking_df.merge(
        stats_df,
        on="feature",
        how="left",
        suffixes=("", "_stats")
    )

    if "abs_cohens_d" not in priority.columns:
        raise ValueError("Stats file must contain 'abs_cohens_d'.")

    if "mannwhitney_p_value" not in priority.columns:
        raise ValueError("Stats file must contain 'mannwhitney_p_value'.")

    priority["abs_cohens_d"] = pd.to_numeric(
        priority["abs_cohens_d"],
        errors="coerce"
    )

    priority["mannwhitney_p_value"] = pd.to_numeric(
        priority["mannwhitney_p_value"],
        errors="coerce"
    )

    priority = priority.sort_values(
        by=["abs_cohens_d", "mannwhitney_p_value"],
        ascending=[False, True]
    ).reset_index(drop=True)

    priority["initial_rank"] = priority.index + 1

    return priority


def is_better(candidate_row, selected_row):
    """
    Return True if candidate is better than selected according to:
    1. higher abs Cohen's d
    2. lower Mann-Whitney p-value
    """
    cand_d = candidate_row.get("abs_cohens_d", np.nan)
    sel_d = selected_row.get("abs_cohens_d", np.nan)

    cand_p = candidate_row.get("mannwhitney_p_value", np.nan)
    sel_p = selected_row.get("mannwhitney_p_value", np.nan)

    if pd.isna(sel_d):
        return True

    if pd.isna(cand_d):
        return False

    if cand_d > sel_d:
        return True

    if cand_d == sel_d:
        if pd.isna(sel_p):
            return True
        if not pd.isna(cand_p) and cand_p < sel_p:
            return True

    return False


# ============================================================
# LOAD DATA
# ============================================================

global_df = load_required(GLOBAL_FEATURE_FILE)
patch_df = load_required(PATCH_FEATURE_FILE)
overall_ranking = load_required(OVERALL_RANKING_FILE)
filtered_stats = load_required(FULL_FILTERED_STATS_FILE)

merged_df = merge_global_and_patch_features(global_df, patch_df)

priority = feature_priority_table(overall_ranking, filtered_stats)

candidate_features = priority["feature"].dropna().astype(str).tolist()

available_features = [f for f in candidate_features if f in merged_df.columns]
missing_features = [f for f in candidate_features if f not in merged_df.columns]

if not available_features:
    raise ValueError("None of the candidate features were found in the merged feature matrix.")

priority = priority[priority["feature"].isin(available_features)].copy()

X = merged_df[available_features].copy()
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median(numeric_only=True))

# Remove zero-variance features before correlation
variances = X.var(axis=0)
nonzero_features = variances[variances > 0].index.tolist()

X = X[nonzero_features]
priority = priority[priority["feature"].isin(nonzero_features)].copy()

priority.to_csv(
    TABLES_DIR / "candidate_feature_priority_table.csv",
    index=False
)

pd.DataFrame({
    "feature": candidate_features,
    "available_in_merged_matrix": [f in available_features for f in candidate_features],
    "nonzero_variance": [f in nonzero_features for f in candidate_features if f in available_features]
}).to_csv(
    TABLES_DIR / "candidate_feature_availability.csv",
    index=False
)


# ============================================================
# CORRELATION MATRIX
# ============================================================

corr_matrix = X.corr().abs()

corr_matrix.to_csv(
    TABLES_DIR / "candidate_feature_abs_correlation_matrix.csv"
)


# ============================================================
# REDUNDANCY PRUNING
# ============================================================

selected_rows = []
removed_rows = []

for _, candidate in priority.iterrows():

    candidate_feature = candidate["feature"]

    if candidate_feature not in X.columns:
        continue

    redundant_with = None
    redundant_corr = None
    redundant_selected_row = None

    for selected in selected_rows:
        selected_feature = selected["feature"]

        if selected_feature not in corr_matrix.columns:
            continue

        corr_value = corr_matrix.loc[candidate_feature, selected_feature]

        if corr_value >= CORRELATION_THRESHOLD:
            redundant_with = selected_feature
            redundant_corr = corr_value
            redundant_selected_row = selected
            break

    if redundant_with is None:
        selected_rows.append(candidate.to_dict())
    else:
        removed_rows.append({
            "removed_feature": candidate_feature,
            "kept_feature": redundant_with,
            "absolute_correlation": redundant_corr,
            "removed_abs_cohens_d": candidate.get("abs_cohens_d", np.nan),
            "kept_abs_cohens_d": redundant_selected_row.get("abs_cohens_d", np.nan),
            "removed_mannwhitney_p": candidate.get("mannwhitney_p_value", np.nan),
            "kept_mannwhitney_p": redundant_selected_row.get("mannwhitney_p_value", np.nan),
            "reason": f"correlation >= {CORRELATION_THRESHOLD}"
        })

    if len(selected_rows) >= TARGET_TOP_N:
        # We stop at 50 retained features, but usually fewer may remain
        # if the candidate pool is only top 50.
        pass


selected_df = pd.DataFrame(selected_rows)
removed_df = pd.DataFrame(removed_rows)

if len(selected_df):
    selected_df = selected_df.sort_values(
        by=["abs_cohens_d", "mannwhitney_p_value"],
        ascending=[False, True]
    ).reset_index(drop=True)

    selected_df["nonredundant_rank"] = selected_df.index + 1

removed_df.to_csv(
    TABLES_DIR / "redundant_features_removed.csv",
    index=False
)

selected_df.to_csv(
    TABLES_DIR / "top50_nonredundant_features.csv",
    index=False
)

pd.DataFrame({
    "feature": selected_df["feature"].tolist()
}).to_csv(
    TABLES_DIR / "selected_nonredundant_feature_names.csv",
    index=False
)


# ============================================================
# SAVE NONREDUNDANT DATASET
# ============================================================

selected_feature_names = selected_df["feature"].tolist()

metadata_cols = [
    c for c in ["participant_id", "label", "class_name", "sex", "age"]
    if c in merged_df.columns
]

nonredundant_dataset = merged_df[metadata_cols + selected_feature_names].copy()

nonredundant_dataset.to_csv(
    TABLES_DIR / "nonredundant_feature_dataset.csv",
    index=False
)


# ============================================================
# SUMMARY TABLES
# ============================================================

if len(selected_df):
    source_summary = (
        selected_df
        .groupby("feature_source")
        .agg(
            selected_count=("feature", "count"),
            mean_abs_cohens_d=("abs_cohens_d", "mean"),
            max_abs_cohens_d=("abs_cohens_d", "max"),
            min_mannwhitney_p=("mannwhitney_p_value", "min")
        )
        .reset_index()
        .sort_values(by="max_abs_cohens_d", ascending=False)
    )
else:
    source_summary = pd.DataFrame()

source_summary.to_csv(
    TABLES_DIR / "nonredundant_summary_by_source.csv",
    index=False
)

if len(selected_df):
    modality_summary = (
        selected_df
        .groupby("modality")
        .agg(
            selected_count=("feature", "count"),
            mean_abs_cohens_d=("abs_cohens_d", "mean"),
            max_abs_cohens_d=("abs_cohens_d", "max"),
            min_mannwhitney_p=("mannwhitney_p_value", "min")
        )
        .reset_index()
        .sort_values(by="max_abs_cohens_d", ascending=False)
    )
else:
    modality_summary = pd.DataFrame()

modality_summary.to_csv(
    TABLES_DIR / "nonredundant_summary_by_modality.csv",
    index=False
)

if len(selected_df):
    family_summary = (
        selected_df
        .groupby(["modality", "feature_family"])
        .agg(
            selected_count=("feature", "count"),
            mean_abs_cohens_d=("abs_cohens_d", "mean"),
            max_abs_cohens_d=("abs_cohens_d", "max"),
            min_mannwhitney_p=("mannwhitney_p_value", "min")
        )
        .reset_index()
        .sort_values(by="max_abs_cohens_d", ascending=False)
    )
else:
    family_summary = pd.DataFrame()

family_summary.to_csv(
    TABLES_DIR / "nonredundant_summary_by_modality_family.csv",
    index=False
)


# ============================================================
# REPORT
# ============================================================

best = selected_df.iloc[0] if len(selected_df) else None

report = f"""# Stage 6D6A Nonredundant Feature Selection Report

Stage 6D6A selected a nonredundant feature subset from the top-ranked overall features.

## Inputs

Global feature matrix:

`{GLOBAL_FEATURE_FILE}`

Local patch feature matrix:

`{PATCH_FEATURE_FILE}`

Overall ranking file:

`{OVERALL_RANKING_FILE}`

Filtered statistical feature table:

`{FULL_FILTERED_STATS_FILE}`

## Redundancy Criterion

Absolute Pearson correlation threshold:

{CORRELATION_THRESHOLD}

If two features had absolute correlation greater than or equal to this threshold, the feature with the larger absolute Cohen's d was retained. If effect sizes were tied, the feature with the smaller Mann–Whitney p-value was retained.

## Feature Counts

Candidate features from overall ranking: {len(candidate_features)}

Available candidate features: {len(available_features)}

Nonzero-variance candidate features: {len(nonzero_features)}

Selected nonredundant features: {len(selected_df)}

Removed redundant features: {len(removed_df)}

## Strongest Selected Feature

{f"Feature: `{best['feature']}`" if best is not None else "No feature selected."}

{f"Source: {best['feature_source']}" if best is not None else ""}

{f"Modality: {best['modality']}" if best is not None else ""}

{f"Feature family: {best['feature_family']}" if best is not None else ""}

{f"Cohen's d: {best['cohens_d']:.4f}" if best is not None else ""}

{f"Mann–Whitney p-value: {best['mannwhitney_p_value']:.6g}" if best is not None else ""}

## Output Tables

- candidate_feature_priority_table.csv
- candidate_feature_availability.csv
- candidate_feature_abs_correlation_matrix.csv
- redundant_features_removed.csv
- top50_nonredundant_features.csv
- selected_nonredundant_feature_names.csv
- nonredundant_feature_dataset.csv
- nonredundant_summary_by_source.csv
- nonredundant_summary_by_modality.csv
- nonredundant_summary_by_modality_family.csv

## Interpretation

This stage reduces feature redundancy among the strongest statistically ranked features. The resulting nonredundant feature set should be benchmarked in Stage 6D6B to determine whether removing correlated features improves generalization, recall, F1-score, or ROC-AUC.
"""

with open(
    REPORTS_DIR / "Stage6D6A_NonredundantFeatureSelection_Report.md",
    "w",
    encoding="utf-8"
) as f:
    f.write(report)


# ============================================================
# CONSOLE OUTPUT
# ============================================================

print("=" * 80)
print("STAGE 6D6A NONREDUNDANT FEATURE SELECTION COMPLETED")
print("=" * 80)
print(f"Candidate features: {len(candidate_features)}")
print(f"Available features: {len(available_features)}")
print(f"Nonzero-variance features: {len(nonzero_features)}")
print(f"Selected nonredundant features: {len(selected_df)}")
print(f"Removed redundant features: {len(removed_df)}")

if best is not None:
    print(f"Best selected feature: {best['feature']}")
    print(f"Cohen's d: {best['cohens_d']:.4f}")
    print(f"Mann–Whitney p-value: {best['mannwhitney_p_value']:.6g}")

print(f"Results saved to: {OUTPUT_DIR}")
print("=" * 80)