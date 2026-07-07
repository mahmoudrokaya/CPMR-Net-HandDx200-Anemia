from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = Path(
    r"D:\47\472\New-Papers\Anemia_Paper\Outputs\Stage5B_Local_Patch_Feature_Extraction\tables\participant_patch_features.csv"
)

OUTPUT_DIR = Path(
    r"D:\47\472\New-Papers\Anemia_Paper\Outputs\Stage6B_Local_Patch_Statistical_Testing"
)

TABLES_DIR = OUTPUT_DIR / "tables"
REPORTS_DIR = OUTPUT_DIR / "reports"

for d in [OUTPUT_DIR, TABLES_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def cohens_d(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    nx = len(x)
    ny = len(y)

    if nx < 2 or ny < 2:
        return np.nan

    pooled_std = np.sqrt(
        ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1))
        / (nx + ny - 2)
    )

    if pooled_std == 0:
        return np.nan

    return float((np.mean(x) - np.mean(y)) / pooled_std)


def safe_ttest(x, y):
    try:
        stat, p = ttest_ind(x, y, equal_var=False, nan_policy="omit")
        return float(stat), float(p)
    except Exception:
        return np.nan, np.nan


def safe_mannwhitney(x, y):
    try:
        stat, p = mannwhitneyu(x, y, alternative="two-sided")
        return float(stat), float(p)
    except Exception:
        return np.nan, np.nan


def classify_modality(feature):
    f = feature.lower()

    if f.startswith("rgb_"):
        return "RGB"

    if f.startswith("thermal_"):
        return "Thermal"

    return "Other"


def classify_feature_family(feature):
    f = feature.lower()

    if "lab" in f:
        return "Lab color"

    if "hsv" in f:
        return "HSV color"

    if "texture" in f or "glcm" in f:
        return "Texture"

    if "edge" in f:
        return "Edge"

    if "entropy" in f:
        return "Entropy"

    if "thermal" in f:
        return "Thermal intensity"

    if "_r_" in f or "_g_" in f or "_b_" in f or "r_g" in f or "r_b" in f or "g_b" in f:
        return "RGB channel"

    return "Other"


def extract_patch_id(feature):
    match = re.search(r"patch(\d+)", feature.lower())

    if match:
        return int(match.group(1))

    return None


def extract_view(feature):
    f = feature.lower()

    views = {
        "l_dorsal": "L_dorsal",
        "l_palmar": "L_palmar",
        "r_dorsal": "R_dorsal",
        "r_palmar": "R_palmar",
    }

    for key, value in views.items():
        if key in f:
            return value

    if "views_mean" in f:
        return "Aggregated_views"

    if "patch_all" in f:
        return "All_patches"

    return "Unknown"


def patch_position(patch_id):
    if patch_id is None:
        return None, None

    patch_id = int(patch_id)
    row = (patch_id - 1) // 3 + 1
    col = (patch_id - 1) % 3 + 1

    return row, col


def feature_scope(feature):
    f = feature.lower()

    if "patch_all" in f:
        return "All patches aggregated"

    if "views_mean" in f:
        return "Patch position across views"

    if "patch" in f:
        return "Patch specific"

    return "Other"


# ============================================================
# LOAD DATA
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

if "label" not in df.columns:
    raise ValueError("Missing label column.")

anemia_df = df[df["label"] == 1]
normal_df = df[df["label"] == 0]

non_feature_cols = {
    "participant_id",
    "label",
    "class_name",
    "sex",
    "age"
}

candidate_features = []

for col in df.columns:
    if col in non_feature_cols:
        continue

    if pd.api.types.is_numeric_dtype(df[col]):
        candidate_features.append(col)


# ============================================================
# STATISTICAL TESTING
# ============================================================

rows = []

for feature in candidate_features:
    anemia_values = pd.to_numeric(anemia_df[feature], errors="coerce").dropna()
    normal_values = pd.to_numeric(normal_df[feature], errors="coerce").dropna()

    if len(anemia_values) < 3 or len(normal_values) < 3:
        continue

    t_stat, t_p = safe_ttest(anemia_values, normal_values)
    u_stat, u_p = safe_mannwhitney(anemia_values, normal_values)
    d = cohens_d(anemia_values, normal_values)

    patch_id = extract_patch_id(feature)
    patch_row, patch_col = patch_position(patch_id)

    rows.append({
        "feature": feature,
        "modality": classify_modality(feature),
        "feature_family": classify_feature_family(feature),
        "view": extract_view(feature),
        "scope": feature_scope(feature),
        "patch_id": patch_id,
        "patch_row": patch_row,
        "patch_col": patch_col,
        "anemia_n": len(anemia_values),
        "normal_n": len(normal_values),
        "anemia_mean": float(anemia_values.mean()),
        "anemia_std": float(anemia_values.std()),
        "normal_mean": float(normal_values.mean()),
        "normal_std": float(normal_values.std()),
        "mean_difference_anemia_minus_normal": float(anemia_values.mean() - normal_values.mean()),
        "t_statistic": t_stat,
        "t_test_p_value": t_p,
        "mannwhitney_u_statistic": u_stat,
        "mannwhitney_p_value": u_p,
        "cohens_d": d,
        "abs_cohens_d": abs(d) if np.isfinite(d) else np.nan,
    })

results = pd.DataFrame(rows)

results = results.sort_values(
    by=["abs_cohens_d", "mannwhitney_p_value"],
    ascending=[False, True]
)

results["rank_by_abs_cohens_d"] = range(1, len(results) + 1)

results.to_csv(
    TABLES_DIR / "local_patch_feature_statistical_tests.csv",
    index=False
)


# ============================================================
# TOP FEATURE TABLES
# ============================================================

top25_overall = results.head(25)
top25_rgb = results[results["modality"] == "RGB"].head(25)
top25_thermal = results[results["modality"] == "Thermal"].head(25)

top25_overall.to_csv(TABLES_DIR / "top25_local_patch_features_overall.csv", index=False)
top25_rgb.to_csv(TABLES_DIR / "top25_local_patch_rgb_features.csv", index=False)
top25_thermal.to_csv(TABLES_DIR / "top25_local_patch_thermal_features.csv", index=False)


# ============================================================
# SUMMARY BY MODALITY, VIEW, PATCH, FAMILY
# ============================================================

family_summary = (
    results
    .groupby(["modality", "feature_family"])
    .agg(
        feature_count=("feature", "count"),
        mean_abs_cohens_d=("abs_cohens_d", "mean"),
        max_abs_cohens_d=("abs_cohens_d", "max"),
        min_mannwhitney_p=("mannwhitney_p_value", "min")
    )
    .reset_index()
    .sort_values(by="max_abs_cohens_d", ascending=False)
)

family_summary.to_csv(TABLES_DIR / "local_patch_feature_family_summary.csv", index=False)

view_summary = (
    results
    .groupby(["modality", "view"])
    .agg(
        feature_count=("feature", "count"),
        mean_abs_cohens_d=("abs_cohens_d", "mean"),
        max_abs_cohens_d=("abs_cohens_d", "max"),
        min_mannwhitney_p=("mannwhitney_p_value", "min")
    )
    .reset_index()
    .sort_values(by="max_abs_cohens_d", ascending=False)
)

view_summary.to_csv(TABLES_DIR / "local_patch_view_summary.csv", index=False)

patch_summary = (
    results
    .dropna(subset=["patch_id"])
    .groupby(["modality", "patch_id", "patch_row", "patch_col"])
    .agg(
        feature_count=("feature", "count"),
        mean_abs_cohens_d=("abs_cohens_d", "mean"),
        max_abs_cohens_d=("abs_cohens_d", "max"),
        min_mannwhitney_p=("mannwhitney_p_value", "min")
    )
    .reset_index()
    .sort_values(by="max_abs_cohens_d", ascending=False)
)

patch_summary.to_csv(TABLES_DIR / "local_patch_position_summary.csv", index=False)


# ============================================================
# HEATMAP TABLES FOR PATCH IMPORTANCE
# ============================================================

def save_patch_heatmap_table(modality):
    subset = patch_summary[patch_summary["modality"] == modality]

    heat = np.full((3, 3), np.nan)

    for _, row in subset.iterrows():
        r = int(row["patch_row"]) - 1
        c = int(row["patch_col"]) - 1
        heat[r, c] = row["max_abs_cohens_d"]

    heat_df = pd.DataFrame(
        heat,
        index=["row1", "row2", "row3"],
        columns=["col1", "col2", "col3"]
    )

    heat_df.to_csv(TABLES_DIR / f"{modality.lower()}_patch_importance_heatmap_table.csv")


save_patch_heatmap_table("RGB")
save_patch_heatmap_table("Thermal")


# ============================================================
# REPORT
# ============================================================

best = results.iloc[0] if len(results) else None

best_rgb = results[results["modality"] == "RGB"].iloc[0] if len(results[results["modality"] == "RGB"]) else None
best_thermal = results[results["modality"] == "Thermal"].iloc[0] if len(results[results["modality"] == "Thermal"]) else None

report = f"""# Stage 6B Local Patch Feature Statistical Testing Report

Stage 6B compared anemia and normal participants using local patch features extracted in Stage 5B.

## Input

`{INPUT_FILE}`

## Groups

Anemia participants: {len(anemia_df)}

Normal participants: {len(normal_df)}

## Features Tested

Local patch numeric features tested: {len(results)}

## Statistical Tests

Each feature was evaluated using:

- Welch's independent-samples t-test
- Mann–Whitney U test
- Cohen's d effect size

Features were ranked primarily by absolute Cohen's d.

## Strongest Local Patch Feature Overall

{f"Feature: `{best['feature']}`" if best is not None else "No valid feature found."}

{f"Modality: {best['modality']}" if best is not None else ""}

{f"View: {best['view']}" if best is not None else ""}

{f"Patch ID: {best['patch_id']}" if best is not None else ""}

{f"Feature family: {best['feature_family']}" if best is not None else ""}

{f"Cohen's d: {best['cohens_d']:.4f}" if best is not None else ""}

{f"Mann–Whitney p-value: {best['mannwhitney_p_value']:.6g}" if best is not None else ""}

## Strongest RGB Local Patch Feature

{f"Feature: `{best_rgb['feature']}`" if best_rgb is not None else "No valid RGB feature found."}

{f"Cohen's d: {best_rgb['cohens_d']:.4f}" if best_rgb is not None else ""}

{f"Mann–Whitney p-value: {best_rgb['mannwhitney_p_value']:.6g}" if best_rgb is not None else ""}

## Strongest Thermal Local Patch Feature

{f"Feature: `{best_thermal['feature']}`" if best_thermal is not None else "No valid thermal feature found."}

{f"Cohen's d: {best_thermal['cohens_d']:.4f}" if best_thermal is not None else ""}

{f"Mann–Whitney p-value: {best_thermal['mannwhitney_p_value']:.6g}" if best_thermal is not None else ""}

## Output Tables

- local_patch_feature_statistical_tests.csv
- top25_local_patch_features_overall.csv
- top25_local_patch_rgb_features.csv
- top25_local_patch_thermal_features.csv
- local_patch_feature_family_summary.csv
- local_patch_view_summary.csv
- local_patch_position_summary.csv
- rgb_patch_importance_heatmap_table.csv
- thermal_patch_importance_heatmap_table.csv

## Interpretation

This stage evaluates whether local image regions contain stronger anemia-related signals than global whole-image features. The patch importance summaries should be used to identify which modality, anatomical view, and spatial region provide the strongest discriminative information before model construction.
"""

with open(
    REPORTS_DIR / "Stage6B_Local_Patch_Feature_Statistical_Testing_Report.md",
    "w",
    encoding="utf-8"
) as f:
    f.write(report)


# ============================================================
# CONSOLE SUMMARY
# ============================================================

print("=" * 80)
print("STAGE 6B LOCAL PATCH FEATURE STATISTICAL TESTING COMPLETED")
print("=" * 80)
print(f"Anemia participants: {len(anemia_df)}")
print(f"Normal participants: {len(normal_df)}")
print(f"Features tested: {len(results)}")

if best is not None:
    print(f"Best overall feature: {best['feature']}")
    print(f"Cohen's d: {best['cohens_d']:.4f}")
    print(f"Mann–Whitney p-value: {best['mannwhitney_p_value']:.6g}")

if best_rgb is not None:
    print(f"Best RGB feature: {best_rgb['feature']}")
    print(f"RGB Cohen's d: {best_rgb['cohens_d']:.4f}")

if best_thermal is not None:
    print(f"Best Thermal feature: {best_thermal['feature']}")
    print(f"Thermal Cohen's d: {best_thermal['cohens_d']:.4f}")

print(f"Results saved to: {OUTPUT_DIR}")
print("=" * 80)