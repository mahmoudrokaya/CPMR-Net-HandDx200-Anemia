from pathlib import Path
import numpy as np
import pandas as pd

from scipy.stats import ttest_ind, mannwhitneyu


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = Path(
    r"D:\47\472\New-Papers\Anemia_Paper\Outputs\Stage5A_Global_Feature_Extraction\tables\participant_global_features.csv"
)

OUTPUT_DIR = Path(
    r"D:\47\472\New-Papers\Anemia_Paper\Outputs\Stage6A_Global_Feature_Statistical_Testing"
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


def classify_feature(feature):
    f = feature.lower()

    if f.startswith("rgb_"):
        modality = "RGB"
    elif f.startswith("thermal_"):
        modality = "Thermal"
    else:
        modality = "Other"

    if "lab" in f:
        family = "Lab color"
    elif "hsv" in f:
        family = "HSV color"
    elif "_r_" in f or "_g_" in f or "_b_" in f or "r_g" in f or "r_b" in f or "g_b" in f:
        family = "RGB channel"
    elif "texture" in f or "glcm" in f:
        family = "Texture"
    elif "edge" in f:
        family = "Edge"
    elif "entropy" in f:
        family = "Entropy"
    elif "thermal" in f:
        family = "Thermal intensity"
    else:
        family = "Other"

    return modality, family


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

exclude_cols = {
    "label",
    "age",
}

non_feature_cols = {
    "participant_id",
    "class_name",
    "sex",
}

candidate_features = []

for col in df.columns:
    if col in exclude_cols or col in non_feature_cols:
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

    modality, family = classify_feature(feature)

    rows.append({
        "feature": feature,
        "modality": modality,
        "feature_family": family,
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
    TABLES_DIR / "global_feature_statistical_tests.csv",
    index=False
)


# ============================================================
# TOP FEATURE TABLES
# ============================================================

top25_overall = results.head(25)
top25_rgb = results[results["modality"] == "RGB"].head(25)
top25_thermal = results[results["modality"] == "Thermal"].head(25)

top25_overall.to_csv(TABLES_DIR / "top25_global_features_overall.csv", index=False)
top25_rgb.to_csv(TABLES_DIR / "top25_global_rgb_features.csv", index=False)
top25_thermal.to_csv(TABLES_DIR / "top25_global_thermal_features.csv", index=False)


# ============================================================
# FAMILY SUMMARY
# ============================================================

family_summary = (
    results
    .groupby(["modality", "feature_family"])
    .agg(
        feature_count=("feature", "count"),
        mean_abs_cohens_d=("abs_cohens_d", "mean"),
        max_abs_cohens_d=("abs_cohens_d", "max"),
        min_p_value=("mannwhitney_p_value", "min")
    )
    .reset_index()
    .sort_values(by="max_abs_cohens_d", ascending=False)
)

family_summary.to_csv(
    TABLES_DIR / "global_feature_family_summary.csv",
    index=False
)


# ============================================================
# REPORT
# ============================================================

best = results.iloc[0] if len(results) else None

report = f"""# Stage 6A Global Feature Statistical Testing Report

Stage 6A compared anemia and normal participants using global RGB and thermal features extracted in Stage 5A.

## Input

`{INPUT_FILE}`

## Groups

Anemia participants: {len(anemia_df)}

Normal participants: {len(normal_df)}

## Features Tested

Global numeric features tested: {len(results)}

## Statistical Tests

Each feature was evaluated using:

- Welch's independent-samples t-test
- Mann–Whitney U test
- Cohen's d effect size

Features were ranked primarily by absolute Cohen's d.

## Strongest Global Feature

{f"Feature: `{best['feature']}`" if best is not None else "No valid feature found."}

{f"Modality: {best['modality']}" if best is not None else ""}

{f"Feature family: {best['feature_family']}" if best is not None else ""}

{f"Cohen's d: {best['cohens_d']:.4f}" if best is not None else ""}

{f"Mann–Whitney p-value: {best['mannwhitney_p_value']:.6g}" if best is not None else ""}

## Output Tables

- global_feature_statistical_tests.csv
- top25_global_features_overall.csv
- top25_global_rgb_features.csv
- top25_global_thermal_features.csv
- global_feature_family_summary.csv

## Interpretation

This stage identifies which whole-image RGB and thermal features show measurable differences between anemia and normal participants. The results should be interpreted as exploratory feature-level evidence, not as final model performance. Stage 6B should repeat the same statistical analysis on local patch features extracted in Stage 5B.
"""

with open(
    REPORTS_DIR / "Stage6A_Global_Feature_Statistical_Testing_Report.md",
    "w",
    encoding="utf-8"
) as f:
    f.write(report)


print("=" * 80)
print("STAGE 6A GLOBAL FEATURE STATISTICAL TESTING COMPLETED")
print("=" * 80)
print(f"Anemia participants: {len(anemia_df)}")
print(f"Normal participants: {len(normal_df)}")
print(f"Features tested: {len(results)}")
if best is not None:
    print(f"Best feature: {best['feature']}")
    print(f"Cohen's d: {best['cohens_d']:.4f}")
    print(f"Mann–Whitney p-value: {best['mannwhitney_p_value']:.6g}")
print(f"Results saved to: {OUTPUT_DIR}")
print("=" * 80)