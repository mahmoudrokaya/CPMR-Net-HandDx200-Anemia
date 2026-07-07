import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from scipy.stats import entropy

from skimage.color import rgb2hsv, rgb2lab
from skimage.filters import sobel
from skimage.feature import graycomatrix, graycoprops


# ============================================================
# PATHS
# ============================================================

STAGE4_DIR = Path(
    r"D:\47\472\New-Papers\Anemia_Paper\Outputs\Stage4_Dataset_Characterization_and_Inventory"
)

PARTICIPANT_DATASET = STAGE4_DIR / "tables" / "participant_dataset.csv"

OUTPUT_DIR = Path(
    r"D:\47\472\New-Papers\Anemia_Paper\Outputs\Stage5A_Global_Feature_Extraction"
)

TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"

for d in [OUTPUT_DIR, TABLES_DIR, FIGURES_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIG
# ============================================================

IMAGE_SIZE = (224, 224)

VIEWS = [
    "l_dorsal",
    "l_palmar",
    "r_dorsal",
    "r_palmar"
]

RGB_COLUMNS = [f"rgb_{v}" for v in VIEWS]
THERMAL_COLUMNS = [f"thermal_{v}" for v in VIEWS]


# ============================================================
# IMAGE LOADING
# ============================================================

def load_rgb(path):
    img = Image.open(path).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr


def load_gray(path):
    img = Image.open(path).convert("L")
    img = img.resize(IMAGE_SIZE)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr


# ============================================================
# FEATURE HELPERS
# ============================================================

def safe_entropy(arr, bins=256):
    values = np.asarray(arr).astype(np.float32).ravel()
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    hist, _ = np.histogram(values, bins=bins, range=(0, 1), density=False)
    hist = hist.astype(np.float64)
    hist = hist / (hist.sum() + 1e-12)
    return float(entropy(hist + 1e-12))


def basic_stats(arr, prefix):
    arr = np.asarray(arr).astype(np.float32)

    return {
        f"{prefix}_mean": float(np.nanmean(arr)),
        f"{prefix}_std": float(np.nanstd(arr)),
        f"{prefix}_var": float(np.nanvar(arr)),
        f"{prefix}_median": float(np.nanmedian(arr)),
        f"{prefix}_q25": float(np.nanquantile(arr, 0.25)),
        f"{prefix}_q75": float(np.nanquantile(arr, 0.75)),
        f"{prefix}_min": float(np.nanmin(arr)),
        f"{prefix}_max": float(np.nanmax(arr)),
        f"{prefix}_range": float(np.nanmax(arr) - np.nanmin(arr)),
        f"{prefix}_entropy": safe_entropy(arr),
    }


def glcm_features(gray, prefix):
    gray_u8 = np.clip(gray * 255, 0, 255).astype(np.uint8)

    glcm = graycomatrix(
        gray_u8,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )

    return {
        f"{prefix}_glcm_contrast": float(graycoprops(glcm, "contrast")[0, 0]),
        f"{prefix}_glcm_dissimilarity": float(graycoprops(glcm, "dissimilarity")[0, 0]),
        f"{prefix}_glcm_homogeneity": float(graycoprops(glcm, "homogeneity")[0, 0]),
        f"{prefix}_glcm_energy": float(graycoprops(glcm, "energy")[0, 0]),
        f"{prefix}_glcm_correlation": float(graycoprops(glcm, "correlation")[0, 0]),
    }


# ============================================================
# RGB GLOBAL FEATURES
# ============================================================

def extract_rgb_global_features(img, prefix):
    features = {}

    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    gray = np.mean(img, axis=2)

    features.update(basic_stats(r, f"{prefix}_R"))
    features.update(basic_stats(g, f"{prefix}_G"))
    features.update(basic_stats(b, f"{prefix}_B"))
    features.update(basic_stats(gray, f"{prefix}_gray"))

    features[f"{prefix}_R_G_ratio"] = float(np.mean(r) / (np.mean(g) + 1e-8))
    features[f"{prefix}_R_B_ratio"] = float(np.mean(r) / (np.mean(b) + 1e-8))
    features[f"{prefix}_G_B_ratio"] = float(np.mean(g) / (np.mean(b) + 1e-8))

    features[f"{prefix}_R_minus_G"] = float(np.mean(r) - np.mean(g))
    features[f"{prefix}_R_minus_B"] = float(np.mean(r) - np.mean(b))
    features[f"{prefix}_G_minus_B"] = float(np.mean(g) - np.mean(b))

    hsv = rgb2hsv(img)
    lab = rgb2lab(img)

    features.update(basic_stats(hsv[:, :, 0], f"{prefix}_HSV_H"))
    features.update(basic_stats(hsv[:, :, 1], f"{prefix}_HSV_S"))
    features.update(basic_stats(hsv[:, :, 2], f"{prefix}_HSV_V"))

    features.update(basic_stats(lab[:, :, 0], f"{prefix}_LAB_L"))
    features.update(basic_stats(lab[:, :, 1], f"{prefix}_LAB_A"))
    features.update(basic_stats(lab[:, :, 2], f"{prefix}_LAB_B"))

    edge = sobel(gray)
    features.update(basic_stats(edge, f"{prefix}_edge"))

    features.update(glcm_features(gray, f"{prefix}_texture"))

    # Simple pallor-related proxy features
    features[f"{prefix}_low_red_ratio"] = float(np.mean(r < np.quantile(r, 0.25)))
    features[f"{prefix}_high_lightness_ratio"] = float(np.mean(lab[:, :, 0] > np.quantile(lab[:, :, 0], 0.75)))
    features[f"{prefix}_low_saturation_ratio"] = float(np.mean(hsv[:, :, 1] < np.quantile(hsv[:, :, 1], 0.25)))

    return features


# ============================================================
# THERMAL GLOBAL FEATURES
# ============================================================

def extract_thermal_global_features(gray, prefix):
    features = {}

    features.update(basic_stats(gray, f"{prefix}_thermal"))

    edge = sobel(gray)
    features.update(basic_stats(edge, f"{prefix}_thermal_edge"))

    features.update(glcm_features(gray, f"{prefix}_thermal_texture"))

    q10 = np.quantile(gray, 0.10)
    q25 = np.quantile(gray, 0.25)
    q75 = np.quantile(gray, 0.75)
    q90 = np.quantile(gray, 0.90)

    features[f"{prefix}_thermal_q10"] = float(q10)
    features[f"{prefix}_thermal_q90"] = float(q90)
    features[f"{prefix}_thermal_iqr"] = float(q75 - q25)
    features[f"{prefix}_thermal_hot_region_ratio"] = float(np.mean(gray >= q90))
    features[f"{prefix}_thermal_cold_region_ratio"] = float(np.mean(gray <= q10))
    features[f"{prefix}_thermal_midrange_ratio"] = float(np.mean((gray > q25) & (gray < q75)))

    return features


# ============================================================
# PARTICIPANT AGGREGATION
# ============================================================

def summarize_across_views(view_feature_list, modality):
    aggregated = {}

    if not view_feature_list:
        return aggregated

    feature_keys = list(view_feature_list[0].keys())

    for key in feature_keys:
        values = np.array(
            [d[key] for d in view_feature_list if key in d],
            dtype=np.float64
        )

        values = values[np.isfinite(values)]

        if len(values) == 0:
            continue

        clean_key = key

        for v in VIEWS:
            clean_key = clean_key.replace(f"{modality}_{v}_", "")

        aggregated[f"{modality}_global_views_mean_{clean_key}"] = float(np.mean(values))
        aggregated[f"{modality}_global_views_std_{clean_key}"] = float(np.std(values))
        aggregated[f"{modality}_global_views_min_{clean_key}"] = float(np.min(values))
        aggregated[f"{modality}_global_views_max_{clean_key}"] = float(np.max(values))

    return aggregated


def process_participant(row):
    base = {
        "participant_id": row["participant_id"],
        "label": int(row["label"]),
        "class_name": row["class_name"],
        "sex": row.get("sex", None),
        "age": row.get("age", None),
    }

    rgb_view_features = []
    thermal_view_features = []

    # RGB views
    for view in VIEWS:
        col = f"rgb_{view}"
        path = row[col]

        img = load_rgb(path)
        prefix = f"rgb_{view}"

        feats = extract_rgb_global_features(img, prefix)
        base.update(feats)
        rgb_view_features.append(feats)

    # Thermal views
    for view in VIEWS:
        col = f"thermal_{view}"
        path = row[col]

        img = load_gray(path)
        prefix = f"thermal_{view}"

        feats = extract_thermal_global_features(img, prefix)
        base.update(feats)
        thermal_view_features.append(feats)

    base.update(summarize_across_views(rgb_view_features, "rgb"))
    base.update(summarize_across_views(thermal_view_features, "thermal"))

    return base


# ============================================================
# MAIN
# ============================================================

if not PARTICIPANT_DATASET.exists():
    raise FileNotFoundError(f"Missing participant dataset: {PARTICIPANT_DATASET}")

df = pd.read_csv(PARTICIPANT_DATASET)

required_cols = [
    "participant_id",
    "label",
    "class_name",
    *RGB_COLUMNS,
    *THERMAL_COLUMNS
]

missing_cols = [c for c in required_cols if c not in df.columns]

if missing_cols:
    raise ValueError(f"Missing required columns in participant dataset: {missing_cols}")

rows = []

for i, row in df.iterrows():
    print(f"Processing {i + 1}/{len(df)}: {row['participant_id']}")
    rows.append(process_participant(row))

global_features = pd.DataFrame(rows)

global_features.to_csv(
    TABLES_DIR / "participant_global_features.csv",
    index=False
)


# ============================================================
# FEATURE SUMMARY
# ============================================================

numeric_cols = global_features.select_dtypes(include=[np.number]).columns.tolist()
numeric_feature_cols = [c for c in numeric_cols if c not in ["label", "age"]]

summary_rows = []

for col in numeric_feature_cols:
    anemia = global_features.loc[global_features["label"] == 1, col].dropna()
    normal = global_features.loc[global_features["label"] == 0, col].dropna()

    summary_rows.append({
        "feature": col,
        "anemia_mean": anemia.mean(),
        "anemia_std": anemia.std(),
        "normal_mean": normal.mean(),
        "normal_std": normal.std(),
        "overall_mean": global_features[col].mean(),
        "overall_std": global_features[col].std(),
    })

feature_summary = pd.DataFrame(summary_rows)
feature_summary.to_csv(
    TABLES_DIR / "global_feature_summary_statistics.csv",
    index=False
)


# ============================================================
# SIMPLE FIGURES
# ============================================================

selected_plot_features = [
    "rgb_global_views_mean_R_mean",
    "rgb_global_views_mean_HSV_S_mean",
    "rgb_global_views_mean_LAB_L_mean",
    "thermal_global_views_mean_thermal_mean",
    "thermal_global_views_mean_thermal_std",
    "thermal_global_views_mean_thermal_entropy",
]

for feature in selected_plot_features:
    if feature not in global_features.columns:
        continue

    plt.figure(figsize=(7, 5))

    anemia = global_features.loc[global_features["label"] == 1, feature].dropna()
    normal = global_features.loc[global_features["label"] == 0, feature].dropna()

    plt.boxplot([anemia, normal], labels=["Anemia", "Normal"])
    plt.ylabel(feature)
    plt.title(f"Distribution of {feature}")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{feature}_boxplot.png", dpi=300)
    plt.close()


# ============================================================
# REPORT
# ============================================================

report = f"""# Stage 5A Global RGB and Thermal Feature Extraction Report

Stage 5A extracted whole-image global features from all RGB and thermal hand views.

## Input

Participant dataset:

`{PARTICIPANT_DATASET}`

## Participants

Participants processed: {len(global_features)}

## Image Views

RGB views per participant: 4

Thermal views per participant: 4

## Output Tables

- participant_global_features.csv
- global_feature_summary_statistics.csv

## Output Figures

Selected boxplots were generated for representative RGB and thermal features.

## Feature Families

RGB global features include channel statistics, grayscale intensity, HSV color statistics, Lab color statistics, edge statistics, entropy, GLCM texture, and pallor-related proxy features.

Thermal global features include intensity statistics, edge statistics, entropy, GLCM texture, hot/cold region ratios, dynamic range, and distributional temperature-map descriptors.

## Interpretation

This stage treats each full image as a global source of measurable information. The extracted features provide a baseline analytical representation before moving to local patch-based features in Stage 5B. These features will later be compared between anemia and normal participants using statistical testing.
"""

with open(
    REPORTS_DIR / "Stage5A_Global_Feature_Extraction_Report.md",
    "w",
    encoding="utf-8"
) as f:
    f.write(report)


print("=" * 80)
print("STAGE 5A GLOBAL FEATURE EXTRACTION COMPLETED")
print("=" * 80)
print(f"Participants processed: {len(global_features)}")
print(f"Global feature table shape: {global_features.shape}")
print(f"Feature summary shape: {feature_summary.shape}")
print(f"Results saved to: {OUTPUT_DIR}")
print("=" * 80)