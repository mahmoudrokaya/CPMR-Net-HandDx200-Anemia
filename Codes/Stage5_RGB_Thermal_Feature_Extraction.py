import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import entropy
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2hsv, rgb2lab
from skimage.filters import sobel


# ============================================================
# PATHS
# ============================================================

STAGE4_DIR = Path(
    r"D:\47\472\New-Papers\Anemia_Paper\Outputs\Stage4_Dataset_Characterization_and_Inventory"
)

PARTICIPANT_DATASET = STAGE4_DIR / "tables" / "participant_dataset.csv"

OUTPUT_DIR = Path(
    r"D:\47\472\New-Papers\Anemia_Paper\Outputs\Stage5_RGB_Thermal_Feature_Extraction"
)

TABLES_DIR = OUTPUT_DIR / "tables"
REPORTS_DIR = OUTPUT_DIR / "reports"

for d in [OUTPUT_DIR, TABLES_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIG
# ============================================================

VIEWS = [
    "l_dorsal",
    "l_palmar",
    "r_dorsal",
    "r_palmar"
]

RGB_COLUMNS = [f"rgb_{v}" for v in VIEWS]
THERMAL_COLUMNS = [f"thermal_{v}" for v in VIEWS]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_rgb(path, resize=(224, 224)):
    img = Image.open(path).convert("RGB")
    img = img.resize(resize)
    return np.asarray(img).astype(np.float32) / 255.0


def load_gray(path, resize=(224, 224)):
    img = Image.open(path).convert("L")
    img = img.resize(resize)
    return np.asarray(img).astype(np.float32) / 255.0


def image_entropy(arr, bins=256):
    hist, _ = np.histogram(arr.flatten(), bins=bins, range=(0, 1), density=True)
    hist = hist + 1e-12
    return float(entropy(hist))


def basic_stats(arr, prefix):
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_var": float(np.var(arr)),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_max": float(np.max(arr)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_q25": float(np.quantile(arr, 0.25)),
        f"{prefix}_q75": float(np.quantile(arr, 0.75)),
        f"{prefix}_entropy": image_entropy(arr),
    }


def glcm_texture_features(gray, prefix):
    gray_uint8 = np.clip(gray * 255, 0, 255).astype(np.uint8)

    glcm = graycomatrix(
        gray_uint8,
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


def rgb_features(img, prefix):
    feats = {}

    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    feats.update(basic_stats(r, f"{prefix}_R"))
    feats.update(basic_stats(g, f"{prefix}_G"))
    feats.update(basic_stats(b, f"{prefix}_B"))

    feats[f"{prefix}_red_green_ratio"] = float(np.mean(r) / (np.mean(g) + 1e-8))
    feats[f"{prefix}_red_blue_ratio"] = float(np.mean(r) / (np.mean(b) + 1e-8))
    feats[f"{prefix}_green_blue_ratio"] = float(np.mean(g) / (np.mean(b) + 1e-8))

    hsv = rgb2hsv(img)
    lab = rgb2lab(img)

    feats.update(basic_stats(hsv[:, :, 0], f"{prefix}_HSV_H"))
    feats.update(basic_stats(hsv[:, :, 1], f"{prefix}_HSV_S"))
    feats.update(basic_stats(hsv[:, :, 2], f"{prefix}_HSV_V"))

    feats.update(basic_stats(lab[:, :, 0], f"{prefix}_LAB_L"))
    feats.update(basic_stats(lab[:, :, 1], f"{prefix}_LAB_A"))
    feats.update(basic_stats(lab[:, :, 2], f"{prefix}_LAB_B"))

    gray = np.mean(img, axis=2)
    edge = sobel(gray)

    feats.update(basic_stats(gray, f"{prefix}_gray"))
    feats.update(basic_stats(edge, f"{prefix}_edge"))
    feats.update(glcm_texture_features(gray, f"{prefix}_texture"))

    return feats


def thermal_features(gray, prefix):
    feats = {}

    feats.update(basic_stats(gray, f"{prefix}_thermal"))

    edge = sobel(gray)
    feats.update(basic_stats(edge, f"{prefix}_thermal_edge"))

    feats.update(glcm_texture_features(gray, f"{prefix}_thermal_texture"))

    hot_threshold = np.quantile(gray, 0.90)
    cold_threshold = np.quantile(gray, 0.10)

    feats[f"{prefix}_hot_region_ratio"] = float(np.mean(gray >= hot_threshold))
    feats[f"{prefix}_cold_region_ratio"] = float(np.mean(gray <= cold_threshold))
    feats[f"{prefix}_thermal_dynamic_range"] = float(np.max(gray) - np.min(gray))

    return feats


def aggregate_participant_features(row, feature_type):
    features = {
        "participant_id": row["participant_id"],
        "label": row["label"],
        "class_name": row["class_name"],
        "sex": row.get("sex", None),
        "age": row.get("age", None),
    }

    view_feature_dicts = []

    if feature_type == "rgb":
        columns = RGB_COLUMNS
        loader = load_rgb
        extractor = rgb_features
    else:
        columns = THERMAL_COLUMNS
        loader = load_gray
        extractor = thermal_features

    for col in columns:
        view = col.replace("rgb_", "").replace("thermal_", "")
        path = row[col]

        img = loader(path)
        feats = extractor(img, f"{feature_type}_{view}")
        features.update(feats)
        view_feature_dicts.append(feats)

    # Participant-level aggregation across four views
    all_keys = sorted(view_feature_dicts[0].keys())

    for key in all_keys:
        values = [d[key] for d in view_feature_dicts if key in d]
        base_key = key.replace(f"{feature_type}_l_dorsal_", "")
        base_key = base_key.replace(f"{feature_type}_l_palmar_", "")
        base_key = base_key.replace(f"{feature_type}_r_dorsal_", "")
        base_key = base_key.replace(f"{feature_type}_r_palmar_", "")

        features[f"{feature_type}_views_mean_{base_key}"] = float(np.mean(values))
        features[f"{feature_type}_views_std_{base_key}"] = float(np.std(values))
        features[f"{feature_type}_views_min_{base_key}"] = float(np.min(values))
        features[f"{feature_type}_views_max_{base_key}"] = float(np.max(values))

    return features


# ============================================================
# MAIN
# ============================================================

df = pd.read_csv(PARTICIPANT_DATASET)

rgb_rows = []
thermal_rows = []

for idx, row in df.iterrows():
    print(f"Processing participant {idx + 1}/{len(df)}: {row['participant_id']}")

    rgb_rows.append(
        aggregate_participant_features(row, "rgb")
    )

    thermal_rows.append(
        aggregate_participant_features(row, "thermal")
    )

rgb_features_df = pd.DataFrame(rgb_rows)
thermal_features_df = pd.DataFrame(thermal_rows)

combined_features = rgb_features_df.merge(
    thermal_features_df,
    on=["participant_id", "label", "class_name", "sex", "age"],
    how="inner"
)

rgb_features_df.to_csv(TABLES_DIR / "participant_rgb_features.csv", index=False)
thermal_features_df.to_csv(TABLES_DIR / "participant_thermal_features.csv", index=False)
combined_features.to_csv(TABLES_DIR / "participant_combined_features.csv", index=False)


# ============================================================
# FEATURE SUMMARY
# ============================================================

numeric_cols = combined_features.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in ["label", "age"]]

summary_rows = []

for col in numeric_cols:
    anemia_values = combined_features.loc[combined_features["label"] == 1, col].dropna()
    normal_values = combined_features.loc[combined_features["label"] == 0, col].dropna()

    summary_rows.append({
        "feature": col,
        "anemia_mean": anemia_values.mean(),
        "anemia_std": anemia_values.std(),
        "normal_mean": normal_values.mean(),
        "normal_std": normal_values.std(),
        "overall_mean": combined_features[col].mean(),
        "overall_std": combined_features[col].std(),
    })

feature_summary = pd.DataFrame(summary_rows)
feature_summary.to_csv(TABLES_DIR / "feature_summary_statistics.csv", index=False)


# ============================================================
# REPORT
# ============================================================

report = f"""# Stage 5 RGB and Thermal Feature Extraction Report

Stage 5 extracted participant-level handcrafted features from RGB and thermal hand images.

## Input

Participant dataset:

`{PARTICIPANT_DATASET}`

## Participants

Total participants processed: {len(df)}

## RGB Features

Output file:

`participant_rgb_features.csv`

Number of RGB feature columns: {rgb_features_df.shape[1]}

## Thermal Features

Output file:

`participant_thermal_features.csv`

Number of thermal feature columns: {thermal_features_df.shape[1]}

## Combined Features

Output file:

`participant_combined_features.csv`

Combined feature table shape:

{combined_features.shape[0]} rows × {combined_features.shape[1]} columns

## Extracted Feature Families

RGB features included channel statistics, HSV features, Lab color features, grayscale intensity, edge features, entropy, and GLCM texture features.

Thermal features included intensity statistics, entropy, thermal edge features, hot/cold region ratios, dynamic range, and GLCM texture features.

## Interpretation

This stage prepares measurable image-derived features for statistical testing. Stage 6 should compare anemia and normal groups using t-test, Mann–Whitney U test, and Cohen's d to determine whether measurable visual or thermal signals differ before deep learning.
"""

with open(REPORTS_DIR / "Stage5_RGB_Thermal_Feature_Extraction_Report.md", "w", encoding="utf-8") as f:
    f.write(report)


print("=" * 80)
print("STAGE 5 RGB AND THERMAL FEATURE EXTRACTION COMPLETED")
print("=" * 80)
print(f"Participants processed: {len(df)}")
print(f"RGB feature table: {rgb_features_df.shape}")
print(f"Thermal feature table: {thermal_features_df.shape}")
print(f"Combined feature table: {combined_features.shape}")
print(f"Results saved to: {OUTPUT_DIR}")
print("=" * 80)