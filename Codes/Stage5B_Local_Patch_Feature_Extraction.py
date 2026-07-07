from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

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
    r"D:\47\472\New-Papers\Anemia_Paper\Outputs\Stage5B_Local_Patch_Feature_Extraction"
)

TABLES_DIR = OUTPUT_DIR / "tables"
REPORTS_DIR = OUTPUT_DIR / "reports"

for d in [OUTPUT_DIR, TABLES_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIG
# ============================================================

IMAGE_SIZE = (224, 224)
GRID_SIZE = 3

VIEWS = [
    "l_dorsal",
    "l_palmar",
    "r_dorsal",
    "r_palmar"
]

RGB_COLUMNS = [f"rgb_{v}" for v in VIEWS]
THERMAL_COLUMNS = [f"thermal_{v}" for v in VIEWS]


# ============================================================
# IMAGE LOADERS
# ============================================================

def load_rgb(path):
    img = Image.open(path).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    return np.asarray(img).astype(np.float32) / 255.0


def load_gray(path):
    img = Image.open(path).convert("L")
    img = img.resize(IMAGE_SIZE)
    return np.asarray(img).astype(np.float32) / 255.0


# ============================================================
# FEATURE HELPERS
# ============================================================

def safe_entropy(arr, bins=128):
    arr = np.asarray(arr).astype(np.float32).ravel()
    arr = arr[np.isfinite(arr)]

    if len(arr) == 0:
        return np.nan

    hist, _ = np.histogram(arr, bins=bins, range=(0, 1), density=False)
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


def split_into_grid(arr, grid_size=3):
    h, w = arr.shape[:2]

    patch_h = h // grid_size
    patch_w = w // grid_size

    patches = []

    patch_id = 1

    for r in range(grid_size):
        for c in range(grid_size):

            y1 = r * patch_h
            y2 = (r + 1) * patch_h if r < grid_size - 1 else h

            x1 = c * patch_w
            x2 = (c + 1) * patch_w if c < grid_size - 1 else w

            patch = arr[y1:y2, x1:x2]

            patches.append({
                "patch_id": patch_id,
                "row": r + 1,
                "col": c + 1,
                "patch": patch
            })

            patch_id += 1

    return patches


# ============================================================
# RGB PATCH FEATURES
# ============================================================

def extract_rgb_patch_features(patch, prefix):
    feats = {}

    r = patch[:, :, 0]
    g = patch[:, :, 1]
    b = patch[:, :, 2]

    gray = np.mean(patch, axis=2)

    feats.update(basic_stats(r, f"{prefix}_R"))
    feats.update(basic_stats(g, f"{prefix}_G"))
    feats.update(basic_stats(b, f"{prefix}_B"))
    feats.update(basic_stats(gray, f"{prefix}_gray"))

    feats[f"{prefix}_R_G_ratio"] = float(np.mean(r) / (np.mean(g) + 1e-8))
    feats[f"{prefix}_R_B_ratio"] = float(np.mean(r) / (np.mean(b) + 1e-8))
    feats[f"{prefix}_G_B_ratio"] = float(np.mean(g) / (np.mean(b) + 1e-8))

    feats[f"{prefix}_R_minus_G"] = float(np.mean(r) - np.mean(g))
    feats[f"{prefix}_R_minus_B"] = float(np.mean(r) - np.mean(b))
    feats[f"{prefix}_G_minus_B"] = float(np.mean(g) - np.mean(b))

    hsv = rgb2hsv(patch)
    lab = rgb2lab(patch)

    feats.update(basic_stats(hsv[:, :, 0], f"{prefix}_HSV_H"))
    feats.update(basic_stats(hsv[:, :, 1], f"{prefix}_HSV_S"))
    feats.update(basic_stats(hsv[:, :, 2], f"{prefix}_HSV_V"))

    feats.update(basic_stats(lab[:, :, 0], f"{prefix}_LAB_L"))
    feats.update(basic_stats(lab[:, :, 1], f"{prefix}_LAB_A"))
    feats.update(basic_stats(lab[:, :, 2], f"{prefix}_LAB_B"))

    edge = sobel(gray)

    feats.update(basic_stats(edge, f"{prefix}_edge"))
    feats.update(glcm_features(gray, f"{prefix}_texture"))

    feats[f"{prefix}_low_red_ratio"] = float(np.mean(r < np.quantile(r, 0.25)))
    feats[f"{prefix}_high_lightness_ratio"] = float(np.mean(lab[:, :, 0] > np.quantile(lab[:, :, 0], 0.75)))
    feats[f"{prefix}_low_saturation_ratio"] = float(np.mean(hsv[:, :, 1] < np.quantile(hsv[:, :, 1], 0.25)))

    return feats


# ============================================================
# THERMAL PATCH FEATURES
# ============================================================

def extract_thermal_patch_features(patch, prefix):
    feats = {}

    feats.update(basic_stats(patch, f"{prefix}_thermal"))

    edge = sobel(patch)

    feats.update(basic_stats(edge, f"{prefix}_thermal_edge"))
    feats.update(glcm_features(patch, f"{prefix}_thermal_texture"))

    q10 = np.quantile(patch, 0.10)
    q25 = np.quantile(patch, 0.25)
    q75 = np.quantile(patch, 0.75)
    q90 = np.quantile(patch, 0.90)

    feats[f"{prefix}_thermal_q10"] = float(q10)
    feats[f"{prefix}_thermal_q90"] = float(q90)
    feats[f"{prefix}_thermal_iqr"] = float(q75 - q25)
    feats[f"{prefix}_thermal_hot_region_ratio"] = float(np.mean(patch >= q90))
    feats[f"{prefix}_thermal_cold_region_ratio"] = float(np.mean(patch <= q10))
    feats[f"{prefix}_thermal_midrange_ratio"] = float(np.mean((patch > q25) & (patch < q75)))

    return feats


# ============================================================
# AGGREGATION
# ============================================================

def aggregate_patch_feature_rows(feature_rows, modality):
    """
    Aggregates patch features across all views and patches per participant.

    Produces:
    - overall mean/std/min/max for each feature family
    - patch-position mean features across views
    """
    aggregated = {}

    if not feature_rows:
        return aggregated

    df = pd.DataFrame(feature_rows)

    metadata_cols = [
        "participant_id",
        "label",
        "class_name",
        "sex",
        "age",
        "modality",
        "view",
        "patch_id",
        "patch_row",
        "patch_col"
    ]

    feature_cols = [
        c for c in df.columns
        if c not in metadata_cols
    ]

    # Overall participant-level patch aggregation
    for col in feature_cols:
        values = pd.to_numeric(df[col], errors="coerce").dropna()

        if len(values) == 0:
            continue

        clean_col = col

        for view in VIEWS:
            for pid in range(1, GRID_SIZE * GRID_SIZE + 1):
                clean_col = clean_col.replace(
                    f"{modality}_{view}_patch{pid}_",
                    ""
                )

        aggregated[f"{modality}_patch_all_mean_{clean_col}"] = float(values.mean())
        aggregated[f"{modality}_patch_all_std_{clean_col}"] = float(values.std())
        aggregated[f"{modality}_patch_all_min_{clean_col}"] = float(values.min())
        aggregated[f"{modality}_patch_all_max_{clean_col}"] = float(values.max())

    # Patch-position aggregation across views
    for patch_id in range(1, GRID_SIZE * GRID_SIZE + 1):
        patch_df = df[df["patch_id"] == patch_id]

        for col in feature_cols:
            values = pd.to_numeric(patch_df[col], errors="coerce").dropna()

            if len(values) == 0:
                continue

            clean_col = col

            for view in VIEWS:
                clean_col = clean_col.replace(
                    f"{modality}_{view}_patch{patch_id}_",
                    ""
                )

            aggregated[f"{modality}_patch{patch_id}_views_mean_{clean_col}"] = float(values.mean())
            aggregated[f"{modality}_patch{patch_id}_views_std_{clean_col}"] = float(values.std())

    return aggregated


# ============================================================
# PARTICIPANT PROCESSING
# ============================================================

def process_participant(row):
    participant_id = row["participant_id"]

    base = {
        "participant_id": participant_id,
        "label": int(row["label"]),
        "class_name": row["class_name"],
        "sex": row.get("sex", None),
        "age": row.get("age", None),
    }

    rgb_patch_rows = []
    thermal_patch_rows = []

    # RGB patches
    for view in VIEWS:
        path = row[f"rgb_{view}"]
        img = load_rgb(path)

        patches = split_into_grid(img, GRID_SIZE)

        for p in patches:
            prefix = f"rgb_{view}_patch{p['patch_id']}"

            feats = extract_rgb_patch_features(
                p["patch"],
                prefix
            )

            patch_row = {
                "participant_id": participant_id,
                "label": int(row["label"]),
                "class_name": row["class_name"],
                "sex": row.get("sex", None),
                "age": row.get("age", None),
                "modality": "RGB",
                "view": view,
                "patch_id": p["patch_id"],
                "patch_row": p["row"],
                "patch_col": p["col"],
                **feats
            }

            rgb_patch_rows.append(patch_row)

    # Thermal patches
    for view in VIEWS:
        path = row[f"thermal_{view}"]
        img = load_gray(path)

        patches = split_into_grid(img, GRID_SIZE)

        for p in patches:
            prefix = f"thermal_{view}_patch{p['patch_id']}"

            feats = extract_thermal_patch_features(
                p["patch"],
                prefix
            )

            patch_row = {
                "participant_id": participant_id,
                "label": int(row["label"]),
                "class_name": row["class_name"],
                "sex": row.get("sex", None),
                "age": row.get("age", None),
                "modality": "Thermal",
                "view": view,
                "patch_id": p["patch_id"],
                "patch_row": p["row"],
                "patch_col": p["col"],
                **feats
            }

            thermal_patch_rows.append(patch_row)

    participant_features = {
        **base,
        **aggregate_patch_feature_rows(rgb_patch_rows, "rgb"),
        **aggregate_patch_feature_rows(thermal_patch_rows, "thermal"),
    }

    return participant_features, rgb_patch_rows, thermal_patch_rows


# ============================================================
# MAIN
# ============================================================

df = pd.read_csv(PARTICIPANT_DATASET)

required_cols = [
    "participant_id",
    "label",
    "class_name",
    "sex",
    "age",
    *RGB_COLUMNS,
    *THERMAL_COLUMNS,
]

missing_cols = [
    c for c in required_cols
    if c not in df.columns
]

if missing_cols:
    raise ValueError(f"Missing required columns in participant_dataset.csv: {missing_cols}")

participant_feature_rows = []
all_rgb_patch_rows = []
all_thermal_patch_rows = []

for i, row in df.iterrows():
    print(f"Processing participant {i + 1}/{len(df)}: {row['participant_id']}")

    participant_features, rgb_patch_rows, thermal_patch_rows = process_participant(row)

    participant_feature_rows.append(participant_features)
    all_rgb_patch_rows.extend(rgb_patch_rows)
    all_thermal_patch_rows.extend(thermal_patch_rows)


participant_patch_features = pd.DataFrame(participant_feature_rows)
rgb_patch_long = pd.DataFrame(all_rgb_patch_rows)
thermal_patch_long = pd.DataFrame(all_thermal_patch_rows)

participant_patch_features.to_csv(
    TABLES_DIR / "participant_patch_features.csv",
    index=False
)

rgb_patch_long.to_csv(
    TABLES_DIR / "rgb_patch_features_long.csv",
    index=False
)

thermal_patch_long.to_csv(
    TABLES_DIR / "thermal_patch_features_long.csv",
    index=False
)


# ============================================================
# SUMMARY STATISTICS
# ============================================================

numeric_cols = participant_patch_features.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in ["label", "age"]]

summary_rows = []

for col in numeric_cols:
    anemia = participant_patch_features.loc[
        participant_patch_features["label"] == 1,
        col
    ].dropna()

    normal = participant_patch_features.loc[
        participant_patch_features["label"] == 0,
        col
    ].dropna()

    summary_rows.append({
        "feature": col,
        "anemia_mean": anemia.mean(),
        "anemia_std": anemia.std(),
        "normal_mean": normal.mean(),
        "normal_std": normal.std(),
        "overall_mean": participant_patch_features[col].mean(),
        "overall_std": participant_patch_features[col].std(),
    })

summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(
    TABLES_DIR / "patch_feature_summary_statistics.csv",
    index=False
)


# ============================================================
# REPORT
# ============================================================

total_images_processed = len(df) * 8
patches_per_image = GRID_SIZE * GRID_SIZE
patches_per_participant = 8 * patches_per_image
total_patches = len(df) * patches_per_participant

report = f"""# Stage 5B Local Patch Feature Extraction Report

Stage 5B extracted local patch-level features from RGB and thermal hand images.

## Input

Participant dataset:

`{PARTICIPANT_DATASET}`

## Participants

Participants processed: {len(df)}

## Patch Design

Each image was resized to {IMAGE_SIZE[0]} × {IMAGE_SIZE[1]} pixels.

Each image was split into a {GRID_SIZE} × {GRID_SIZE} grid.

Patches per image: {patches_per_image}

Images per participant: 8

Patches per participant: {patches_per_participant}

Total expected patches: {total_patches}

## Output Tables

- participant_patch_features.csv
- rgb_patch_features_long.csv
- thermal_patch_features_long.csv
- patch_feature_summary_statistics.csv

## Extracted Feature Families

RGB patch features include channel statistics, HSV statistics, Lab statistics, grayscale intensity, entropy, edge strength, texture features, and pallor-related proxy measures.

Thermal patch features include intensity statistics, entropy, edge strength, texture features, hot/cold region ratios, interquartile range, and dynamic thermal descriptors.

## Interpretation

This stage treats each image as a source of multiple local regions rather than as a single sample. The resulting features preserve local spatial information that may be hidden by whole-image averages. These features are intended for statistical comparison and feature ranking before model development.
"""

with open(
    REPORTS_DIR / "Stage5B_Local_Patch_Feature_Extraction_Report.md",
    "w",
    encoding="utf-8"
) as f:
    f.write(report)


# ============================================================
# CONSOLE OUTPUT
# ============================================================

print("=" * 80)
print("STAGE 5B LOCAL PATCH FEATURE EXTRACTION COMPLETED")
print("=" * 80)
print(f"Participants processed: {len(df)}")
print(f"Images processed: {total_images_processed}")
print(f"Patches per image: {patches_per_image}")
print(f"Patches per participant: {patches_per_participant}")
print(f"Total patches expected: {total_patches}")
print(f"RGB patch rows: {len(rgb_patch_long)}")
print(f"Thermal patch rows: {len(thermal_patch_long)}")
print(f"Participant patch feature table shape: {participant_patch_features.shape}")
print(f"Results saved to: {OUTPUT_DIR}")
print("=" * 80)