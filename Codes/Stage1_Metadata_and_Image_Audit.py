import os
import json
import hashlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path(r"D:\47\472\New-Papers\Anemia_Paper\Data")
OUTPUT_DIR = Path(r"D:\47\472\New-Papers\Anemia_Paper\Outputs\Stage1_Metadata_and_Image_Audit")
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"

for d in [OUTPUT_DIR, TABLES_DIR, FIGURES_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIG
# ============================================================

CSV_FILES = [
    DATA_DIR / "FILES" / "FINAL_anemia_ml.csv",
    DATA_DIR / "FILES" / "FINAL_normal_ml.csv",
]

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")


def file_hash(path, block_size=65536):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def infer_modality(path):
    p = str(path).lower()
    if "thermal" in p:
        return "Thermal"
    if "rgb" in p:
        return "RGB"
    if path.suffix.lower() == ".bmp":
        return "Thermal_possible"
    if path.suffix.lower() == ".png":
        return "RGB_possible"
    return "Unknown"


def image_quality_metrics(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    return {
        "mean_intensity": float(np.mean(gray)),
        "std_intensity": float(np.std(gray)),
        "min_intensity": int(np.min(gray)),
        "max_intensity": int(np.max(gray)),
        "blur_laplacian_var": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "dark_pixel_ratio": float(np.mean(gray < 15)),
        "bright_pixel_ratio": float(np.mean(gray > 240)),
    }


# ============================================================
# 1. LOAD AND AUDIT CSV FILES
# ============================================================

metadata_summaries = []
metadata_columns = []
missing_reports = []

loaded_dfs = []

for csv_path in CSV_FILES:
    if not csv_path.exists():
        print(f"Missing CSV: {csv_path}")
        continue

    df = safe_read_csv(csv_path)
    df["source_file"] = csv_path.name
    loaded_dfs.append(df)

    metadata_summaries.append({
        "file": csv_path.name,
        "rows": len(df),
        "columns": df.shape[1],
        "duplicate_rows": int(df.duplicated().sum()),
    })

    for col in df.columns:
        metadata_columns.append({
            "file": csv_path.name,
            "column": col,
            "dtype": str(df[col].dtype),
            "non_null": int(df[col].notna().sum()),
            "missing": int(df[col].isna().sum()),
            "unique_values": int(df[col].nunique(dropna=True)),
        })

    miss = df.isna().sum().reset_index()
    miss.columns = ["column", "missing_count"]
    miss["missing_percent"] = 100 * miss["missing_count"] / len(df)
    miss["file"] = csv_path.name
    missing_reports.append(miss)

pd.DataFrame(metadata_summaries).to_csv(TABLES_DIR / "metadata_file_summary.csv", index=False)
pd.DataFrame(metadata_columns).to_csv(TABLES_DIR / "metadata_columns_audit.csv", index=False)

if missing_reports:
    pd.concat(missing_reports, ignore_index=True).to_csv(TABLES_DIR / "metadata_missing_values.csv", index=False)

if loaded_dfs:
    combined_df = pd.concat(loaded_dfs, ignore_index=True, sort=False)
    combined_df.to_csv(TABLES_DIR / "combined_metadata_preview.csv", index=False)
else:
    combined_df = pd.DataFrame()


# ============================================================
# 2. SEARCH FOR POSSIBLE KEY COLUMNS
# ============================================================

possible_columns = []

if not combined_df.empty:
    for col in combined_df.columns:
        low = col.lower()

        if any(k in low for k in ["id", "participant", "subject", "patient"]):
            role = "possible_participant_id"
        elif any(k in low for k in ["hb", "hemoglobin", "haemoglobin"]):
            role = "possible_hemoglobin"
        elif "sex" in low or "gender" in low:
            role = "possible_sex"
        elif "age" in low:
            role = "possible_age"
        elif any(k in low for k in ["image", "file", "path", "rgb", "thermal"]):
            role = "possible_image_reference"
        else:
            continue

        possible_columns.append({
            "column": col,
            "suggested_role": role,
            "dtype": str(combined_df[col].dtype),
            "non_null": int(combined_df[col].notna().sum()),
            "unique_values": int(combined_df[col].nunique(dropna=True)),
        })

pd.DataFrame(possible_columns).to_csv(TABLES_DIR / "possible_key_columns.csv", index=False)


# ============================================================
# 3. IMAGE INVENTORY AND QUALITY AUDIT
# ============================================================

image_records = []
failed_images = []

image_paths = [
    p for p in DATA_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
]

for img_path in image_paths:
    try:
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)

        if img is None:
            failed_images.append({"path": str(img_path), "reason": "cv2_read_failed"})
            continue

        height, width = img.shape[:2]
        channels = 1 if len(img.shape) == 2 else img.shape[2]

        if channels == 4:
            img_for_quality = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif channels == 1:
            img_for_quality = img
        else:
            img_for_quality = img

        q = image_quality_metrics(img_for_quality)

        image_records.append({
            "file_name": img_path.name,
            "full_path": str(img_path),
            "relative_path": str(img_path.relative_to(DATA_DIR)),
            "folder": str(img_path.parent),
            "extension": img_path.suffix.lower(),
            "modality_inferred": infer_modality(img_path),
            "width": width,
            "height": height,
            "channels": channels,
            "size_mb": round(img_path.stat().st_size / (1024 * 1024), 4),
            "hash_md5": file_hash(img_path),
            **q
        })

    except Exception as e:
        failed_images.append({"path": str(img_path), "reason": str(e)})

image_df = pd.DataFrame(image_records)
failed_df = pd.DataFrame(failed_images)

image_df.to_csv(TABLES_DIR / "image_inventory_quality_audit.csv", index=False)
failed_df.to_csv(TABLES_DIR / "failed_images.csv", index=False)


# ============================================================
# 4. DUPLICATE IMAGE DETECTION
# ============================================================

if not image_df.empty:
    duplicate_hashes = (
        image_df.groupby("hash_md5")
        .filter(lambda x: len(x) > 1)
        .sort_values("hash_md5")
    )
    duplicate_hashes.to_csv(TABLES_DIR / "duplicate_images_by_hash.csv", index=False)


# ============================================================
# 5. IMAGE SUMMARY TABLES
# ============================================================

if not image_df.empty:
    modality_summary = image_df.groupby("modality_inferred").agg(
        image_count=("file_name", "count"),
        mean_width=("width", "mean"),
        mean_height=("height", "mean"),
        min_width=("width", "min"),
        max_width=("width", "max"),
        min_height=("height", "min"),
        max_height=("height", "max"),
        mean_size_mb=("size_mb", "mean"),
        mean_blur=("blur_laplacian_var", "mean"),
        mean_intensity=("mean_intensity", "mean"),
        mean_std_intensity=("std_intensity", "mean")
    ).reset_index()

    modality_summary.to_csv(TABLES_DIR / "image_modality_summary.csv", index=False)

    resolution_summary = image_df.groupby(
        ["modality_inferred", "width", "height", "channels"]
    ).size().reset_index(name="count")

    resolution_summary.to_csv(TABLES_DIR / "image_resolution_summary.csv", index=False)


# ============================================================
# 6. FIGURES
# ============================================================

if not image_df.empty:
    plt.figure(figsize=(8, 5))
    image_df["modality_inferred"].value_counts().plot(kind="bar")
    plt.title("Image Count by Inferred Modality")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "image_count_by_modality.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(image_df["width"], bins=30)
    plt.title("Image Width Distribution")
    plt.xlabel("Width")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "image_width_distribution.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(image_df["height"], bins=30)
    plt.title("Image Height Distribution")
    plt.xlabel("Height")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "image_height_distribution.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(image_df["blur_laplacian_var"], bins=40)
    plt.title("Blur Score Distribution")
    plt.xlabel("Laplacian Variance")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "blur_score_distribution.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(image_df["mean_intensity"], bins=40)
    plt.title("Mean Intensity Distribution")
    plt.xlabel("Mean Intensity")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "mean_intensity_distribution.png", dpi=300)
    plt.close()


# ============================================================
# 7. SAMPLE IMAGE CONTACT SHEETS
# ============================================================

def make_contact_sheet(df, modality, output_name, max_images=16):
    subset = df[df["modality_inferred"].str.contains(modality, case=False, na=False)].head(max_images)

    if subset.empty:
        return

    thumbs = []

    for _, row in subset.iterrows():
        img = cv2.imread(row["full_path"], cv2.IMREAD_UNCHANGED)

        if img is None:
            continue

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (180, 180))
        thumbs.append(img)

    if not thumbs:
        return

    cols = 4
    rows = int(np.ceil(len(thumbs) / cols))
    canvas = np.ones((rows * 180, cols * 180, 3), dtype=np.uint8) * 255

    for i, img in enumerate(thumbs):
        r = i // cols
        c = i % cols
        canvas[r*180:(r+1)*180, c*180:(c+1)*180, :] = img

    plt.figure(figsize=(8, 8))
    plt.imshow(canvas)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / output_name, dpi=300)
    plt.close()


if not image_df.empty:
    make_contact_sheet(image_df, "RGB", "rgb_sample_contact_sheet.png")
    make_contact_sheet(image_df, "Thermal", "thermal_sample_contact_sheet.png")


# ============================================================
# 8. BASIC DATA CONSISTENCY REPORT
# ============================================================

report_lines = []

report_lines.append("# Stage 1 Metadata and Image Audit Report\n")
report_lines.append(f"Data folder: `{DATA_DIR}`\n")
report_lines.append(f"Output folder: `{OUTPUT_DIR}`\n")

report_lines.append("\n## Metadata Files\n")
if metadata_summaries:
    for item in metadata_summaries:
        report_lines.append(
            f"- `{item['file']}`: {item['rows']} rows, {item['columns']} columns, "
            f"{item['duplicate_rows']} duplicate rows."
        )
else:
    report_lines.append("- No metadata CSV files were loaded.")

report_lines.append("\n## Possible Key Columns\n")
if possible_columns:
    for item in possible_columns:
        report_lines.append(
            f"- `{item['column']}` → {item['suggested_role']} "
            f"({item['unique_values']} unique values)"
        )
else:
    report_lines.append("- No obvious participant, hemoglobin, sex, age, or image-reference columns were detected.")

report_lines.append("\n## Image Audit Summary\n")
report_lines.append(f"- Total readable images: {len(image_df)}")
report_lines.append(f"- Failed images: {len(failed_df)}")

if not image_df.empty:
    counts = image_df["modality_inferred"].value_counts()
    for modality, count in counts.items():
        report_lines.append(f"- {modality}: {count} images")

    report_lines.append("\n## Image Resolution Summary\n")
    for _, row in resolution_summary.iterrows():
        report_lines.append(
            f"- {row['modality_inferred']}: {row['width']} × {row['height']} × "
            f"{row['channels']} channels = {row['count']} images"
        )

    duplicate_count = len(duplicate_hashes) if "duplicate_hashes" in locals() else 0
    report_lines.append(f"\n## Duplicate Image Check\n")
    report_lines.append(f"- Duplicate images by MD5 hash: {duplicate_count}")

report_lines.append("\n## Important Interpretation\n")
report_lines.append(
    "This audit should be reviewed before model training. If the number of RGB and thermal "
    "images does not match the expected dataset description, the current folder may represent "
    "an incomplete subset or a differently organized release. Training should not begin until "
    "the metadata-to-image mapping and participant-level grouping are verified."
)

with open(REPORTS_DIR / "Stage1_Metadata_and_Image_Audit_Report.md", "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))


# ============================================================
# 9. SAVE JSON SUMMARY
# ============================================================

summary = {
    "data_folder": str(DATA_DIR),
    "output_folder": str(OUTPUT_DIR),
    "metadata_files_loaded": metadata_summaries,
    "total_images": int(len(image_df)),
    "failed_images": int(len(failed_df)),
    "modalities": image_df["modality_inferred"].value_counts().to_dict() if not image_df.empty else {},
}

with open(OUTPUT_DIR / "stage1_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4)


print("=" * 80)
print("STAGE 1 METADATA AND IMAGE AUDIT COMPLETED")
print("=" * 80)
print(f"Results saved to: {OUTPUT_DIR}")
print(f"Main report: {REPORTS_DIR / 'Stage1_Metadata_and_Image_Audit_Report.md'}")
print("=" * 80)