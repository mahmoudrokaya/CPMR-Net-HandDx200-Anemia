from pathlib import Path
import shutil
import pandas as pd


# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path(r"D:\47\472\New-Papers\Anemia_Paper\Data")

STAGE2_DIR = Path(
    r"D:\47\472\New-Papers\Anemia_Paper\Outputs\Stage2_Thermal_BMP_Decoder"
)

CONVERTED_THERMAL_DIR = STAGE2_DIR / "converted_thermal_png"

THERMAL_AUDIT_CSV = (
    STAGE2_DIR / "tables" / "thermal_bmp_decoding_audit.csv"
)

OUTPUT_DIR = Path(
    r"D:\47\472\New-Papers\Anemia_Paper\Outputs\Stage3_Organize_Thermal_PNG_By_Class"
)

TABLES_DIR = OUTPUT_DIR / "tables"
REPORTS_DIR = OUTPUT_DIR / "reports"

for d in [OUTPUT_DIR, TABLES_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# TARGET FOLDERS
# ============================================================

TARGETS = {
    "Anemia": DATA_DIR / "Anemia" / "Thermal_PNG",
    "Normal": DATA_DIR / "Normal" / "Thermal_PNG",
}

for path in TARGETS.values():
    path.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD THERMAL DECODING AUDIT
# ============================================================

if not THERMAL_AUDIT_CSV.exists():
    raise FileNotFoundError(f"Missing audit file: {THERMAL_AUDIT_CSV}")

df = pd.read_csv(THERMAL_AUDIT_CSV)

required_cols = {"file_name", "label", "decoded", "converted_png"}

missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns in audit CSV: {missing}")

df = df[df["decoded"] == True].copy()


# ============================================================
# COPY CONVERTED THERMAL PNG FILES INTO CLASS FOLDERS
# ============================================================

records = []

for _, row in df.iterrows():

    label = str(row["label"]).strip()

    if label not in TARGETS:
        records.append({
            "file_name": row["file_name"],
            "label": label,
            "status": "skipped_unknown_label",
            "source_png": row["converted_png"],
            "target_png": None
        })
        continue

    source_png = Path(str(row["converted_png"]))

    if not source_png.exists():
        fallback_name = Path(row["file_name"]).stem + "_thermal_norm.png"
        source_png = CONVERTED_THERMAL_DIR / fallback_name

    if not source_png.exists():
        records.append({
            "file_name": row["file_name"],
            "label": label,
            "status": "missing_converted_png",
            "source_png": str(source_png),
            "target_png": None
        })
        continue

    target_png = TARGETS[label] / source_png.name

    shutil.copy2(source_png, target_png)

    records.append({
        "file_name": row["file_name"],
        "label": label,
        "status": "copied",
        "source_png": str(source_png),
        "target_png": str(target_png)
    })


result_df = pd.DataFrame(records)
result_df.to_csv(TABLES_DIR / "thermal_png_organization_log.csv", index=False)


# ============================================================
# VALIDATION COUNTS
# ============================================================

validation = []

for label, folder in TARGETS.items():
    count = len(list(folder.glob("*.png")))

    validation.append({
        "label": label,
        "thermal_png_count": count,
        "target_folder": str(folder)
    })

validation_df = pd.DataFrame(validation)
validation_df.to_csv(TABLES_DIR / "thermal_png_validation_counts.csv", index=False)


# ============================================================
# WRITE REPORT
# ============================================================

copied_count = int((result_df["status"] == "copied").sum())
missing_count = int((result_df["status"] == "missing_converted_png").sum())
skipped_count = int((result_df["status"] == "skipped_unknown_label").sum())

report = f"""# Stage 3 Thermal PNG Organization Report

The converted thermal PNG files generated in Stage 2 were copied into class-specific folders while preserving the original thermal BMP files.

## Source

Converted thermal PNG folder:

`{CONVERTED_THERMAL_DIR}`

Thermal decoding audit file:

`{THERMAL_AUDIT_CSV}`

## Target Structure

`{DATA_DIR}\\Anemia\\Thermal_PNG`

`{DATA_DIR}\\Normal\\Thermal_PNG`

## Results

Total decoded thermal records processed: {len(df)}

Successfully copied: {copied_count}

Missing converted PNG files: {missing_count}

Skipped because of unknown label: {skipped_count}

## Validation Counts

{validation_df.to_markdown(index=False)}

## Interpretation

The original BMP thermal files were preserved in their original folders. The normalized PNG versions were added in separate `Thermal_PNG` folders for downstream machine learning experiments.
"""

with open(REPORTS_DIR / "Stage3_Thermal_PNG_Organization_Report.md", "w", encoding="utf-8") as f:
    f.write(report)


print("=" * 80)
print("STAGE 3 THERMAL PNG ORGANIZATION COMPLETED")
print("=" * 80)
print(f"Successfully copied: {copied_count}")
print(f"Missing converted PNG files: {missing_count}")
print(f"Skipped unknown labels: {skipped_count}")
print(f"Results saved to: {OUTPUT_DIR}")
print("=" * 80)