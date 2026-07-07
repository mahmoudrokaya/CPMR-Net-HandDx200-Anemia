from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path(r"D:\47\472\New-Papers\Anemia_Paper\Data")

OUTPUT_DIR = Path(
    r"D:\47\472\New-Papers\Anemia_Paper\Outputs\Stage4_1_Metadata_Clinical_Variable_Check"
)

TABLES_DIR = OUTPUT_DIR / "tables"
REPORTS_DIR = OUTPUT_DIR / "reports"

for d in [OUTPUT_DIR, TABLES_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# INPUT FILES
# ============================================================

FILES = {
    "Anemia": DATA_DIR / "FILES" / "FINAL_anemia_ml.csv",
    "Normal": DATA_DIR / "FILES" / "FINAL_normal_ml.csv",
}


# ============================================================
# CLINICAL KEYWORDS
# ============================================================

CLINICAL_KEYWORDS = [
    "hb",
    "hgb",
    "hemoglobin",
    "haemoglobin",
    "cbc",
    "rbc",
    "red",
    "hematocrit",
    "haematocrit",
    "hct",
    "mcv",
    "mch",
    "mchc",
    "rdw",
    "wbc",
    "platelet",
    "plt",
    "iron",
    "ferritin",
    "transferrin",
    "b12",
    "folate",
    "folic",
    "glucose",
    "hba1c",
    "cholesterol",
    "triglyceride",
    "hdl",
    "ldl",
    "bmi",
    "weight",
    "height",
    "pressure",
    "bp",
    "systolic",
    "diastolic",
    "spo2",
    "oxygen",
    "temperature"
]


DEMOGRAPHIC_OR_LABEL_COLUMNS = [
    "participant_id",
    "sex",
    "gender",
    "age",
    "label",
    "class",
    "source_file"
]


# ============================================================
# HELPERS
# ============================================================

def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")


def classify_column(column_name):
    c = str(column_name).lower()

    matched_keywords = [
        kw for kw in CLINICAL_KEYWORDS
        if kw in c
    ]

    if matched_keywords:
        return "possible_clinical_variable", "; ".join(matched_keywords)

    if c in DEMOGRAPHIC_OR_LABEL_COLUMNS:
        return "demographic_or_label", ""

    return "other_or_unknown", ""


def summarize_numeric(series):
    x = pd.to_numeric(series, errors="coerce").dropna()

    if len(x) == 0:
        return {
            "numeric_count": 0,
            "mean": None,
            "std": None,
            "median": None,
            "q1": None,
            "q3": None,
            "min": None,
            "max": None,
        }

    return {
        "numeric_count": int(len(x)),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "median": float(x.median()),
        "q1": float(x.quantile(0.25)),
        "q3": float(x.quantile(0.75)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


# ============================================================
# LOAD FILES AND AUDIT COLUMNS
# ============================================================

all_rows = []
clinical_rows = []
combined_frames = []

for group, path in FILES.items():

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = safe_read_csv(path)
    df["source_group"] = group
    combined_frames.append(df)

    for col in df.columns:

        role, matched_keywords = classify_column(col)

        numeric_summary = summarize_numeric(df[col])

        row = {
            "file": path.name,
            "group": group,
            "column": col,
            "dtype": str(df[col].dtype),
            "non_missing": int(df[col].notna().sum()),
            "missing": int(df[col].isna().sum()),
            "missing_percent": round(100 * df[col].isna().sum() / len(df), 2),
            "unique_values": int(df[col].nunique(dropna=True)),
            "classified_role": role,
            "matched_keywords": matched_keywords,
            **numeric_summary
        }

        all_rows.append(row)

        if role == "possible_clinical_variable":
            clinical_rows.append(row)


column_audit = pd.DataFrame(all_rows)
clinical_candidates = pd.DataFrame(clinical_rows)

column_audit.to_csv(
    TABLES_DIR / "metadata_full_column_audit.csv",
    index=False
)

clinical_candidates.to_csv(
    TABLES_DIR / "possible_clinical_variables.csv",
    index=False
)

combined = pd.concat(combined_frames, ignore_index=True, sort=False)
combined.to_csv(TABLES_DIR / "combined_metadata_full.csv", index=False)


# ============================================================
# CHECK WHETHER HEMOGLOBIN-LIKE VARIABLES EXIST
# ============================================================

hb_keywords = ["hb", "hgb", "hemoglobin", "haemoglobin"]

hb_candidates = []

for col in combined.columns:
    c = col.lower()
    if any(k in c for k in hb_keywords):
        hb_candidates.append(col)

hb_df = pd.DataFrame({
    "hemoglobin_candidate_column": hb_candidates
})

hb_df.to_csv(TABLES_DIR / "hemoglobin_candidate_columns.csv", index=False)


# ============================================================
# NON-DEMOGRAPHIC COLUMNS
# ============================================================

non_demo_rows = []

for col in combined.columns:
    c = col.lower()

    if c not in DEMOGRAPHIC_OR_LABEL_COLUMNS and c != "source_group":
        non_demo_rows.append({
            "column": col,
            "dtype": str(combined[col].dtype),
            "non_missing": int(combined[col].notna().sum()),
            "unique_values": int(combined[col].nunique(dropna=True)),
        })

non_demo_df = pd.DataFrame(non_demo_rows)
non_demo_df.to_csv(TABLES_DIR / "non_demographic_columns.csv", index=False)


# ============================================================
# INTERPRETATION FLAGS
# ============================================================

has_hb = len(hb_candidates) > 0
has_clinical = len(clinical_candidates) > 0
has_non_demo = len(non_demo_df) > 0

if has_hb:
    conclusion = (
        "Hemoglobin-like variables were detected. The dataset may support "
        "hemoglobin-based verification, regression analysis, or anemia severity analysis."
    )
elif has_clinical:
    conclusion = (
        "No hemoglobin-like variable was detected, but other possible clinical variables "
        "were identified. The dataset may support limited clinical covariate analysis but "
        "not direct hemoglobin regression unless hemoglobin values are located elsewhere."
    )
elif has_non_demo:
    conclusion = (
        "No hemoglobin-like or obvious clinical variables were detected. Some additional "
        "non-demographic columns exist and should be inspected manually."
    )
else:
    conclusion = (
        "No hemoglobin-like, CBC-related, or additional clinical variables were detected. "
        "The available metadata appear limited to participant ID, sex, age, and class label. "
        "Therefore, the current analysis should be treated as binary anemia classification "
        "rather than hemoglobin regression or severity modeling."
    )


# ============================================================
# WRITE REPORT
# ============================================================

report = f"""# Stage 4.1 Metadata Clinical Variable Check Report

This stage examined the original metadata files to determine whether additional clinical variables are available beyond participant ID, sex, age, and anemia label.

## Files Checked

- `{FILES["Anemia"]}`
- `{FILES["Normal"]}`

## Metadata Dimensions

Anemia file rows: {len(safe_read_csv(FILES["Anemia"]))}

Normal file rows: {len(safe_read_csv(FILES["Normal"]))}

Combined rows: {len(combined)}

Combined columns: {combined.shape[1]}

## Hemoglobin Candidate Columns

Number of hemoglobin-like columns detected: {len(hb_candidates)}

Detected hemoglobin-like columns:

{hb_candidates if hb_candidates else "None"}

## Possible Clinical Variables

Number of possible clinical-variable columns detected: {len(clinical_candidates)}

## Non-Demographic Columns

Number of non-demographic columns detected: {len(non_demo_df)}

## Interpretation

{conclusion}

## Generated Tables

- metadata_full_column_audit.csv
- possible_clinical_variables.csv
- hemoglobin_candidate_columns.csv
- non_demographic_columns.csv
- combined_metadata_full.csv
"""

with open(
    REPORTS_DIR / "Stage4_1_Metadata_Clinical_Variable_Check_Report.md",
    "w",
    encoding="utf-8"
) as f:
    f.write(report)


# ============================================================
# CONSOLE OUTPUT
# ============================================================

print("=" * 80)
print("STAGE 4.1 METADATA CLINICAL VARIABLE CHECK COMPLETED")
print("=" * 80)
print(f"Combined rows: {len(combined)}")
print(f"Combined columns: {combined.shape[1]}")
print(f"Hemoglobin-like columns detected: {len(hb_candidates)}")
print(f"Possible clinical variables detected: {len(clinical_candidates)}")
print(f"Non-demographic columns detected: {len(non_demo_df)}")
print("-" * 80)
print(conclusion)
print("=" * 80)
print(f"Results saved to: {OUTPUT_DIR}")
print("=" * 80)