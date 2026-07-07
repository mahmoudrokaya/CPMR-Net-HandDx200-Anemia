import re
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps


# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path(r"D:\47\472\New-Papers\Anemia_Paper\Data")

OUTPUT_DIR = Path(
    r"D:\47\472\New-Papers\Anemia_Paper\Outputs\Stage4_Dataset_Characterization_and_Inventory"
)

TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"

for d in [OUTPUT_DIR, TABLES_DIR, FIGURES_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIG
# ============================================================

EXPECTED_VIEWS = [
    "L_dorsal",
    "L_palmar",
    "R_dorsal",
    "R_palmar"
]

CLASS_INFO = {
    "Anemia": {
        "label": 1,
        "metadata_file": DATA_DIR / "FILES" / "FINAL_anemia_ml.csv",
        "rgb_dir": DATA_DIR / "Anemia" / "RGB",
        "thermal_dir": DATA_DIR / "Anemia" / "Thermal_PNG",
    },
    "Normal": {
        "label": 0,
        "metadata_file": DATA_DIR / "FILES" / "FINAL_normal_ml.csv",
        "rgb_dir": DATA_DIR / "Normal" / "RGB",
        "thermal_dir": DATA_DIR / "Normal" / "Thermal_PNG",
    }
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_pid(x):
    """
    Normalize participant IDs to P001-style format when possible.
    """
    x = str(x).strip()
    match = re.search(r"P\s*0*\d+", x, flags=re.IGNORECASE)

    if match:
        digits = re.findall(r"\d+", match.group(0))[0]
        return f"P{int(digits):03d}"

    digits = re.findall(r"\d+", x)
    if digits:
        return f"P{int(digits[0]):03d}"

    return x.upper()


def extract_pid_from_filename(filename):
    """
    Extract participant ID from image filename.
    Handles names such as:
    P001_RGB_P001_L_dorsal_processed.png
    P001_L_dorsal.png
    patient_001_left_palmar.png
    """
    name = Path(filename).stem

    match = re.search(r"P\s*0*\d+", name, flags=re.IGNORECASE)

    if match:
        digits = re.findall(r"\d+", match.group(0))[0]
        return f"P{int(digits):03d}"

    digits = re.findall(r"\d+", name)
    if digits:
        return f"P{int(digits[0]):03d}"

    return None


def clean_filename_text(filename):
    text = Path(filename).stem.lower()

    text = text.replace("-", "_")
    text = text.replace(" ", "_")
    text = text.replace(".", "_")
    text = text.replace("__", "_")

    while "__" in text:
        text = text.replace("__", "_")

    return text


def extract_view_from_filename(filename):
    """
    Robust view extraction.

    Expected views:
    L_dorsal, L_palmar, R_dorsal, R_palmar

    Supports several common naming styles:
    L_dorsal, left_dorsal, ldorsal, leftdorsal
    L_palmar, left_palmar, lpalmar, leftpalm
    R_dorsal, right_dorsal, rdorsal, rightback
    R_palmar, right_palmar, rpalmar, rightpalm
    """
    clean = clean_filename_text(filename)

    view_patterns = {
        "L_dorsal": [
            "l_dorsal",
            "left_dorsal",
            "ldorsal",
            "leftdorsal",
            "l_back",
            "left_back",
            "left_hand_dorsal",
            "left_dorsum",
            "l_dorsum",
        ],
        "L_palmar": [
            "l_palmar",
            "left_palmar",
            "lpalmar",
            "leftpalmar",
            "l_palm",
            "left_palm",
            "left_hand_palmar",
            "left_palma",
            "l_palma",
        ],
        "R_dorsal": [
            "r_dorsal",
            "right_dorsal",
            "rdorsal",
            "rightdorsal",
            "r_back",
            "right_back",
            "right_hand_dorsal",
            "right_dorsum",
            "r_dorsum",
        ],
        "R_palmar": [
            "r_palmar",
            "right_palmar",
            "rpalmar",
            "rightpalmar",
            "r_palm",
            "right_palm",
            "right_hand_palmar",
            "right_palma",
            "r_palma",
        ],
    }

    for view, patterns in view_patterns.items():
        for pattern in patterns:
            if pattern in clean:
                return view

    # Fallback token logic
    tokens = clean.split("_")

    has_left = any(t in tokens for t in ["l", "left"])
    has_right = any(t in tokens for t in ["r", "right"])
    has_dorsal = any(t in tokens for t in ["dorsal", "dorsum", "back"])
    has_palmar = any(t in tokens for t in ["palmar", "palm", "palma"])

    if has_left and has_dorsal:
        return "L_dorsal"
    if has_left and has_palmar:
        return "L_palmar"
    if has_right and has_dorsal:
        return "R_dorsal"
    if has_right and has_palmar:
        return "R_palmar"

    return None


def standardize_sex(x):
    s = str(x).strip().lower()

    if s in ["m", "male", "ذكر", "1"]:
        return "Male"

    if s in ["f", "female", "أنثى", "انثى", "0"]:
        return "Female"

    return str(x).strip()


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def safe_image_open(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def make_contact_sheet(image_paths, output_path, title, max_images=16, thumb_size=(220, 220)):
    image_paths = [Path(p) for p in image_paths if pd.notna(p) and Path(p).exists()]
    image_paths = image_paths[:max_images]

    if not image_paths:
        return

    cols = 4
    rows = math.ceil(len(image_paths) / cols)

    canvas = Image.new(
        "RGB",
        (cols * thumb_size[0], rows * thumb_size[1]),
        "white"
    )

    for i, path in enumerate(image_paths):
        img = safe_image_open(path)

        if img is None:
            continue

        img = ImageOps.contain(img, thumb_size)

        x = (i % cols) * thumb_size[0] + (thumb_size[0] - img.width) // 2
        y = (i // cols) * thumb_size[1] + (thumb_size[1] - img.height) // 2

        canvas.paste(img, (x, y))

    plt.figure(figsize=(8, 8))
    plt.imshow(canvas)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# ============================================================
# 1. LOAD METADATA
# ============================================================

metadata_frames = []

for class_name, info in CLASS_INFO.items():

    if not info["metadata_file"].exists():
        raise FileNotFoundError(f"Missing metadata file: {info['metadata_file']}")

    df = pd.read_csv(info["metadata_file"])

    if "participant_id" not in df.columns:
        raise ValueError(f"`participant_id` column missing in {info['metadata_file']}")

    df["participant_id"] = df["participant_id"].apply(normalize_pid)
    df["class_name"] = class_name
    df["label"] = info["label"]

    if "Sex" in df.columns:
        df["Sex_standardized"] = df["Sex"].apply(standardize_sex)

    if "Age" in df.columns:
        df["Age_numeric"] = safe_numeric(df["Age"])

    metadata_frames.append(df)

metadata = pd.concat(metadata_frames, ignore_index=True)
metadata.to_csv(TABLES_DIR / "combined_participant_metadata.csv", index=False)


# ============================================================
# 2. PARTICIPANT DEMOGRAPHICS TABLE
# ============================================================

demographic_rows = []

for group_name, group_df in [
    ("Anemia", metadata[metadata["label"] == 1]),
    ("Normal", metadata[metadata["label"] == 0]),
    ("Total", metadata),
]:

    row = {
        "Variable": group_name,
        "Participants": len(group_df),
    }

    if "Sex_standardized" in group_df.columns:
        row["Male"] = int((group_df["Sex_standardized"] == "Male").sum())
        row["Female"] = int((group_df["Sex_standardized"] == "Female").sum())

    if "Age_numeric" in group_df.columns:
        age = group_df["Age_numeric"].dropna()

        row["Age Mean"] = round(age.mean(), 2)
        row["Age SD"] = round(age.std(), 2)
        row["Age Mean ± SD"] = f"{age.mean():.2f} ± {age.std():.2f}"

        q1 = age.quantile(0.25)
        q3 = age.quantile(0.75)

        row["Age Median"] = round(age.median(), 2)
        row["Age Q1"] = round(q1, 2)
        row["Age Q3"] = round(q3, 2)
        row["Age Median (IQR)"] = f"{age.median():.2f} ({q1:.2f}–{q3:.2f})"
        row["Age Min"] = round(age.min(), 2)
        row["Age Max"] = round(age.max(), 2)

    demographic_rows.append(row)

demographics = pd.DataFrame(demographic_rows)
demographics.to_csv(TABLES_DIR / "table1_participant_demographics.csv", index=False)


# ============================================================
# 3. IMAGE INVENTORY
# ============================================================

image_records = []

for class_name, info in CLASS_INFO.items():

    for modality, folder, extension in [
        ("RGB", info["rgb_dir"], "*.png"),
        ("Thermal", info["thermal_dir"], "*.png"),
    ]:

        if not folder.exists():
            print(f"WARNING: Missing folder: {folder}")
            continue

        for img_path in folder.rglob(extension):

            pid = extract_pid_from_filename(img_path.name)
            view = extract_view_from_filename(img_path.name)

            image_records.append({
                "participant_id": pid,
                "class_name": class_name,
                "label": info["label"],
                "modality": modality,
                "view": view,
                "file_name": img_path.name,
                "path": str(img_path),
                "folder": str(img_path.parent),
            })

image_inventory = pd.DataFrame(image_records)
image_inventory.to_csv(TABLES_DIR / "image_inventory.csv", index=False)


# ============================================================
# 4. PARSER DIAGNOSTICS
# ============================================================

view_diagnostics = (
    image_inventory
    .groupby(["modality", "view"], dropna=False)
    .size()
    .reset_index(name="count")
)

view_diagnostics.to_csv(TABLES_DIR / "filename_view_parser_diagnostics.csv", index=False)

unparsed = image_inventory[
    image_inventory["participant_id"].isna() |
    image_inventory["view"].isna()
].copy()

unparsed.to_csv(TABLES_DIR / "unparsed_image_filenames.csv", index=False)


# ============================================================
# 5. IMAGE STATISTICS TABLE
# ============================================================

image_stats = (
    image_inventory
    .groupby("modality")
    .size()
    .reset_index(name="Count")
)

image_stats = pd.concat(
    [
        image_stats,
        pd.DataFrame([{
            "modality": "Total",
            "Count": int(image_stats["Count"].sum())
        }])
    ],
    ignore_index=True
)

image_stats.to_csv(TABLES_DIR / "table2_image_statistics.csv", index=False)

class_modality_stats = (
    image_inventory
    .groupby(["class_name", "modality"])
    .size()
    .reset_index(name="Count")
)

class_modality_stats.to_csv(TABLES_DIR / "image_statistics_by_class.csv", index=False)


# ============================================================
# 6. PARTICIPANT-LEVEL MASTER DATASET
# ============================================================

participant_rows = []

for _, meta_row in metadata.iterrows():

    pid = meta_row["participant_id"]

    row = {
        "participant_id": pid,
        "label": meta_row["label"],
        "class_name": meta_row["class_name"],
    }

    if "Sex_standardized" in metadata.columns:
        row["sex"] = meta_row.get("Sex_standardized", None)

    if "Age_numeric" in metadata.columns:
        row["age"] = meta_row.get("Age_numeric", None)

    participant_images = image_inventory[
        image_inventory["participant_id"] == pid
    ]

    for view in EXPECTED_VIEWS:

        rgb_match = participant_images[
            (participant_images["modality"] == "RGB") &
            (participant_images["view"] == view)
        ]

        thermal_match = participant_images[
            (participant_images["modality"] == "Thermal") &
            (participant_images["view"] == view)
        ]

        row[f"rgb_{view.lower()}"] = (
            rgb_match.iloc[0]["path"]
            if len(rgb_match) > 0 else None
        )

        row[f"thermal_{view.lower()}"] = (
            thermal_match.iloc[0]["path"]
            if len(thermal_match) > 0 else None
        )

    row["rgb_count"] = sum(
        pd.notna(row[f"rgb_{view.lower()}"])
        for view in EXPECTED_VIEWS
    )

    row["thermal_count"] = sum(
        pd.notna(row[f"thermal_{view.lower()}"])
        for view in EXPECTED_VIEWS
    )

    row["complete"] = (
        row["rgb_count"] == 4
        and row["thermal_count"] == 4
    )

    participant_rows.append(row)

participant_dataset = pd.DataFrame(participant_rows)
participant_dataset.to_csv(TABLES_DIR / "participant_dataset.csv", index=False)

participant_inventory = participant_dataset[
    [
        "participant_id",
        "class_name",
        "label",
        "rgb_count",
        "thermal_count",
        "complete"
    ]
]

participant_inventory.to_csv(TABLES_DIR / "participant_multimodal_inventory.csv", index=False)

missing_participants = participant_dataset[
    participant_dataset["complete"] == False
]

missing_participants.to_csv(TABLES_DIR / "missing_participants_or_views.csv", index=False)

participant_summary = pd.DataFrame([{
    "total_participants": len(participant_dataset),
    "complete_participants": int(participant_dataset["complete"].sum()),
    "participants_with_missing_views": int((~participant_dataset["complete"]).sum()),
    "total_rgb_images": int((image_inventory["modality"] == "RGB").sum()),
    "total_thermal_images": int((image_inventory["modality"] == "Thermal").sum()),
    "total_images": len(image_inventory),
    "unparsed_image_filenames": len(unparsed),
}])

participant_summary.to_csv(TABLES_DIR / "participant_inventory_summary.csv", index=False)


# ============================================================
# 7. FIGURES
# ============================================================

if "Age_numeric" in metadata.columns:
    plt.figure(figsize=(7, 5))

    for class_name in ["Anemia", "Normal"]:
        values = metadata.loc[
            metadata["class_name"] == class_name,
            "Age_numeric"
        ].dropna()

        plt.hist(values, bins=15, alpha=0.6, label=class_name)

    plt.xlabel("Age")
    plt.ylabel("Participants")
    plt.title("Age Distribution by Class")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure1_age_distribution.png", dpi=300)
    plt.close()

if "Sex_standardized" in metadata.columns:
    sex_counts = (
        metadata
        .groupby(["class_name", "Sex_standardized"])
        .size()
        .unstack(fill_value=0)
    )

    sex_counts.plot(kind="bar", figsize=(7, 5))
    plt.xlabel("Class")
    plt.ylabel("Participants")
    plt.title("Sex Distribution by Class")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure2_sex_distribution.png", dpi=300)
    plt.close()

class_counts = (
    metadata["class_name"]
    .value_counts()
    .reindex(["Anemia", "Normal"])
)

plt.figure(figsize=(6, 5))
plt.bar(class_counts.index, class_counts.values)
plt.xlabel("Class")
plt.ylabel("Participants")
plt.title("Class Distribution")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "figure3_class_distribution.png", dpi=300)
plt.close()

rgb_examples = (
    image_inventory[image_inventory["modality"] == "RGB"]["path"]
    .dropna()
    .head(16)
    .tolist()
)

make_contact_sheet(
    rgb_examples,
    FIGURES_DIR / "figure4_rgb_examples.png",
    "RGB Hand Image Examples"
)

thermal_examples = (
    image_inventory[image_inventory["modality"] == "Thermal"]["path"]
    .dropna()
    .head(16)
    .tolist()
)

make_contact_sheet(
    thermal_examples,
    FIGURES_DIR / "figure5_thermal_examples.png",
    "Thermal Hand Image Examples"
)


# ============================================================
# 8. REPORT
# ============================================================

complete_count = int(participant_dataset["complete"].sum())
missing_count = int((~participant_dataset["complete"]).sum())

report = f"""# Stage 4 Dataset Characterization and Participant Inventory Report

The dataset characterization and participant-level multimodal inventory stage was completed.

## Participant Metadata

Total participants: {len(metadata)}

Anemia participants: {int((metadata["label"] == 1).sum())}

Normal participants: {int((metadata["label"] == 0).sum())}

Duplicate participant IDs: {int(metadata["participant_id"].duplicated().sum())}

## Image Inventory

RGB images: {int((image_inventory["modality"] == "RGB").sum())}

Thermal images: {int((image_inventory["modality"] == "Thermal").sum())}

Total images: {len(image_inventory)}

## Filename Parsing Diagnostics

Unparsed image filenames: {len(unparsed)}

The parser diagnostics are saved in:

`filename_view_parser_diagnostics.csv`

`unparsed_image_filenames.csv`

## Participant-Level Completeness

Complete participants: {complete_count}

Participants with missing views: {missing_count}

Expected per participant:

- 4 RGB views
- 4 thermal views

## Generated Tables

- table1_participant_demographics.csv
- table2_image_statistics.csv
- image_statistics_by_class.csv
- image_inventory.csv
- filename_view_parser_diagnostics.csv
- unparsed_image_filenames.csv
- participant_dataset.csv
- participant_multimodal_inventory.csv
- participant_inventory_summary.csv
- missing_participants_or_views.csv

## Generated Figures

- figure1_age_distribution.png
- figure2_sex_distribution.png
- figure3_class_distribution.png
- figure4_rgb_examples.png
- figure5_thermal_examples.png

## Interpretation

This stage verifies dataset readiness for participant-level multimodal modeling. The master file `participant_dataset.csv` should be used as the reference file for all subsequent experiments. Model splitting must be performed at the participant level to avoid leakage across multiple images from the same participant.
"""

with open(
    REPORTS_DIR / "Stage4_Dataset_Characterization_and_Inventory_Report.md",
    "w",
    encoding="utf-8"
) as f:
    f.write(report)


# ============================================================
# 9. CONSOLE SUMMARY
# ============================================================

print("=" * 80)
print("STAGE 4 DATASET CHARACTERIZATION AND INVENTORY COMPLETED")
print("=" * 80)
print(f"Participants: {len(metadata)}")
print(f"Anemia participants: {int((metadata['label'] == 1).sum())}")
print(f"Normal participants: {int((metadata['label'] == 0).sum())}")
print(f"RGB images: {int((image_inventory['modality'] == 'RGB').sum())}")
print(f"Thermal images: {int((image_inventory['modality'] == 'Thermal').sum())}")
print(f"Total images: {len(image_inventory)}")
print(f"Unparsed image filenames: {len(unparsed)}")
print(f"Complete participants: {complete_count}")
print(f"Participants with missing views: {missing_count}")
print("=" * 80)
print("View extraction diagnostics:")
print(view_diagnostics.to_string(index=False))
print("=" * 80)
print(f"Results saved to: {OUTPUT_DIR}")
print("=" * 80)