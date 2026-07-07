# -*- coding: utf-8 -*-
"""
Stage 10A - Participant-Level Dataset Loader

First implementation step for CPMR-Net.

Purpose:
- Load participant_dataset.csv from Stage 4
- Validate eight image paths per participant
- Create participant-level deep-learning manifest
- Create participant-level stratified split plan
- Prevent image-level leakage
"""

from pathlib import Path
import json
import pandas as pd
from datetime import datetime
from sklearn.model_selection import StratifiedKFold

BASE_DIR = Path(r"D:\47\472\New-Papers\Anemia_Paper")
OUTPUTS_DIR = BASE_DIR / "Outputs"

STAGE4_DIR = OUTPUTS_DIR / "Stage4_Dataset_Characterization_and_Inventory"
INPUT_FILE = STAGE4_DIR / "tables" / "participant_dataset.csv"

STAGE_OUT = OUTPUTS_DIR / "Stage10A_ParticipantLevel_Dataset_Loader"
TABLES_OUT = STAGE_OUT / "tables"
REPORTS_OUT = STAGE_OUT / "reports"

TABLES_OUT.mkdir(parents=True, exist_ok=True)
REPORTS_OUT.mkdir(parents=True, exist_ok=True)

REQUIRED_VIEW_COLUMNS = [
    "rgb_l_dorsal",
    "rgb_l_palmar",
    "rgb_r_dorsal",
    "rgb_r_palmar",
    "thermal_l_dorsal",
    "thermal_l_palmar",
    "thermal_r_dorsal",
    "thermal_r_palmar",
]

REQUIRED_METADATA_COLUMNS = [
    "participant_id",
    "label",
    "class_name",
    "sex",
    "age",
]


def normalize_path(p):
    if pd.isna(p):
        return ""
    p = str(p).strip()
    if not p:
        return ""
    path = Path(p)
    if path.is_absolute():
        return str(path)
    return str(BASE_DIR / p)


def path_exists(p):
    if not p:
        return False
    return Path(p).exists()


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing Stage 4 participant dataset: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    missing_cols = [
        c for c in REQUIRED_METADATA_COLUMNS + REQUIRED_VIEW_COLUMNS
        if c not in df.columns
    ]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    manifest = df[REQUIRED_METADATA_COLUMNS + REQUIRED_VIEW_COLUMNS].copy()

    for col in REQUIRED_VIEW_COLUMNS:
        manifest[col] = manifest[col].apply(normalize_path)
        manifest[col + "_exists"] = manifest[col].apply(path_exists)

    existence_cols = [c + "_exists" for c in REQUIRED_VIEW_COLUMNS]

    manifest["all_images_exist"] = manifest[existence_cols].all(axis=1)
    manifest["num_existing_images"] = manifest[existence_cols].sum(axis=1)
    manifest["expected_images"] = len(REQUIRED_VIEW_COLUMNS)

    manifest["is_complete_for_deep_learning"] = (
        manifest["all_images_exist"] &
        manifest["participant_id"].notna() &
        manifest["label"].notna()
    )

    manifest.to_csv(TABLES_OUT / "participant_level_deep_learning_manifest.csv", index=False)

    missing_image_rows = manifest[~manifest["all_images_exist"]].copy()
    missing_image_rows.to_csv(TABLES_OUT / "participants_with_missing_deep_learning_images.csv", index=False)

    valid_manifest = manifest[manifest["is_complete_for_deep_learning"]].copy()
    valid_manifest.to_csv(TABLES_OUT / "valid_participant_level_manifest.csv", index=False)

    # Stratified participant-level folds
    split_rows = []

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    X = valid_manifest["participant_id"].values
    y = valid_manifest["label"].values

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        for idx in train_idx:
            split_rows.append({
                "participant_id": valid_manifest.iloc[idx]["participant_id"],
                "label": valid_manifest.iloc[idx]["label"],
                "class_name": valid_manifest.iloc[idx]["class_name"],
                "fold": fold_idx,
                "split": "train"
            })

        for idx in test_idx:
            split_rows.append({
                "participant_id": valid_manifest.iloc[idx]["participant_id"],
                "label": valid_manifest.iloc[idx]["label"],
                "class_name": valid_manifest.iloc[idx]["class_name"],
                "fold": fold_idx,
                "split": "test"
            })

    splits_df = pd.DataFrame(split_rows)
    splits_df.to_csv(TABLES_OUT / "participant_level_stratified_5fold_splits.csv", index=False)

    split_summary = (
        splits_df.groupby(["fold", "split", "class_name"])
        .size()
        .reset_index(name="count")
    )
    split_summary.to_csv(TABLES_OUT / "participant_level_split_summary.csv", index=False)

    summary = {
        "stage": "Stage10A",
        "title": "Participant-Level Dataset Loader",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_file": str(INPUT_FILE),
        "total_participants": int(len(manifest)),
        "valid_participants": int(len(valid_manifest)),
        "participants_with_missing_images": int(len(missing_image_rows)),
        "expected_images_per_participant": len(REQUIRED_VIEW_COLUMNS),
        "total_expected_images_valid_manifest": int(len(valid_manifest) * len(REQUIRED_VIEW_COLUMNS)),
        "n_splits": n_splits,
        "outputs_saved_to": str(STAGE_OUT)
    }

    with open(STAGE_OUT / "Stage10A_ParticipantLevel_Dataset_Loader_Summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    report = []
    report.append("# Stage 10A Participant-Level Dataset Loader\n")
    report.append(f"Generated at: {summary['created_at']}\n")
    report.append("## Purpose\n")
    report.append(
        "This stage creates the participant-level manifest required by CPMR-Net. "
        "It validates that each participant has all eight expected RGB and thermal views "
        "and creates participant-level stratified folds to avoid image-level leakage.\n"
    )

    report.append("## Summary\n")
    report.append(f"- Total participants: {summary['total_participants']}")
    report.append(f"- Valid participants: {summary['valid_participants']}")
    report.append(f"- Participants with missing images: {summary['participants_with_missing_images']}")
    report.append(f"- Expected images per participant: {summary['expected_images_per_participant']}")
    report.append(f"- Total expected images in valid manifest: {summary['total_expected_images_valid_manifest']}")
    report.append(f"- Stratified folds: {summary['n_splits']}\n")

    report.append("## Output Files\n")
    report.append("- `participant_level_deep_learning_manifest.csv`")
    report.append("- `valid_participant_level_manifest.csv`")
    report.append("- `participants_with_missing_deep_learning_images.csv`")
    report.append("- `participant_level_stratified_5fold_splits.csv`")
    report.append("- `participant_level_split_summary.csv`")
    report.append("- `Stage10A_ParticipantLevel_Dataset_Loader_Summary.json`\n")

    report.append("## Implementation Role\n")
    report.append(
        "This file is the first implementation artifact for CPMR-Net. "
        "All later modules should use the valid participant-level manifest and split file, "
        "not independent image-level splitting."
    )

    with open(REPORTS_OUT / "Stage10A_ParticipantLevel_Dataset_Loader_Report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("=" * 80)
    print("STAGE 10A PARTICIPANT-LEVEL DATASET LOADER COMPLETED")
    print("=" * 80)
    print(f"Total participants: {summary['total_participants']}")
    print(f"Valid participants: {summary['valid_participants']}")
    print(f"Participants with missing images: {summary['participants_with_missing_images']}")
    print(f"Expected images per participant: {summary['expected_images_per_participant']}")
    print(f"Total expected images in valid manifest: {summary['total_expected_images_valid_manifest']}")
    print(f"Outputs saved to: {STAGE_OUT}")
    print("=" * 80)


if __name__ == "__main__":
    main()