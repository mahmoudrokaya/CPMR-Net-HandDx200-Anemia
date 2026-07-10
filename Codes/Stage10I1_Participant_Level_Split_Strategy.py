# -*- coding: utf-8 -*-
"""
Stage 10I1 - Participant-Level Train/Validation/Test Split Strategy

Purpose:
Create reproducible participant-level split files for CPMR-Net supervised training.

This stage:
- Uses the Stage 10A valid participant-level manifest
- Prevents image-level leakage by splitting only by participant_id
- Creates:
    1) Holdout train/validation/test split
    2) Repeated stratified 5-fold outer evaluation splits
    3) Fold-level train/validation split inside each training fold
- Summarizes class balance for all splits

No model training is performed.
"""

from pathlib import Path
import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

BASE_DIR = Path(r"D:\47\472\New-Papers\Anemia_Paper")
OUTPUTS_DIR = BASE_DIR / "Outputs"

STAGE10A_DIR = OUTPUTS_DIR / "Stage10A_ParticipantLevel_Dataset_Loader"
INPUT_MANIFEST = STAGE10A_DIR / "tables" / "valid_participant_level_manifest.csv"

STAGE_OUT = OUTPUTS_DIR / "Stage10I1_Participant_Level_Split_Strategy"
TABLES_OUT = STAGE_OUT / "tables"
REPORTS_OUT = STAGE_OUT / "reports"

TABLES_OUT.mkdir(parents=True, exist_ok=True)
REPORTS_OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

RANDOM_SEED = 42

HOLDOUT_TEST_SIZE = 0.20
HOLDOUT_VAL_SIZE_FROM_REMAINING = 0.20

N_SPLITS = 5
N_REPEATS = 5

REQUIRED_COLUMNS = [
    "participant_id",
    "label",
    "class_name",
    "sex",
    "age",
]


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def class_summary(df, split_name, scheme_name, fold=None, repeat=None):
    rows = []

    total = len(df)

    for class_name, g in df.groupby("class_name"):
        rows.append({
            "scheme": scheme_name,
            "repeat": repeat,
            "fold": fold,
            "split": split_name,
            "class_name": class_name,
            "count": int(len(g)),
            "percentage": round(100 * len(g) / total, 2) if total > 0 else 0.0,
            "total_split_size": int(total)
        })

    return rows


def leakage_check(split_df, scheme_cols):
    """
    Checks whether the same participant appears in more than one split
    within the same split scheme/fold/repeat.
    """

    leakage_records = []

    group_cols = scheme_cols

    for keys, g in split_df.groupby(group_cols):
        participant_split_counts = (
            g.groupby("participant_id")["split"]
            .nunique()
            .reset_index(name="num_splits")
        )

        leaked = participant_split_counts[participant_split_counts["num_splits"] > 1]

        if len(leaked) > 0:
            if not isinstance(keys, tuple):
                keys = (keys,)

            key_dict = dict(zip(group_cols, keys))

            for _, row in leaked.iterrows():
                rec = key_dict.copy()
                rec["participant_id"] = row["participant_id"]
                rec["num_splits"] = int(row["num_splits"])
                leakage_records.append(rec)

    return pd.DataFrame(leakage_records)


def add_split_rows(df, split_name, scheme_name, fold=None, repeat=None):
    out = df.copy()
    out["scheme"] = scheme_name
    out["repeat"] = repeat
    out["fold"] = fold
    out["split"] = split_name
    return out


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    if not INPUT_MANIFEST.exists():
        raise FileNotFoundError(f"Missing valid participant manifest: {INPUT_MANIFEST}")

    manifest = pd.read_csv(INPUT_MANIFEST)

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in manifest.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns in participant manifest: {missing_cols}")

    participants = manifest[REQUIRED_COLUMNS].drop_duplicates("participant_id").copy()
    participants = participants.sort_values("participant_id").reset_index(drop=True)

    X = participants["participant_id"].values
    y = participants["label"].values

    # -----------------------------------------------------------------
    # Holdout train/validation/test split
    # -----------------------------------------------------------------

    train_val_df, test_df = train_test_split(
        participants,
        test_size=HOLDOUT_TEST_SIZE,
        stratify=participants["label"],
        random_state=RANDOM_SEED
    )

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=HOLDOUT_VAL_SIZE_FROM_REMAINING,
        stratify=train_val_df["label"],
        random_state=RANDOM_SEED
    )

    holdout_rows = pd.concat([
        add_split_rows(train_df, "train", "holdout", fold=None, repeat=None),
        add_split_rows(val_df, "val", "holdout", fold=None, repeat=None),
        add_split_rows(test_df, "test", "holdout", fold=None, repeat=None),
    ], ignore_index=True)

    holdout_rows.to_csv(TABLES_OUT / "holdout_train_val_test_split.csv", index=False)

    # -----------------------------------------------------------------
    # Repeated stratified folds
    # -----------------------------------------------------------------

    repeated_rows = []

    for repeat in range(1, N_REPEATS + 1):
        seed = RANDOM_SEED + repeat - 1

        skf = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=seed
        )

        for fold, (train_val_idx, test_idx) in enumerate(skf.split(X, y), start=1):
            fold_train_val = participants.iloc[train_val_idx].copy()
            fold_test = participants.iloc[test_idx].copy()

            fold_train, fold_val = train_test_split(
                fold_train_val,
                test_size=HOLDOUT_VAL_SIZE_FROM_REMAINING,
                stratify=fold_train_val["label"],
                random_state=seed + fold
            )

            repeated_rows.append(
                add_split_rows(
                    fold_train,
                    "train",
                    "repeated_stratified_5fold",
                    fold=fold,
                    repeat=repeat
                )
            )

            repeated_rows.append(
                add_split_rows(
                    fold_val,
                    "val",
                    "repeated_stratified_5fold",
                    fold=fold,
                    repeat=repeat
                )
            )

            repeated_rows.append(
                add_split_rows(
                    fold_test,
                    "test",
                    "repeated_stratified_5fold",
                    fold=fold,
                    repeat=repeat
                )
            )

    repeated_split_df = pd.concat(repeated_rows, ignore_index=True)
    repeated_split_df.to_csv(TABLES_OUT / "repeated_stratified_5fold_train_val_test_splits.csv", index=False)

    # -----------------------------------------------------------------
    # Combined split file
    # -----------------------------------------------------------------

    combined_splits = pd.concat([holdout_rows, repeated_split_df], ignore_index=True)
    combined_splits.to_csv(TABLES_OUT / "all_participant_level_training_splits.csv", index=False)

    # -----------------------------------------------------------------
    # Class-balance summaries
    # -----------------------------------------------------------------

    summary_rows = []

    for split_name, g in holdout_rows.groupby("split"):
        summary_rows.extend(
            class_summary(
                g,
                split_name=split_name,
                scheme_name="holdout",
                fold=None,
                repeat=None
            )
        )

    for (repeat, fold, split_name), g in repeated_split_df.groupby(["repeat", "fold", "split"]):
        summary_rows.extend(
            class_summary(
                g,
                split_name=split_name,
                scheme_name="repeated_stratified_5fold",
                fold=int(fold),
                repeat=int(repeat)
            )
        )

    class_balance_df = pd.DataFrame(summary_rows)
    class_balance_df.to_csv(TABLES_OUT / "split_class_balance_summary.csv", index=False)

    # -----------------------------------------------------------------
    # Leakage checks
    # -----------------------------------------------------------------

    holdout_leakage = leakage_check(
        holdout_rows,
        scheme_cols=["scheme"]
    )

    repeated_leakage = leakage_check(
        repeated_split_df,
        scheme_cols=["scheme", "repeat", "fold"]
    )

    holdout_leakage.to_csv(TABLES_OUT / "holdout_split_leakage_check.csv", index=False)
    repeated_leakage.to_csv(TABLES_OUT / "repeated_fold_leakage_check.csv", index=False)

    leakage_detected = len(holdout_leakage) > 0 or len(repeated_leakage) > 0

    # -----------------------------------------------------------------
    # Split size summary
    # -----------------------------------------------------------------

    split_size_summary = (
        combined_splits.groupby(["scheme", "repeat", "fold", "split"], dropna=False)
        .size()
        .reset_index(name="count")
    )

    split_size_summary.to_csv(TABLES_OUT / "split_size_summary.csv", index=False)

    holdout_size_summary = (
        holdout_rows.groupby("split")
        .size()
        .reset_index(name="count")
    )

    repeated_size_summary = (
        repeated_split_df.groupby(["repeat", "fold", "split"])
        .size()
        .reset_index(name="count")
    )

    holdout_size_summary.to_csv(TABLES_OUT / "holdout_split_size_summary.csv", index=False)
    repeated_size_summary.to_csv(TABLES_OUT / "repeated_fold_split_size_summary.csv", index=False)

    # -----------------------------------------------------------------
    # Summary JSON
    # -----------------------------------------------------------------

    class_counts = (
        participants.groupby("class_name")
        .size()
        .reset_index(name="count")
        .to_dict(orient="records")
    )

    summary = {
        "stage": "Stage10I1",
        "title": "Participant-Level Train/Validation/Test Split Strategy",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_manifest": str(INPUT_MANIFEST),
        "total_participants": int(len(participants)),
        "class_counts": class_counts,
        "random_seed": RANDOM_SEED,
        "holdout_test_size": HOLDOUT_TEST_SIZE,
        "holdout_val_size_from_remaining": HOLDOUT_VAL_SIZE_FROM_REMAINING,
        "holdout_train_size": int(len(train_df)),
        "holdout_val_size": int(len(val_df)),
        "holdout_test_size_count": int(len(test_df)),
        "n_splits": N_SPLITS,
        "n_repeats": N_REPEATS,
        "total_repeated_fold_configurations": int(N_SPLITS * N_REPEATS),
        "leakage_detected": bool(leakage_detected),
        "holdout_leakage_records": int(len(holdout_leakage)),
        "repeated_leakage_records": int(len(repeated_leakage)),
        "outputs_saved_to": str(STAGE_OUT),
    }

    with open(STAGE_OUT / "Stage10I1_Participant_Level_Split_Strategy_Summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    # -----------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------

    report = []
    report.append("# Stage 10I1 Participant-Level Train/Validation/Test Split Strategy\n")
    report.append(f"Generated at: {summary['created_at']}\n")

    report.append("## Purpose\n")
    report.append(
        "This stage defines the supervised-learning split strategy for CPMR-Net. "
        "All splits are performed at the participant level to prevent image-level leakage. "
        "This ensures that all RGB, thermal, representation, specialist, and fused evidence derived from the same participant "
        "remain within the same split.\n"
    )

    report.append("## Dataset Summary\n")
    report.append(f"- Total participants: {summary['total_participants']}")
    for item in class_counts:
        report.append(f"- {item['class_name']}: {item['count']} participants")

    report.append("\n## Holdout Split\n")
    report.append(f"- Train participants: {summary['holdout_train_size']}")
    report.append(f"- Validation participants: {summary['holdout_val_size']}")
    report.append(f"- Test participants: {summary['holdout_test_size_count']}")

    report.append("\n## Repeated Cross-Validation Strategy\n")
    report.append(f"- Stratified folds: {summary['n_splits']}")
    report.append(f"- Repeats: {summary['n_repeats']}")
    report.append(f"- Total fold configurations: {summary['total_repeated_fold_configurations']}")

    report.append("\n## Leakage Check\n")
    report.append(f"- Leakage detected: {summary['leakage_detected']}")
    report.append(f"- Holdout leakage records: {summary['holdout_leakage_records']}")
    report.append(f"- Repeated-fold leakage records: {summary['repeated_leakage_records']}")

    report.append("\n## Output Files\n")
    report.append("- `holdout_train_val_test_split.csv`")
    report.append("- `repeated_stratified_5fold_train_val_test_splits.csv`")
    report.append("- `all_participant_level_training_splits.csv`")
    report.append("- `split_class_balance_summary.csv`")
    report.append("- `split_size_summary.csv`")
    report.append("- `holdout_split_size_summary.csv`")
    report.append("- `repeated_fold_split_size_summary.csv`")
    report.append("- `holdout_split_leakage_check.csv`")
    report.append("- `repeated_fold_leakage_check.csv`")
    report.append("- `Stage10I1_Participant_Level_Split_Strategy_Summary.json`\n")

    report.append("## Implementation Role\n")
    report.append(
        "These split files define the official supervised-learning protocol for CPMR-Net. "
        "All subsequent model training, validation, testing, ablation studies, and baseline comparisons should use these participant-level splits."
    )

    with open(REPORTS_OUT / "Stage10I1_Participant_Level_Split_Strategy_Report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("=" * 80)
    print("STAGE 10I1 PARTICIPANT-LEVEL SPLIT STRATEGY COMPLETED")
    print("=" * 80)
    print(f"Total participants: {summary['total_participants']}")
    print("Class counts:")
    for item in class_counts:
        print(f"  {item['class_name']}: {item['count']}")
    print(f"Holdout train/val/test: {summary['holdout_train_size']}/{summary['holdout_val_size']}/{summary['holdout_test_size_count']}")
    print(f"Repeated CV: {summary['n_repeats']} repeats × {summary['n_splits']} folds")
    print(f"Leakage detected: {summary['leakage_detected']}")
    print(f"Outputs saved to: {STAGE_OUT}")
    print("=" * 80)


if __name__ == "__main__":
    main()