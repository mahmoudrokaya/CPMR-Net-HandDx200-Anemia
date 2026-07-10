# -*- coding: utf-8 -*-
"""
Stage 10I2 - PyTorch Participant-Level Dataset and Dataloader Construction

Purpose:
Build and validate reusable PyTorch Dataset/DataLoader objects for CPMR-Net.

This stage:
- Uses Stage 10A valid participant-level manifest.
- Uses Stage 10I1 participant-level split files.
- Loads all 8 original participant images.
- Loads all 64 generated representations per participant from Stage 10B.
- Organizes each sample hierarchically:
    participant
      ├── rgb
      │     ├── l_dorsal
      │     ├── l_palmar
      │     ├── r_dorsal
      │     └── r_palmar
      └── thermal
            ├── l_dorsal
            ├── l_palmar
            ├── r_dorsal
            └── r_palmar
- Validates holdout train/val/test dataloaders.
- Does NOT train a model.
"""

from pathlib import Path
import json
from datetime import datetime

import cv2
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

BASE_DIR = Path(r"D:\47\472\New-Papers\Anemia_Paper")
OUTPUTS_DIR = BASE_DIR / "Outputs"

STAGE10A_DIR = OUTPUTS_DIR / "Stage10A_ParticipantLevel_Dataset_Loader"
STAGE10B_DIR = OUTPUTS_DIR / "Stage10B_MultiRepresentation_Generator"
STAGE10I1_DIR = OUTPUTS_DIR / "Stage10I1_Participant_Level_Split_Strategy"

PARTICIPANT_MANIFEST = STAGE10A_DIR / "tables" / "valid_participant_level_manifest.csv"
REPRESENTATION_MANIFEST = STAGE10B_DIR / "tables" / "multi_representation_manifest.csv"
HOLDOUT_SPLIT_FILE = STAGE10I1_DIR / "tables" / "holdout_train_val_test_split.csv"
REPEATED_SPLIT_FILE = STAGE10I1_DIR / "tables" / "repeated_stratified_5fold_train_val_test_splits.csv"

STAGE_OUT = OUTPUTS_DIR / "Stage10I2_PyTorch_Participant_Dataset_Dataloader"
TABLES_OUT = STAGE_OUT / "tables"
REPORTS_OUT = STAGE_OUT / "reports"

TABLES_OUT.mkdir(parents=True, exist_ok=True)
REPORTS_OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

BATCH_SIZE = 4
NUM_WORKERS = 0
IMAGE_SIZE = 224
PATCH_SIZE = 112

MODALITIES = ["rgb", "thermal"]
VIEWS = ["l_dorsal", "l_palmar", "r_dorsal", "r_palmar"]

RGB_REPRESENTATIONS = [
    "rgb_original",
    "rgb_hsv",
    "rgb_lab",
    "rgb_texture",
    "patch_center",
    "patch_upper",
    "patch_lower",
    "patch_left",
    "patch_right",
]

THERMAL_REPRESENTATIONS = [
    "thermal_normalized",
    "thermal_texture",
    "patch_center",
    "patch_upper",
    "patch_lower",
    "patch_left",
    "patch_right",
]

EXPECTED_REPRESENTATIONS_PER_PARTICIPANT = 64


# ---------------------------------------------------------------------
# Image loading helpers
# ---------------------------------------------------------------------

def read_image_as_tensor(path, modality, representation):
    """
    Load generated representation image as torch tensor.

    RGB full and RGB patches are 3-channel.
    RGB texture, thermal full, and thermal patches are 1-channel.
    """

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if img is None:
        raise RuntimeError(f"Could not read image: {path}")

    if modality == "rgb" and representation != "rgb_texture":
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    else:
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img[:, :, None]

    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))

    return torch.tensor(img, dtype=torch.float32)


def empty_tensor_for(modality, representation):
    if modality == "rgb" and representation != "rgb_texture":
        channels = 3
    else:
        channels = 1

    if representation.startswith("patch_"):
        return torch.zeros((channels, PATCH_SIZE, PATCH_SIZE), dtype=torch.float32)

    return torch.zeros((channels, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32)


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class CPMRParticipantDataset(Dataset):
    def __init__(
        self,
        participant_manifest,
        representation_manifest,
        split_df,
        split_name,
        training_mode=False
    ):
        self.participant_manifest = participant_manifest.copy()
        self.representation_manifest = representation_manifest.copy()
        self.split_df = split_df[split_df["split"] == split_name].copy()
        self.split_name = split_name
        self.training_mode = training_mode

        self.participant_ids = self.split_df["participant_id"].astype(str).tolist()

        self.participant_manifest["participant_id"] = self.participant_manifest["participant_id"].astype(str)
        self.representation_manifest["participant_id"] = self.representation_manifest["participant_id"].astype(str)

        self.participant_lookup = {
            str(row["participant_id"]): row
            for _, row in self.participant_manifest.iterrows()
        }

        self.rep_groups = {
            pid: g.copy()
            for pid, g in self.representation_manifest.groupby("participant_id")
        }

    def __len__(self):
        return len(self.participant_ids)

    def _load_representations_for_participant(self, participant_id):
        if participant_id not in self.rep_groups:
            raise RuntimeError(f"No representations found for participant: {participant_id}")

        rep_df = self.rep_groups[participant_id]

        data = {
            "rgb": {},
            "thermal": {},
        }

        for modality in MODALITIES:
            reps_expected = RGB_REPRESENTATIONS if modality == "rgb" else THERMAL_REPRESENTATIONS

            for view in VIEWS:
                data[modality][view] = {}

                view_df = rep_df[
                    (rep_df["modality"] == modality)
                    & (rep_df["view"] == view)
                    & (rep_df["status"] == "saved")
                ]

                for rep_name in reps_expected:
                    row_df = view_df[view_df["representation"] == rep_name]

                    if len(row_df) == 0:
                        tensor = empty_tensor_for(modality, rep_name)
                        exists = False
                        path = ""
                    else:
                        path = row_df.iloc[0]["path"]
                        tensor = read_image_as_tensor(path, modality, rep_name)
                        exists = True

                    data[modality][view][rep_name] = {
                        "tensor": tensor,
                        "exists": exists,
                        "path": path
                    }

        return data

    def __getitem__(self, idx):
        participant_id = self.participant_ids[idx]

        if participant_id not in self.participant_lookup:
            raise RuntimeError(f"Participant not found in manifest: {participant_id}")

        p_row = self.participant_lookup[participant_id]

        label = int(p_row["label"])
        class_name = str(p_row["class_name"])

        data = self._load_representations_for_participant(participant_id)

        sample = {
            "participant_id": participant_id,
            "label": torch.tensor(label, dtype=torch.long),
            "class_name": class_name,
            "sex": str(p_row["sex"]),
            "age": torch.tensor(float(p_row["age"]), dtype=torch.float32),
            "split": self.split_name,
            "representations": data,
        }

        return sample


# ---------------------------------------------------------------------
# Custom collate function
# ---------------------------------------------------------------------

def cpmr_collate_fn(batch):
    """
    Collate participant-level hierarchical samples.

    Because representations have different channel counts and spatial sizes,
    tensors are stacked only within the same modality/view/representation.
    """

    output = {
        "participant_id": [item["participant_id"] for item in batch],
        "label": torch.stack([item["label"] for item in batch]),
        "class_name": [item["class_name"] for item in batch],
        "sex": [item["sex"] for item in batch],
        "age": torch.stack([item["age"] for item in batch]),
        "split": [item["split"] for item in batch],
        "representations": {
            "rgb": {},
            "thermal": {},
        }
    }

    for modality in MODALITIES:
        reps_expected = RGB_REPRESENTATIONS if modality == "rgb" else THERMAL_REPRESENTATIONS

        for view in VIEWS:
            output["representations"][modality][view] = {}

            for rep_name in reps_expected:
                tensors = [
                    item["representations"][modality][view][rep_name]["tensor"]
                    for item in batch
                ]

                exists = [
                    item["representations"][modality][view][rep_name]["exists"]
                    for item in batch
                ]

                paths = [
                    item["representations"][modality][view][rep_name]["path"]
                    for item in batch
                ]

                output["representations"][modality][view][rep_name] = {
                    "tensor": torch.stack(tensors, dim=0),
                    "exists": exists,
                    "path": paths
                }

    return output


# ---------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------

def validate_loader(loader, split_name, max_batches=2):
    rows = []

    checked_batches = 0
    checked_participants = 0

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break

        batch_size = len(batch["participant_id"])
        checked_batches += 1
        checked_participants += batch_size

        for modality in MODALITIES:
            reps_expected = RGB_REPRESENTATIONS if modality == "rgb" else THERMAL_REPRESENTATIONS

            for view in VIEWS:
                for rep_name in reps_expected:
                    tensor = batch["representations"][modality][view][rep_name]["tensor"]
                    exists = batch["representations"][modality][view][rep_name]["exists"]

                    rows.append({
                        "split": split_name,
                        "batch_idx": batch_idx,
                        "modality": modality,
                        "view": view,
                        "representation": rep_name,
                        "tensor_shape": str(list(tensor.shape)),
                        "batch_size": batch_size,
                        "all_exist": bool(all(exists)),
                        "has_nan": bool(torch.isnan(tensor).any().item()),
                        "has_inf": bool(torch.isinf(tensor).any().item()),
                        "min_value": float(tensor.min().item()),
                        "max_value": float(tensor.max().item()),
                    })

    return rows, checked_batches, checked_participants


def count_expected_representations(dataset):
    rows = []

    for pid in dataset.participant_ids:
        rep_df = dataset.rep_groups.get(pid, pd.DataFrame())
        saved_count = int((rep_df["status"] == "saved").sum()) if len(rep_df) else 0

        rows.append({
            "participant_id": pid,
            "split": dataset.split_name,
            "saved_representations": saved_count,
            "expected_representations": EXPECTED_REPRESENTATIONS_PER_PARTICIPANT,
            "complete": saved_count == EXPECTED_REPRESENTATIONS_PER_PARTICIPANT
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    for path in [PARTICIPANT_MANIFEST, REPRESENTATION_MANIFEST, HOLDOUT_SPLIT_FILE, REPEATED_SPLIT_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")

    participant_manifest = pd.read_csv(PARTICIPANT_MANIFEST)
    representation_manifest = pd.read_csv(REPRESENTATION_MANIFEST)
    holdout_split = pd.read_csv(HOLDOUT_SPLIT_FILE)
    repeated_split = pd.read_csv(REPEATED_SPLIT_FILE)

    # Use holdout split for dataloader construction validation
    train_dataset = CPMRParticipantDataset(
        participant_manifest,
        representation_manifest,
        holdout_split,
        split_name="train",
        training_mode=True
    )

    val_dataset = CPMRParticipantDataset(
        participant_manifest,
        representation_manifest,
        holdout_split,
        split_name="val",
        training_mode=False
    )

    test_dataset = CPMRParticipantDataset(
        participant_manifest,
        representation_manifest,
        holdout_split,
        split_name="test",
        training_mode=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=cpmr_collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=cpmr_collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=cpmr_collate_fn
    )

    all_validation_rows = []
    checked_info = []

    for split_name, loader in [
        ("train", train_loader),
        ("val", val_loader),
        ("test", test_loader)
    ]:
        rows, checked_batches, checked_participants = validate_loader(
            loader,
            split_name=split_name,
            max_batches=2
        )
        all_validation_rows.extend(rows)
        checked_info.append({
            "split": split_name,
            "checked_batches": checked_batches,
            "checked_participants": checked_participants
        })

    validation_df = pd.DataFrame(all_validation_rows)
    validation_df.to_csv(TABLES_OUT / "dataloader_tensor_validation_summary.csv", index=False)

    checked_info_df = pd.DataFrame(checked_info)
    checked_info_df.to_csv(TABLES_OUT / "dataloader_checked_batches_summary.csv", index=False)

    completeness_df = pd.concat([
        count_expected_representations(train_dataset),
        count_expected_representations(val_dataset),
        count_expected_representations(test_dataset)
    ], ignore_index=True)

    completeness_df.to_csv(TABLES_OUT / "dataloader_participant_representation_completeness.csv", index=False)

    split_dataset_summary = pd.DataFrame([
        {
            "split": "train",
            "participants": len(train_dataset),
            "batches": len(train_loader),
            "training_mode": True
        },
        {
            "split": "val",
            "participants": len(val_dataset),
            "batches": len(val_loader),
            "training_mode": False
        },
        {
            "split": "test",
            "participants": len(test_dataset),
            "batches": len(test_loader),
            "training_mode": False
        }
    ])

    split_dataset_summary.to_csv(TABLES_OUT / "dataloader_split_summary.csv", index=False)

    total_checked_tensors = len(validation_df)
    missing_tensor_groups = int((~validation_df["all_exist"]).sum()) if len(validation_df) else 0
    nan_groups = int(validation_df["has_nan"].sum()) if len(validation_df) else 0
    inf_groups = int(validation_df["has_inf"].sum()) if len(validation_df) else 0
    incomplete_participants = int((~completeness_df["complete"]).sum())

    summary = {
        "stage": "Stage10I2",
        "title": "PyTorch Participant-Level Dataset and Dataloader Construction",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "participant_manifest": str(PARTICIPANT_MANIFEST),
        "representation_manifest": str(REPRESENTATION_MANIFEST),
        "holdout_split_file": str(HOLDOUT_SPLIT_FILE),
        "repeated_split_file": str(REPEATED_SPLIT_FILE),
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "train_participants": int(len(train_dataset)),
        "val_participants": int(len(val_dataset)),
        "test_participants": int(len(test_dataset)),
        "train_batches": int(len(train_loader)),
        "val_batches": int(len(val_loader)),
        "test_batches": int(len(test_loader)),
        "expected_representations_per_participant": EXPECTED_REPRESENTATIONS_PER_PARTICIPANT,
        "participants_checked_for_completeness": int(len(completeness_df)),
        "incomplete_participants": incomplete_participants,
        "validated_tensor_groups": total_checked_tensors,
        "missing_tensor_groups_in_checked_batches": missing_tensor_groups,
        "nan_tensor_groups_in_checked_batches": nan_groups,
        "inf_tensor_groups_in_checked_batches": inf_groups,
        "hierarchical_structure": "participant -> modality -> view -> representation",
        "outputs_saved_to": str(STAGE_OUT)
    }

    with open(STAGE_OUT / "Stage10I2_PyTorch_Participant_Dataset_Dataloader_Summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    report = []
    report.append("# Stage 10I2 PyTorch Participant-Level Dataset and Dataloader Construction\n")
    report.append(f"Generated at: {summary['created_at']}\n")

    report.append("## Purpose\n")
    report.append(
        "This stage constructs and validates the participant-level PyTorch Dataset and DataLoader objects for CPMR-Net. "
        "Each sample corresponds to one participant and contains all generated multimodal representations organized as "
        "participant → modality → anatomical view → representation.\n"
    )

    report.append("## Dataset Split Summary\n")
    report.append(f"- Train participants: {summary['train_participants']}")
    report.append(f"- Validation participants: {summary['val_participants']}")
    report.append(f"- Test participants: {summary['test_participants']}")
    report.append(f"- Batch size: {summary['batch_size']}")
    report.append(f"- Train batches: {summary['train_batches']}")
    report.append(f"- Validation batches: {summary['val_batches']}")
    report.append(f"- Test batches: {summary['test_batches']}\n")

    report.append("## Validation Summary\n")
    report.append(f"- Expected representations per participant: {summary['expected_representations_per_participant']}")
    report.append(f"- Participants checked for completeness: {summary['participants_checked_for_completeness']}")
    report.append(f"- Incomplete participants: {summary['incomplete_participants']}")
    report.append(f"- Validated tensor groups in checked batches: {summary['validated_tensor_groups']}")
    report.append(f"- Missing tensor groups in checked batches: {summary['missing_tensor_groups_in_checked_batches']}")
    report.append(f"- NaN tensor groups in checked batches: {summary['nan_tensor_groups_in_checked_batches']}")
    report.append(f"- Inf tensor groups in checked batches: {summary['inf_tensor_groups_in_checked_batches']}\n")

    report.append("## Output Files\n")
    report.append("- `dataloader_tensor_validation_summary.csv`")
    report.append("- `dataloader_checked_batches_summary.csv`")
    report.append("- `dataloader_participant_representation_completeness.csv`")
    report.append("- `dataloader_split_summary.csv`")
    report.append("- `Stage10I2_PyTorch_Participant_Dataset_Dataloader_Summary.json`\n")

    report.append("## Implementation Role\n")
    report.append(
        "This stage provides the reusable participant-level data interface for CPMR-Net training. "
        "All subsequent training, validation, testing, and ablation scripts should use this hierarchical data organization."
    )

    with open(REPORTS_OUT / "Stage10I2_PyTorch_Participant_Dataset_Dataloader_Report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("=" * 80)
    print("STAGE 10I2 PYTORCH PARTICIPANT DATASET AND DATALOADER COMPLETED")
    print("=" * 80)
    print(f"Train participants: {summary['train_participants']}")
    print(f"Validation participants: {summary['val_participants']}")
    print(f"Test participants: {summary['test_participants']}")
    print(f"Train/Val/Test batches: {summary['train_batches']}/{summary['val_batches']}/{summary['test_batches']}")
    print(f"Expected representations per participant: {summary['expected_representations_per_participant']}")
    print(f"Incomplete participants: {summary['incomplete_participants']}")
    print(f"Missing tensor groups in checked batches: {summary['missing_tensor_groups_in_checked_batches']}")
    print(f"NaN tensor groups in checked batches: {summary['nan_tensor_groups_in_checked_batches']}")
    print(f"Inf tensor groups in checked batches: {summary['inf_tensor_groups_in_checked_batches']}")
    print(f"Outputs saved to: {STAGE_OUT}")
    print("=" * 80)


if __name__ == "__main__":
    main()