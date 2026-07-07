# -*- coding: utf-8 -*-
"""
Stage 10J - Complete CPMR-Net Model Implementation

Purpose:
Implement the full trainable CPMR-Net architecture and validate forward passes.

This stage:
- Loads Stage 10I4 config
- Reuses the participant-level Dataset/DataLoader logic
- Implements:
    1) Lightweight representation encoders
    2) Representation-level attention
    3) Anatomical specialist aggregation
    4) RGB-centered branch
    5) Thermal auxiliary branch
    6) Adaptive cooperative fusion
    7) Binary classification head
- Runs forward-pass validation on train/val/test batches
- Saves model architecture summary

No supervised training is performed here.
"""

from pathlib import Path
import json
from datetime import datetime

import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

BASE_DIR = Path(r"D:\47\472\New-Papers\Anemia_Paper")
OUTPUTS_DIR = BASE_DIR / "Outputs"

STAGE10I4_DIR = OUTPUTS_DIR / "Stage10I4_Training_Configuration_Experiment_Control"
CONFIG_FILE = STAGE10I4_DIR / "configs" / "CPMRNet_training_config_v1.json"

STAGE_OUT = OUTPUTS_DIR / "Stage10J_CPMRNet_Model_Implementation"
TABLES_OUT = STAGE_OUT / "tables"
REPORTS_OUT = STAGE_OUT / "reports"
MODELS_OUT = STAGE_OUT / "models"

for p in [TABLES_OUT, REPORTS_OUT, MODELS_OUT]:
    p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def read_image_as_tensor(path, modality, representation, image_size=224, patch_size=112):
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


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class CPMRParticipantDataset(Dataset):
    def __init__(self, participant_manifest, representation_manifest, split_df, split_name, config):
        self.participant_manifest = participant_manifest.copy()
        self.representation_manifest = representation_manifest.copy()
        self.split_df = split_df[split_df["split"] == split_name].copy()
        self.split_name = split_name
        self.config = config

        self.modalities = config["data"]["modalities"]
        self.views = config["data"]["views"]
        self.rgb_reps = config["data"]["rgb_representations"]
        self.thermal_reps = config["data"]["thermal_representations"]

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

    def __getitem__(self, idx):
        pid = self.participant_ids[idx]
        p_row = self.participant_lookup[pid]
        rep_df = self.rep_groups[pid]

        sample = {
            "participant_id": pid,
            "label": torch.tensor(int(p_row["label"]), dtype=torch.float32),
            "class_name": str(p_row["class_name"]),
            "representations": {"rgb": {}, "thermal": {}}
        }

        for modality in self.modalities:
            reps = self.rgb_reps if modality == "rgb" else self.thermal_reps
            for view in self.views:
                sample["representations"][modality][view] = {}
                view_df = rep_df[
                    (rep_df["modality"] == modality) &
                    (rep_df["view"] == view) &
                    (rep_df["status"] == "saved")
                ]

                for rep in reps:
                    row = view_df[view_df["representation"] == rep]
                    if len(row) == 0:
                        raise RuntimeError(f"Missing representation: {pid} {modality} {view} {rep}")
                    path = row.iloc[0]["path"]
                    sample["representations"][modality][view][rep] = read_image_as_tensor(
                        path,
                        modality,
                        rep,
                        image_size=self.config["data"]["image_size"],
                        patch_size=self.config["data"]["patch_size"]
                    )

        return sample


def cpmr_collate_fn(batch):
    first = batch[0]
    output = {
        "participant_id": [b["participant_id"] for b in batch],
        "label": torch.stack([b["label"] for b in batch]),
        "class_name": [b["class_name"] for b in batch],
        "representations": {"rgb": {}, "thermal": {}}
    }

    for modality in ["rgb", "thermal"]:
        for view in first["representations"][modality].keys():
            output["representations"][modality][view] = {}
            for rep in first["representations"][modality][view].keys():
                output["representations"][modality][view][rep] = torch.stack(
                    [b["representations"][modality][view][rep] for b in batch],
                    dim=0
                )

    return output


# ---------------------------------------------------------------------
# Model modules
# ---------------------------------------------------------------------

class LightweightEncoder(nn.Module):
    def __init__(self, in_channels, embedding_dim=128, dropout=0.25):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 96, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),

            nn.Conv2d(96, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, x):
        return self.projection(self.features(x))


class AttentionAggregator(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.Tanh(),
            nn.Linear(embedding_dim // 2, 1)
        )

    def forward(self, x):
        """
        x: [B, N, D]
        returns:
            aggregated: [B, D]
            weights: [B, N]
        """
        logits = self.score(x).squeeze(-1)
        weights = torch.softmax(logits, dim=1)
        aggregated = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return aggregated, weights


class AdaptiveRGBThermalFusion(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, 2)
        )

    def forward(self, rgb_vec, thermal_vec):
        x = torch.cat([rgb_vec, thermal_vec], dim=-1)
        weights = torch.softmax(self.gate(x), dim=-1)
        fused = weights[:, 0:1] * rgb_vec + weights[:, 1:2] * thermal_vec
        return fused, weights


class CPMRNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.embedding_dim = config["model"]["embedding_dim"]
        dropout = config["model"]["dropout"]

        self.views = config["data"]["views"]
        self.rgb_reps = config["data"]["rgb_representations"]
        self.thermal_reps = config["data"]["thermal_representations"]

        self.rgb_3ch_encoder = LightweightEncoder(3, self.embedding_dim, dropout)
        self.rgb_1ch_encoder = LightweightEncoder(1, self.embedding_dim, dropout)
        self.thermal_encoder = LightweightEncoder(1, self.embedding_dim, dropout)

        self.representation_attention = AttentionAggregator(self.embedding_dim)
        self.rgb_view_attention = AttentionAggregator(self.embedding_dim)
        self.thermal_view_attention = AttentionAggregator(self.embedding_dim)
        self.fusion = AdaptiveRGBThermalFusion(self.embedding_dim)

        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.embedding_dim, 1)
        )

    def encode_representation(self, modality, representation, tensor):
        if modality == "rgb":
            if representation == "rgb_texture":
                return self.rgb_1ch_encoder(tensor)
            return self.rgb_3ch_encoder(tensor)

        if modality == "thermal":
            return self.thermal_encoder(tensor)

        raise ValueError(f"Unsupported modality: {modality}")

    def encode_specialist(self, batch, modality, view):
        reps = self.rgb_reps if modality == "rgb" else self.thermal_reps
        embeddings = []

        for rep in reps:
            tensor = batch["representations"][modality][view][rep]
            emb = self.encode_representation(modality, rep, tensor)
            embeddings.append(emb)

        rep_stack = torch.stack(embeddings, dim=1)
        specialist, rep_weights = self.representation_attention(rep_stack)
        return specialist, rep_weights

    def forward(self, batch):
        rgb_specialists = []
        thermal_specialists = []
        rep_weights = {"rgb": {}, "thermal": {}}

        for view in self.views:
            rgb_spec, rgb_rep_w = self.encode_specialist(batch, "rgb", view)
            th_spec, th_rep_w = self.encode_specialist(batch, "thermal", view)

            rgb_specialists.append(rgb_spec)
            thermal_specialists.append(th_spec)

            rep_weights["rgb"][view] = rgb_rep_w
            rep_weights["thermal"][view] = th_rep_w

        rgb_stack = torch.stack(rgb_specialists, dim=1)
        thermal_stack = torch.stack(thermal_specialists, dim=1)

        rgb_branch, rgb_view_weights = self.rgb_view_attention(rgb_stack)
        thermal_branch, thermal_view_weights = self.thermal_view_attention(thermal_stack)

        fused, modality_weights = self.fusion(rgb_branch, thermal_branch)

        logits = self.classifier(fused).squeeze(-1)
        probs = torch.sigmoid(logits)

        return {
            "logits": logits,
            "probabilities": probs,
            "fused_embedding": fused,
            "rgb_branch_embedding": rgb_branch,
            "thermal_branch_embedding": thermal_branch,
            "modality_weights": modality_weights,
            "rgb_view_weights": rgb_view_weights,
            "thermal_view_weights": thermal_view_weights,
            "representation_weights": rep_weights,
        }


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def validate_forward_pass(model, loader, device, split_name, max_batches=2):
    rows = []

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break

            for modality in ["rgb", "thermal"]:
                for view in batch["representations"][modality].keys():
                    for rep in batch["representations"][modality][view].keys():
                        batch["representations"][modality][view][rep] = (
                            batch["representations"][modality][view][rep].to(device)
                        )

            batch["label"] = batch["label"].to(device)

            out = model(batch)

            rows.append({
                "split": split_name,
                "batch_idx": batch_idx,
                "batch_size": int(batch["label"].shape[0]),
                "logits_shape": str(list(out["logits"].shape)),
                "probabilities_shape": str(list(out["probabilities"].shape)),
                "fused_embedding_shape": str(list(out["fused_embedding"].shape)),
                "rgb_branch_shape": str(list(out["rgb_branch_embedding"].shape)),
                "thermal_branch_shape": str(list(out["thermal_branch_embedding"].shape)),
                "modality_weights_shape": str(list(out["modality_weights"].shape)),
                "rgb_view_weights_shape": str(list(out["rgb_view_weights"].shape)),
                "thermal_view_weights_shape": str(list(out["thermal_view_weights"].shape)),
                "prob_min": float(out["probabilities"].min().item()),
                "prob_max": float(out["probabilities"].max().item()),
                "has_nan": bool(
                    torch.isnan(out["logits"]).any().item()
                    or torch.isnan(out["probabilities"]).any().item()
                    or torch.isnan(out["fused_embedding"]).any().item()
                ),
                "has_inf": bool(
                    torch.isinf(out["logits"]).any().item()
                    or torch.isinf(out["probabilities"]).any().item()
                    or torch.isinf(out["fused_embedding"]).any().item()
                ),
                "mean_rgb_modality_weight": float(out["modality_weights"][:, 0].mean().item()),
                "mean_thermal_modality_weight": float(out["modality_weights"][:, 1].mean().item()),
            })

    return rows


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Missing config file: {CONFIG_FILE}")

    config = load_json(CONFIG_FILE)

    participant_manifest = pd.read_csv(config["paths"]["participant_manifest"])
    representation_manifest = pd.read_csv(config["paths"]["representation_manifest"])
    holdout_split = pd.read_csv(config["paths"]["holdout_split"])

    batch_size = int(config["data"]["batch_size"])
    num_workers = int(config["data"]["num_workers"])

    train_ds = CPMRParticipantDataset(participant_manifest, representation_manifest, holdout_split, "train", config)
    val_ds = CPMRParticipantDataset(participant_manifest, representation_manifest, holdout_split, "val", config)
    test_ds = CPMRParticipantDataset(participant_manifest, representation_manifest, holdout_split, "test", config)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=cpmr_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=cpmr_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=cpmr_collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CPMRNet(config).to(device)

    validation_rows = []
    for split_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        validation_rows.extend(validate_forward_pass(model, loader, device, split_name, max_batches=2))

    validation_df = pd.DataFrame(validation_rows)
    validation_df.to_csv(TABLES_OUT / "forward_pass_validation_summary.csv", index=False)

    module_summary = pd.DataFrame([
        {"module": name, "parameters": count_parameters(module)}
        for name, module in model.named_children()
    ])
    module_summary.to_csv(TABLES_OUT / "model_module_parameter_summary.csv", index=False)

    torch.save(model.state_dict(), MODELS_OUT / "CPMRNet_initial_state.pt")

    summary = {
        "stage": "Stage10J",
        "title": "Complete CPMR-Net Model Implementation",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "architecture": config["project"]["architecture"],
        "embedding_dim": config["model"]["embedding_dim"],
        "batch_size": batch_size,
        "train_participants": len(train_ds),
        "val_participants": len(val_ds),
        "test_participants": len(test_ds),
        "total_parameters": int(count_parameters(model)),
        "trainable_parameters": int(count_trainable_parameters(model)),
        "forward_validation_batches": len(validation_df),
        "nan_detected": bool(validation_df["has_nan"].any()),
        "inf_detected": bool(validation_df["has_inf"].any()),
        "model_state_file": str(MODELS_OUT / "CPMRNet_initial_state.pt"),
        "outputs_saved_to": str(STAGE_OUT),
        "note": "This stage validates model implementation and forward passes only. No supervised training is performed."
    }

    with open(STAGE_OUT / "Stage10J_CPMRNet_Model_Implementation_Summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    report = []
    report.append("# Stage 10J Complete CPMR-Net Model Implementation\n")
    report.append(f"Generated at: {summary['created_at']}\n")
    report.append("## Purpose\n")
    report.append(
        "This stage implements the full trainable CPMR-Net architecture and validates forward passes "
        "on train, validation, and test dataloaders. No supervised optimization is performed in this stage.\n"
    )
    report.append("## Model Components\n")
    report.append("- Lightweight RGB 3-channel encoder")
    report.append("- Lightweight RGB 1-channel texture encoder")
    report.append("- Lightweight thermal encoder")
    report.append("- Representation-level attention")
    report.append("- RGB view attention branch")
    report.append("- Thermal view attention branch")
    report.append("- Adaptive RGB-thermal cooperative fusion")
    report.append("- Binary classification head\n")
    report.append("## Summary\n")
    report.append(f"- Device: {summary['device']}")
    report.append(f"- Train/Val/Test participants: {summary['train_participants']} / {summary['val_participants']} / {summary['test_participants']}")
    report.append(f"- Total parameters: {summary['total_parameters']}")
    report.append(f"- Trainable parameters: {summary['trainable_parameters']}")
    report.append(f"- Forward-validation batches: {summary['forward_validation_batches']}")
    report.append(f"- NaN detected: {summary['nan_detected']}")
    report.append(f"- Inf detected: {summary['inf_detected']}\n")
    report.append("## Output Files\n")
    report.append("- `tables/forward_pass_validation_summary.csv`")
    report.append("- `tables/model_module_parameter_summary.csv`")
    report.append("- `models/CPMRNet_initial_state.pt`")
    report.append("- `Stage10J_CPMRNet_Model_Implementation_Summary.json`\n")
    report.append("## Important Note\n")
    report.append(summary["note"])

    with open(REPORTS_OUT / "Stage10J_CPMRNet_Model_Implementation_Report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("=" * 80)
    print("STAGE 10J COMPLETE CPMR-NET MODEL IMPLEMENTATION COMPLETED")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Train/Val/Test participants: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")
    print(f"Total parameters: {summary['total_parameters']}")
    print(f"Trainable parameters: {summary['trainable_parameters']}")
    print(f"Forward-validation batches: {summary['forward_validation_batches']}")
    print(f"NaN detected: {summary['nan_detected']}")
    print(f"Inf detected: {summary['inf_detected']}")
    print(f"Outputs saved to: {STAGE_OUT}")
    print("=" * 80)


if __name__ == "__main__":
    main()