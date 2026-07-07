# -*- coding: utf-8 -*-
"""
Stage 10D - Representation Encoder Implementation

Corrected complete version.

Purpose:
- Load encoder_ready_representation_manifest.csv from Stage 10C.
- Implement custom lightweight CNN representation encoders.
- Encode representations into 128-dimensional embeddings.
- Avoid DataLoader tensor-stacking errors by grouping inputs by:
    modality + channel count + spatial size.
- Save embedding arrays, manifests, validation summaries, and initial encoder states.

Important:
The embeddings generated here are produced from initialized encoders for pipeline validation.
They are NOT final trained scientific embeddings. Final embeddings will be learned later
during supervised CPMR-Net training.
"""

from pathlib import Path
import json
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

BASE_DIR = Path(r"D:\47\472\New-Papers\Anemia_Paper")
OUTPUTS_DIR = BASE_DIR / "Outputs"

STAGE10C_DIR = OUTPUTS_DIR / "Stage10C_Representation_Encoder_Dataset_Tensor_Validation"
INPUT_MANIFEST = STAGE10C_DIR / "tables" / "encoder_ready_representation_manifest.csv"

STAGE_OUT = OUTPUTS_DIR / "Stage10D_Representation_Encoder_Implementation"
TABLES_OUT = STAGE_OUT / "tables"
REPORTS_OUT = STAGE_OUT / "reports"
EMBEDDINGS_OUT = STAGE_OUT / "embeddings"
MODELS_OUT = STAGE_OUT / "models"

for p in [TABLES_OUT, REPORTS_OUT, EMBEDDINGS_OUT, MODELS_OUT]:
    p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

EMBEDDING_DIM = 128
BATCH_SIZE = 64
NUM_WORKERS = 0
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class RepresentationDataset(Dataset):
    """
    Dataset for one homogeneous representation group.

    Every item in one dataset must have the same:
    - channel count
    - height
    - width
    """

    def __init__(self, manifest_df, forced_channels):
        self.df = manifest_df.reset_index(drop=True)
        self.forced_channels = int(forced_channels)

    def __len__(self):
        return len(self.df)

    def _read_image(self, path):
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

        if img is None:
            raise RuntimeError(f"Could not read image: {path}")

        if self.forced_channels == 3:
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        elif self.forced_channels == 1:
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img[:, :, None]

        else:
            raise ValueError(f"Unsupported forced_channels: {self.forced_channels}")

        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # C, H, W

        return torch.tensor(img, dtype=torch.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        return {
            "tensor": self._read_image(row["path"]),
            "participant_id": str(row["participant_id"]),
            "label": int(row["label"]),
            "modality": str(row["modality"]),
            "view": str(row["view"]),
            "representation": str(row["representation"]),
            "encoder_input_type": str(row["encoder_input_type"]),
            "path": str(row["path"]),
            "expected_channels": int(row["expected_channels"]),
            "expected_height": int(row["expected_height"]),
            "expected_width": int(row["expected_width"]),
        }


# ---------------------------------------------------------------------
# Lightweight CNN Encoder
# ---------------------------------------------------------------------

class LightweightRepresentationEncoder(nn.Module):
    """
    Compact CNN encoder for representation-level forward-pass validation.

    It supports both 224x224 full images and 112x112 patches because of
    AdaptiveAvgPool2d before projection.
    """

    def __init__(self, in_channels=3, embedding_dim=128, dropout=0.25):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),

            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1, bias=False),
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
        x = self.features(x)
        x = self.projection(x)
        return x


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def make_safe_name(value):
    return str(value).replace(" ", "_").replace("/", "_").replace("\\", "_").replace("-", "_")


def encode_group(df, model, forced_channels, group_name):
    """
    Encode one homogeneous group of representations.

    The group must have identical channel count and spatial size.
    """

    if df.empty:
        empty_records = pd.DataFrame()
        empty_embeddings = np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        return empty_records, empty_embeddings

    dataset = RepresentationDataset(df, forced_channels=forced_channels)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    model.eval()

    records = []
    embeddings = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Encoding {group_name}"):
            x = batch["tensor"].to(DEVICE)

            emb = model(x)
            emb_np = emb.detach().cpu().numpy().astype(np.float32)

            for i in range(emb_np.shape[0]):
                records.append({
                    "participant_id": batch["participant_id"][i],
                    "label": int(batch["label"][i]),
                    "modality": batch["modality"][i],
                    "view": batch["view"][i],
                    "representation": batch["representation"][i],
                    "encoder_input_type": batch["encoder_input_type"][i],
                    "path": batch["path"][i],
                    "expected_channels": int(batch["expected_channels"][i]),
                    "expected_height": int(batch["expected_height"][i]),
                    "expected_width": int(batch["expected_width"][i]),
                    "embedding_group": group_name,
                    "group_embedding_index": len(records),
                })

                embeddings.append(emb_np[i])

    embeddings = np.vstack(embeddings).astype(np.float32)
    records_df = pd.DataFrame(records)

    return records_df, embeddings


def summarize_embeddings(name, records_df, embeddings):
    if embeddings.shape[0] == 0:
        return {
            "embedding_group": name,
            "records": 0,
            "embedding_shape": str(list(embeddings.shape)),
            "has_nan": False,
            "has_inf": False,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }

    return {
        "embedding_group": name,
        "records": int(len(records_df)),
        "embedding_shape": str(list(embeddings.shape)),
        "has_nan": bool(np.isnan(embeddings).any()),
        "has_inf": bool(np.isinf(embeddings).any()),
        "mean": float(np.mean(embeddings)),
        "std": float(np.std(embeddings)),
        "min": float(np.min(embeddings)),
        "max": float(np.max(embeddings)),
    }


def build_encoding_groups(manifest, rgb_3ch_encoder, rgb_1ch_encoder, thermal_encoder):
    """
    Create homogeneous encoding groups.

    Grouping dimensions:
    - modality
    - expected_channels
    - expected_height
    - expected_width

    This prevents batching tensors with incompatible shapes.
    """

    groups = []

    unique_groups = (
        manifest.groupby(["modality", "expected_channels", "expected_height", "expected_width"])
        .size()
        .reset_index(name="count")
        .sort_values(["modality", "expected_channels", "expected_height", "expected_width"])
    )

    for _, row in unique_groups.iterrows():
        modality = row["modality"]
        channels = int(row["expected_channels"])
        height = int(row["expected_height"])
        width = int(row["expected_width"])

        group_df = manifest[
            (manifest["modality"] == modality) &
            (manifest["expected_channels"] == channels) &
            (manifest["expected_height"] == height) &
            (manifest["expected_width"] == width)
        ].copy()

        if modality == "rgb" and channels == 3:
            encoder = rgb_3ch_encoder
        elif modality == "rgb" and channels == 1:
            encoder = rgb_1ch_encoder
        elif modality == "thermal" and channels == 1:
            encoder = thermal_encoder
        else:
            raise ValueError(
                f"Unsupported group: modality={modality}, channels={channels}, "
                f"height={height}, width={width}"
            )

        group_name = f"{modality}_{channels}ch_{height}x{width}"

        groups.append({
            "group_name": group_name,
            "modality": modality,
            "channels": channels,
            "height": height,
            "width": width,
            "count": int(row["count"]),
            "df": group_df,
            "encoder": encoder,
        })

    return groups


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    if not INPUT_MANIFEST.exists():
        raise FileNotFoundError(f"Missing Stage 10C manifest: {INPUT_MANIFEST}")

    manifest = pd.read_csv(INPUT_MANIFEST)

    required_columns = [
        "participant_id",
        "label",
        "modality",
        "view",
        "representation",
        "path",
        "encoder_input_type",
        "expected_channels",
        "expected_height",
        "expected_width",
    ]

    missing_columns = [c for c in required_columns if c not in manifest.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns in encoder manifest: {missing_columns}")

    rgb_3ch_encoder = LightweightRepresentationEncoder(
        in_channels=3,
        embedding_dim=EMBEDDING_DIM,
        dropout=0.25
    ).to(DEVICE)

    rgb_1ch_encoder = LightweightRepresentationEncoder(
        in_channels=1,
        embedding_dim=EMBEDDING_DIM,
        dropout=0.25
    ).to(DEVICE)

    thermal_encoder = LightweightRepresentationEncoder(
        in_channels=1,
        embedding_dim=EMBEDDING_DIM,
        dropout=0.25
    ).to(DEVICE)

    groups = build_encoding_groups(
        manifest,
        rgb_3ch_encoder=rgb_3ch_encoder,
        rgb_1ch_encoder=rgb_1ch_encoder,
        thermal_encoder=thermal_encoder
    )

    group_inventory = pd.DataFrame([
        {
            "embedding_group": g["group_name"],
            "modality": g["modality"],
            "channels": g["channels"],
            "height": g["height"],
            "width": g["width"],
            "records": g["count"]
        }
        for g in groups
    ])

    group_inventory.to_csv(TABLES_OUT / "encoding_group_inventory.csv", index=False)

    all_record_dfs = []
    all_embedding_arrays = []
    group_summaries = []

    for g in groups:
        records_df, embeddings = encode_group(
            df=g["df"],
            model=g["encoder"],
            forced_channels=g["channels"],
            group_name=g["group_name"]
        )

        safe_group_name = make_safe_name(g["group_name"])

        records_df.to_csv(TABLES_OUT / f"{safe_group_name}_embedding_manifest.csv", index=False)
        np.save(EMBEDDINGS_OUT / f"{safe_group_name}_embeddings.npy", embeddings)

        all_record_dfs.append(records_df)
        all_embedding_arrays.append(embeddings)

        group_summaries.append(
            summarize_embeddings(g["group_name"], records_df, embeddings)
        )

    all_records = pd.concat(all_record_dfs, ignore_index=True)
    all_embeddings = np.vstack(all_embedding_arrays).astype(np.float32)

    all_records["global_embedding_index"] = np.arange(len(all_records))

    all_records.to_csv(TABLES_OUT / "all_representation_embedding_manifest.csv", index=False)
    np.save(EMBEDDINGS_OUT / "all_representation_embeddings.npy", all_embeddings)

    group_summaries.append(
        summarize_embeddings("all", all_records, all_embeddings)
    )

    embedding_summary = pd.DataFrame(group_summaries)
    embedding_summary.to_csv(TABLES_OUT / "embedding_validation_summary.csv", index=False)

    representation_counts = (
        all_records.groupby(["modality", "representation", "embedding_group"])
        .size()
        .reset_index(name="encoded_count")
    )
    representation_counts.to_csv(TABLES_OUT / "encoded_representation_count_summary.csv", index=False)

    participant_counts = (
        all_records.groupby(["participant_id", "modality"])
        .size()
        .reset_index(name="encoded_count")
    )
    participant_counts.to_csv(TABLES_OUT / "participant_encoded_representation_counts.csv", index=False)

    participant_total_counts = (
        all_records.groupby("participant_id")
        .size()
        .reset_index(name="total_encoded_representations")
    )

    participant_total_counts["expected_total_encoded_representations"] = 64
    participant_total_counts["complete_encoding"] = (
        participant_total_counts["total_encoded_representations"]
        == participant_total_counts["expected_total_encoded_representations"]
    )

    participant_total_counts.to_csv(TABLES_OUT / "participant_encoding_completeness.csv", index=False)

    torch.save(
        rgb_3ch_encoder.state_dict(),
        MODELS_OUT / "lightweight_rgb_3channel_encoder_initial_state.pt"
    )

    torch.save(
        rgb_1ch_encoder.state_dict(),
        MODELS_OUT / "lightweight_rgb_1channel_encoder_initial_state.pt"
    )

    torch.save(
        thermal_encoder.state_dict(),
        MODELS_OUT / "lightweight_thermal_1channel_encoder_initial_state.pt"
    )

    total_params = (
        count_parameters(rgb_3ch_encoder)
        + count_parameters(rgb_1ch_encoder)
        + count_parameters(thermal_encoder)
    )

    summary = {
        "stage": "Stage10D",
        "title": "Representation Encoder Implementation",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_manifest": str(INPUT_MANIFEST),
        "device": str(DEVICE),
        "embedding_dim": EMBEDDING_DIM,
        "batch_size": BATCH_SIZE,
        "encoder_type": "Custom lightweight CNN",
        "grouping_strategy": "modality + expected_channels + expected_height + expected_width",
        "encoding_groups": group_inventory.to_dict(orient="records"),
        "total_input_representations": int(len(manifest)),
        "total_encoded_representations": int(len(all_records)),
        "all_embeddings_shape": list(all_embeddings.shape),
        "participants_encoded": int(all_records["participant_id"].nunique()),
        "participants_complete_encoding": int(participant_total_counts["complete_encoding"].sum()),
        "participants_incomplete_encoding": int((~participant_total_counts["complete_encoding"]).sum()),
        "rgb_3channel_encoder_parameters": int(count_parameters(rgb_3ch_encoder)),
        "rgb_1channel_encoder_parameters": int(count_parameters(rgb_1ch_encoder)),
        "thermal_1channel_encoder_parameters": int(count_parameters(thermal_encoder)),
        "total_encoder_parameters": int(total_params),
        "has_nan": bool(np.isnan(all_embeddings).any()),
        "has_inf": bool(np.isinf(all_embeddings).any()),
        "note": (
            "These embeddings are generated from initialized encoders for pipeline validation only. "
            "They are not final trained scientific embeddings."
        ),
        "outputs_saved_to": str(STAGE_OUT)
    }

    with open(STAGE_OUT / "Stage10D_Representation_Encoder_Implementation_Summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    report = []
    report.append("# Stage 10D Representation Encoder Implementation\n")
    report.append(f"Generated at: {summary['created_at']}\n")

    report.append("## Purpose\n")
    report.append(
        "This stage implements the representation encoder validation pipeline for CPMR-Net. "
        "It generates 128-dimensional embeddings for every encoder-ready representation using "
        "custom lightweight CNN encoders.\n"
    )

    report.append("## Important Correction\n")
    report.append(
        "Representations are grouped by modality, channel count, height, and width before batching. "
        "This prevents PyTorch DataLoader errors caused by mixing full images and patches or mixing "
        "one-channel and three-channel inputs in the same batch.\n"
    )

    report.append("## Encoder Configuration\n")
    report.append(f"- Encoder type: {summary['encoder_type']}")
    report.append(f"- Embedding dimension: {summary['embedding_dim']}")
    report.append(f"- Device: {summary['device']}")
    report.append(f"- Grouping strategy: {summary['grouping_strategy']}")
    report.append(f"- RGB 3-channel encoder parameters: {summary['rgb_3channel_encoder_parameters']}")
    report.append(f"- RGB 1-channel encoder parameters: {summary['rgb_1channel_encoder_parameters']}")
    report.append(f"- Thermal 1-channel encoder parameters: {summary['thermal_1channel_encoder_parameters']}")
    report.append(f"- Total encoder parameters: {summary['total_encoder_parameters']}\n")

    report.append("## Encoding Groups\n")
    for g in summary["encoding_groups"]:
        report.append(
            f"- {g['embedding_group']}: {g['records']} records "
            f"({g['channels']} channels, {g['height']}x{g['width']})"
        )

    report.append("\n## Output Summary\n")
    report.append(f"- Total input representations: {summary['total_input_representations']}")
    report.append(f"- Total encoded representations: {summary['total_encoded_representations']}")
    report.append(f"- All embeddings shape: {summary['all_embeddings_shape']}")
    report.append(f"- Participants encoded: {summary['participants_encoded']}")
    report.append(f"- Participants complete encoding: {summary['participants_complete_encoding']}")
    report.append(f"- Participants incomplete encoding: {summary['participants_incomplete_encoding']}")
    report.append(f"- NaN detected: {summary['has_nan']}")
    report.append(f"- Inf detected: {summary['has_inf']}\n")

    report.append("## Important Note\n")
    report.append(summary["note"] + "\n")

    report.append("## Output Files\n")
    report.append("- `tables/encoding_group_inventory.csv`")
    report.append("- `tables/all_representation_embedding_manifest.csv`")
    report.append("- `tables/embedding_validation_summary.csv`")
    report.append("- `tables/encoded_representation_count_summary.csv`")
    report.append("- `tables/participant_encoded_representation_counts.csv`")
    report.append("- `tables/participant_encoding_completeness.csv`")
    report.append("- `embeddings/all_representation_embeddings.npy`")
    report.append("- group-specific embedding `.npy` files")
    report.append("- group-specific embedding manifest `.csv` files")
    report.append("- `models/lightweight_rgb_3channel_encoder_initial_state.pt`")
    report.append("- `models/lightweight_rgb_1channel_encoder_initial_state.pt`")
    report.append("- `models/lightweight_thermal_1channel_encoder_initial_state.pt`")
    report.append("- `Stage10D_Representation_Encoder_Implementation_Summary.json`")

    with open(REPORTS_OUT / "Stage10D_Representation_Encoder_Implementation_Report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("=" * 80)
    print("STAGE 10D REPRESENTATION ENCODER IMPLEMENTATION COMPLETED")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Total input representations: {summary['total_input_representations']}")
    print(f"Total encoded representations: {summary['total_encoded_representations']}")
    print(f"All embeddings shape: {all_embeddings.shape}")
    print(f"Participants encoded: {summary['participants_encoded']}")
    print(f"Participants complete encoding: {summary['participants_complete_encoding']}")
    print(f"Participants incomplete encoding: {summary['participants_incomplete_encoding']}")
    print(f"NaN detected: {summary['has_nan']}")
    print(f"Inf detected: {summary['has_inf']}")
    print(f"Total encoder parameters: {summary['total_encoder_parameters']}")
    print(f"Outputs saved to: {STAGE_OUT}")
    print("=" * 80)
    print("NOTE: These embeddings are for encoder pipeline validation only, not final trained scientific embeddings.")


if __name__ == "__main__":
    main()