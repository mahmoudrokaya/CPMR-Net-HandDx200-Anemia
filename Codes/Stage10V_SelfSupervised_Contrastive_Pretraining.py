# -*- coding: utf-8 -*-
"""
Stage 10V - Self-Supervised Contrastive Pretraining on HandDx Representations

Purpose:
Pretrain representation encoders using participant-level contrastive learning.

Core idea:
- Two representations from the same participant are treated as a positive pair.
- Representations from different participants are treated as negatives.
- No anemia labels are used for the contrastive objective.

Outputs:
- Pretrained RGB 3-channel encoder
- Pretrained RGB 1-channel texture encoder
- Pretrained thermal encoder
- Training history
- Encoder checkpoint for later CPMR-Net supervised training
"""

from pathlib import Path
import json
from datetime import datetime
import random
import warnings

import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

BASE_DIR = Path(r"D:\47\472\New-Papers\Anemia_Paper")
OUTPUTS_DIR = BASE_DIR / "Outputs"

CONFIG_FILE = (
    OUTPUTS_DIR
    / "Stage10I4_Training_Configuration_Experiment_Control"
    / "configs"
    / "CPMRNet_training_config_v1.json"
)

STAGE_OUT = OUTPUTS_DIR / "Stage10V_SelfSupervised_Contrastive_Pretraining"
TABLES_OUT = STAGE_OUT / "tables"
REPORTS_OUT = STAGE_OUT / "reports"
MODELS_OUT = STAGE_OUT / "models"

for p in [TABLES_OUT, REPORTS_OUT, MODELS_OUT]:
    p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------

RANDOM_SEED = 42
BATCH_SIZE = 64
NUM_WORKERS = 0

PRETRAIN_EPOCHS = 80
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

EMBEDDING_DIM = 128
PROJECTION_DIM = 128
TEMPERATURE = 0.20

EARLY_STOP_PATIENCE = 12

SUPPORTED_ENCODER_GROUPS = [
    "rgb_3ch",
    "rgb_1ch",
    "thermal_1ch",
]


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model, trainable_only=False):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def infer_encoder_group(modality, representation):
    if modality == "rgb":
        if representation == "rgb_texture":
            return "rgb_1ch"
        return "rgb_3ch"
    return "thermal_1ch"


def read_tensor(path, modality, representation):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if img is None:
        raise RuntimeError(f"Could not read image: {path}")

    group = infer_encoder_group(modality, representation)

    if group == "rgb_3ch":
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img[:, :, None]

    # Critical fix: force all representations in each batch to same size
    if group == "rgb_3ch":
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    else:
        img_2d = img[:, :, 0]
        img_2d = cv2.resize(img_2d, (224, 224), interpolation=cv2.INTER_AREA)
        img = img_2d[:, :, None]

    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))

    return torch.tensor(img, dtype=torch.float32)


def simple_tensor_augmentation(x):
    """
    Mild augmentation for contrastive pretraining.
    Works for both 1-channel and 3-channel tensors.
    """

    # horizontal flip
    if random.random() < 0.5:
        x = torch.flip(x, dims=[2])

    # mild brightness scaling
    if random.random() < 0.3:
        scale = 0.90 + random.random() * 0.20
        x = torch.clamp(x * scale, 0.0, 1.0)

    # mild Gaussian noise
    if random.random() < 0.2:
        noise = torch.randn_like(x) * 0.015
        x = torch.clamp(x + noise, 0.0, 1.0)

    return x


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class ContrastiveRepresentationPairDataset(Dataset):
    """
    Samples two representations from the same participant and same encoder group.

    Example:
    - anchor: participant P, rgb_3ch representation A
    - positive: participant P, rgb_3ch representation B

    Negatives are other samples in the batch.
    """

    def __init__(self, representation_manifest, encoder_group):
        self.manifest = representation_manifest.copy()
        self.encoder_group = encoder_group

        self.manifest["encoder_group"] = self.manifest.apply(
            lambda r: infer_encoder_group(r["modality"], r["representation"]),
            axis=1
        )

        self.manifest = self.manifest[
            (self.manifest["status"] == "saved")
            & (self.manifest["encoder_group"] == encoder_group)
        ].copy()

        if self.manifest.empty:
            raise ValueError(f"No representations found for encoder group: {encoder_group}")

        self.participant_groups = {
            str(pid): g.reset_index(drop=True)
            for pid, g in self.manifest.groupby("participant_id")
            if len(g) >= 2
        }

        self.participant_ids = sorted(list(self.participant_groups.keys()))

        if len(self.participant_ids) < 2:
            raise ValueError(f"Not enough participants for contrastive pretraining: {encoder_group}")

    def __len__(self):
        return len(self.participant_ids) * 8

    def __getitem__(self, idx):
        pid = self.participant_ids[idx % len(self.participant_ids)]
        g = self.participant_groups[pid]

        i, j = np.random.choice(len(g), size=2, replace=False)

        row_i = g.iloc[i]
        row_j = g.iloc[j]

        x1 = read_tensor(row_i["path"], row_i["modality"], row_i["representation"])
        x2 = read_tensor(row_j["path"], row_j["modality"], row_j["representation"])

        x1 = simple_tensor_augmentation(x1)
        x2 = simple_tensor_augmentation(x2)

        return {
            "participant_id": pid,
            "x1": x1,
            "x2": x2,
            "encoder_group": self.encoder_group,
        }


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

class LightweightEncoder(nn.Module):
    def __init__(self, in_channels, embedding_dim=128, dropout=0.10):
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


class ContrastiveProjectionHead(nn.Module):
    def __init__(self, embedding_dim=128, projection_dim=128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, projection_dim)
        )

    def forward(self, x):
        return self.head(x)


class ContrastiveEncoder(nn.Module):
    def __init__(self, in_channels, embedding_dim=128, projection_dim=128):
        super().__init__()
        self.encoder = LightweightEncoder(in_channels, embedding_dim)
        self.projector = ContrastiveProjectionHead(embedding_dim, projection_dim)

    def forward(self, x):
        embedding = self.encoder(x)
        z = self.projector(embedding)
        z = F.normalize(z, dim=1)
        return embedding, z


def nt_xent_loss(z1, z2, temperature=0.2):
    """
    SimCLR NT-Xent loss.
    """

    batch_size = z1.shape[0]

    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / temperature

    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim = sim.masked_fill(mask, -9e15)

    positives = torch.cat([
        torch.arange(batch_size, 2 * batch_size),
        torch.arange(0, batch_size)
    ]).to(z.device)

    loss = F.cross_entropy(sim, positives)

    return loss


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def train_encoder_group(encoder_group, representation_manifest, device):
    in_channels = 3 if encoder_group == "rgb_3ch" else 1

    dataset = ContrastiveRepresentationPairDataset(
        representation_manifest=representation_manifest,
        encoder_group=encoder_group
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True
    )

    model = ContrastiveEncoder(
        in_channels=in_channels,
        embedding_dim=EMBEDDING_DIM,
        projection_dim=PROJECTION_DIM
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    best_loss = np.inf
    best_epoch = 0
    patience_counter = 0
    history = []

    for epoch in range(1, PRETRAIN_EPOCHS + 1):
        model.train()

        epoch_losses = []

        for batch in loader:
            x1 = batch["x1"].to(device)
            x2 = batch["x2"].to(device)

            optimizer.zero_grad(set_to_none=True)

            _, z1 = model(x1)
            _, z2 = model(x2)

            loss = nt_xent_loss(z1, z2, TEMPERATURE)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(float(loss.item()))

        mean_loss = float(np.mean(epoch_losses))

        improved = mean_loss < best_loss

        if improved:
            best_loss = mean_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                model.encoder.state_dict(),
                MODELS_OUT / f"{encoder_group}_contrastive_pretrained_encoder.pt"
            )
            torch.save(
                model.state_dict(),
                MODELS_OUT / f"{encoder_group}_contrastive_full_model.pt"
            )
        else:
            patience_counter += 1

        history.append({
            "encoder_group": encoder_group,
            "epoch": epoch,
            "contrastive_loss": mean_loss,
            "best_loss_so_far": best_loss,
            "patience_counter": patience_counter,
            "num_pairs_dataset": len(dataset),
            "num_batches": len(loader),
        })

        print(
            f"{encoder_group} | Epoch {epoch:03d} | "
            f"loss={mean_loss:.4f} | best={best_loss:.4f} | "
            f"patience={patience_counter}/{EARLY_STOP_PATIENCE}"
        )

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"Early stopping for {encoder_group} at epoch {epoch}.")
            break

    result = {
        "encoder_group": encoder_group,
        "in_channels": in_channels,
        "participants": len(dataset.participant_ids),
        "dataset_pairs_per_epoch": len(dataset),
        "batches_per_epoch": len(loader),
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_contrastive_loss": best_loss,
        "encoder_checkpoint": str(MODELS_OUT / f"{encoder_group}_contrastive_pretrained_encoder.pt"),
        "full_model_checkpoint": str(MODELS_OUT / f"{encoder_group}_contrastive_full_model.pt"),
        "total_parameters": count_parameters(model),
        "encoder_parameters": count_parameters(model.encoder),
    }

    return result, pd.DataFrame(history)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    set_seed(RANDOM_SEED)

    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Missing config file: {CONFIG_FILE}")

    config = load_json(CONFIG_FILE)

    representation_manifest_path = Path(config["paths"]["representation_manifest"])

    if not representation_manifest_path.exists():
        raise FileNotFoundError(f"Missing representation manifest: {representation_manifest_path}")

    representation_manifest = pd.read_csv(representation_manifest_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = []
    histories = []

    for group in SUPPORTED_ENCODER_GROUPS:
        print("=" * 80)
        print(f"Training contrastive encoder group: {group}")
        print("=" * 80)

        result, history = train_encoder_group(
            encoder_group=group,
            representation_manifest=representation_manifest,
            device=device
        )

        results.append(result)
        histories.append(history)

    results_df = pd.DataFrame(results)
    history_df = pd.concat(histories, ignore_index=True)

    results_df.to_csv(TABLES_OUT / "contrastive_pretraining_encoder_results.csv", index=False)
    history_df.to_csv(TABLES_OUT / "contrastive_pretraining_history.csv", index=False)

    checkpoint_config = {
        "stage": "Stage10V",
        "title": "Self-Supervised Contrastive Pretraining on HandDx Representations",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "objective": "participant-level contrastive pretraining",
        "temperature": TEMPERATURE,
        "embedding_dim": EMBEDDING_DIM,
        "projection_dim": PROJECTION_DIM,
        "pretrain_epochs": PRETRAIN_EPOCHS,
        "batch_size": BATCH_SIZE,
        "encoder_checkpoints": {
            row["encoder_group"]: row["encoder_checkpoint"]
            for row in results
        },
        "results": results,
        "outputs_saved_to": str(STAGE_OUT)
    }

    with open(
        STAGE_OUT / "Stage10V_SelfSupervised_Contrastive_Pretraining_Summary.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(checkpoint_config, f, indent=4, ensure_ascii=False)

    with open(
        MODELS_OUT / "contrastive_pretrained_encoder_checkpoint_map.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(checkpoint_config["encoder_checkpoints"], f, indent=4, ensure_ascii=False)

    report = []
    report.append("# Stage 10V Self-Supervised Contrastive Pretraining\n")
    report.append(f"Generated at: {checkpoint_config['created_at']}\n")

    report.append("## Purpose\n")
    report.append(
        "This stage pretrains the representation encoders using participant-level contrastive learning. "
        "Positive pairs are two representations from the same participant and encoder group. "
        "Other samples in the batch serve as negatives. No anemia labels are used.\n"
    )

    report.append("## Encoder Groups\n")
    for row in results:
        report.append(f"### {row['encoder_group']}")
        report.append(f"- Participants: {row['participants']}")
        report.append(f"- Epochs completed: {row['epochs_completed']}")
        report.append(f"- Best epoch: {row['best_epoch']}")
        report.append(f"- Best contrastive loss: {row['best_contrastive_loss']:.4f}")
        report.append(f"- Encoder checkpoint: `{row['encoder_checkpoint']}`\n")

    report.append("## Output Files\n")
    report.append("- `tables/contrastive_pretraining_encoder_results.csv`")
    report.append("- `tables/contrastive_pretraining_history.csv`")
    report.append("- `models/rgb_3ch_contrastive_pretrained_encoder.pt`")
    report.append("- `models/rgb_1ch_contrastive_pretrained_encoder.pt`")
    report.append("- `models/thermal_1ch_contrastive_pretrained_encoder.pt`")
    report.append("- `models/contrastive_pretrained_encoder_checkpoint_map.json`")
    report.append("- `Stage10V_SelfSupervised_Contrastive_Pretraining_Summary.json`\n")

    report.append("## Implementation Role\n")
    report.append(
        "The saved encoder checkpoints should be loaded into CPMR-Net in the next supervised stage. "
        "The next model should compare random initialization versus contrastive-pretrained initialization."
    )

    with open(REPORTS_OUT / "Stage10V_SelfSupervised_Contrastive_Pretraining_Report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("=" * 80)
    print("STAGE 10V SELF-SUPERVISED CONTRASTIVE PRETRAINING COMPLETED")
    print("=" * 80)
    print(f"Device: {device}")
    for row in results:
        print(
            f"{row['encoder_group']}: "
            f"epochs={row['epochs_completed']}, "
            f"best_loss={row['best_contrastive_loss']:.4f}, "
            f"checkpoint={row['encoder_checkpoint']}"
        )
    print(f"Outputs saved to: {STAGE_OUT}")
    print("=" * 80)


if __name__ == "__main__":
    main()