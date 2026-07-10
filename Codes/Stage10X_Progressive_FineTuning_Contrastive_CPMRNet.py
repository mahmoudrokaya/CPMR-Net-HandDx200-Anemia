# -*- coding: utf-8 -*-
"""
Stage 10X - Progressive Fine-Tuning of Contrastive-Pretrained CPMR-Net

Purpose:
Train CPMR-Net using Stage 10V contrastive-pretrained encoders, but without
multi-task auxiliary losses.

Training strategy:
1. Load contrastive-pretrained encoders.
2. Freeze encoders for warm-up.
3. Unfreeze encoders progressively.
4. Train only for binary anemia classification.

No modality/view/representation auxiliary losses are used.
"""

from pathlib import Path
import json
from datetime import datetime
import copy
import warnings

import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix, brier_score_loss
)

warnings.filterwarnings("ignore")


BASE_DIR = Path(r"D:\47\472\New-Papers\Anemia_Paper")
OUTPUTS_DIR = BASE_DIR / "Outputs"

CONFIG_FILE = OUTPUTS_DIR / "Stage10I4_Training_Configuration_Experiment_Control" / "configs" / "CPMRNet_training_config_v1.json"
LOSS_CONFIG_FILE = OUTPUTS_DIR / "Stage10K_Loss_Functions_Class_Imbalance" / "configs" / "CPMRNet_loss_config_v1.json"
ENCODER_MAP_FILE = OUTPUTS_DIR / "Stage10V_SelfSupervised_Contrastive_Pretraining" / "models" / "contrastive_pretrained_encoder_checkpoint_map.json"

STAGE_OUT = OUTPUTS_DIR / "Stage10X_Progressive_FineTuning_Contrastive_CPMRNet"
TABLES_OUT = STAGE_OUT / "tables"
REPORTS_OUT = STAGE_OUT / "reports"
MODELS_OUT = STAGE_OUT / "models"
PRED_OUT = STAGE_OUT / "predictions"

for p in [TABLES_OUT, REPORTS_OUT, MODELS_OUT, PRED_OUT]:
    p.mkdir(parents=True, exist_ok=True)


FREEZE_ENCODERS_FIRST_EPOCHS = 5
ENCODER_LR_MULTIPLIER = 0.25


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
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
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    else:
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        img = img[:, :, None]

    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))

    return torch.tensor(img, dtype=torch.float32)


class ParticipantDataset(Dataset):
    def __init__(self, participant_manifest, representation_manifest, split_df, split_name, config):
        self.pm = participant_manifest.copy()
        self.rm = representation_manifest.copy()
        self.split_df = split_df[split_df["split"] == split_name].copy()
        self.config = config

        self.modalities = config["data"]["modalities"]
        self.views = config["data"]["views"]
        self.rgb_reps = config["data"]["rgb_representations"]
        self.thermal_reps = config["data"]["thermal_representations"]

        self.pm["participant_id"] = self.pm["participant_id"].astype(str)
        self.rm["participant_id"] = self.rm["participant_id"].astype(str)
        self.participant_ids = self.split_df["participant_id"].astype(str).tolist()

        self.participant_lookup = {
            str(r["participant_id"]): r for _, r in self.pm.iterrows()
        }

        self.rep_groups = {
            pid: g.copy() for pid, g in self.rm.groupby("participant_id")
        }

    def __len__(self):
        return len(self.participant_ids)

    def __getitem__(self, idx):
        pid = self.participant_ids[idx]
        row = self.participant_lookup[pid]
        rep_df = self.rep_groups[pid]

        sample = {
            "participant_id": pid,
            "label": torch.tensor(int(row["label"]), dtype=torch.float32),
            "representations": {"rgb": {}, "thermal": {}}
        }

        for modality in self.modalities:
            reps = self.rgb_reps if modality == "rgb" else self.thermal_reps

            for view in self.views:
                sample["representations"][modality][view] = {}

                view_df = rep_df[
                    (rep_df["modality"] == modality)
                    & (rep_df["view"] == view)
                    & (rep_df["status"] == "saved")
                ]

                for rep in reps:
                    r = view_df[view_df["representation"] == rep]
                    if len(r) == 0:
                        raise RuntimeError(f"Missing representation: {pid} {modality} {view} {rep}")

                    sample["representations"][modality][view][rep] = read_tensor(
                        r.iloc[0]["path"],
                        modality,
                        rep
                    )

        return sample


def collate_fn(batch):
    first = batch[0]

    out = {
        "participant_id": [b["participant_id"] for b in batch],
        "label": torch.stack([b["label"] for b in batch]),
        "representations": {"rgb": {}, "thermal": {}}
    }

    for modality in ["rgb", "thermal"]:
        for view in first["representations"][modality]:
            out["representations"][modality][view] = {}

            for rep in first["representations"][modality][view]:
                out["representations"][modality][view][rep] = torch.stack(
                    [b["representations"][modality][view][rep] for b in batch],
                    dim=0
                )

    return out


def move_batch(batch, device):
    batch["label"] = batch["label"].to(device)

    for modality in ["rgb", "thermal"]:
        for view in batch["representations"][modality]:
            for rep in batch["representations"][modality][view]:
                batch["representations"][modality][view][rep] = (
                    batch["representations"][modality][view][rep].to(device)
                )

    return batch


class LightweightEncoder(nn.Module):
    def __init__(self, in_channels, embedding_dim=128, dropout=0.10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),

            nn.Conv2d(64, 96, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96), nn.ReLU(inplace=True),

            nn.Conv2d(96, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
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
            nn.Linear(embedding_dim // 2, 1),
        )

    def forward(self, x):
        logits = self.score(x).squeeze(-1)
        weights = torch.softmax(logits, dim=1)
        aggregated = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return aggregated, weights


class AdaptiveFusion(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, 2),
        )

    def forward(self, rgb_vec, thermal_vec):
        weights = torch.softmax(self.gate(torch.cat([rgb_vec, thermal_vec], dim=-1)), dim=-1)
        fused = weights[:, 0:1] * rgb_vec + weights[:, 1:2] * thermal_vec
        return fused, weights


class ContrastiveFineTunedCPMRNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embedding_dim = int(config["model"]["embedding_dim"])
        dropout = float(config["model"]["dropout"])

        self.views = config["data"]["views"]
        self.rgb_reps = config["data"]["rgb_representations"]
        self.thermal_reps = config["data"]["thermal_representations"]

        self.rgb_3ch_encoder = LightweightEncoder(3, self.embedding_dim)
        self.rgb_1ch_encoder = LightweightEncoder(1, self.embedding_dim)
        self.thermal_encoder = LightweightEncoder(1, self.embedding_dim)

        self.representation_attention = AttentionAggregator(self.embedding_dim)
        self.rgb_view_attention = AttentionAggregator(self.embedding_dim)
        self.thermal_view_attention = AttentionAggregator(self.embedding_dim)
        self.fusion = AdaptiveFusion(self.embedding_dim)

        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.embedding_dim, 1),
        )

    def load_contrastive_encoders(self, checkpoint_map, device):
        self.rgb_3ch_encoder.load_state_dict(torch.load(checkpoint_map["rgb_3ch"], map_location=device))
        self.rgb_1ch_encoder.load_state_dict(torch.load(checkpoint_map["rgb_1ch"], map_location=device))
        self.thermal_encoder.load_state_dict(torch.load(checkpoint_map["thermal_1ch"], map_location=device))

    def set_encoder_trainable(self, trainable):
        for module in [self.rgb_3ch_encoder, self.rgb_1ch_encoder, self.thermal_encoder]:
            for p in module.parameters():
                p.requires_grad = trainable
            if not trainable:
                module.eval()

    def encoder_parameters(self):
        for module in [self.rgb_3ch_encoder, self.rgb_1ch_encoder, self.thermal_encoder]:
            for p in module.parameters():
                yield p

    def non_encoder_parameters(self):
        modules = [
            self.representation_attention,
            self.rgb_view_attention,
            self.thermal_view_attention,
            self.fusion,
            self.classifier
        ]
        for module in modules:
            for p in module.parameters():
                yield p

    def encode_representation(self, modality, rep, x):
        if modality == "rgb":
            if rep == "rgb_texture":
                return self.rgb_1ch_encoder(x)
            return self.rgb_3ch_encoder(x)
        return self.thermal_encoder(x)

    def encode_specialist(self, batch, modality, view):
        reps = self.rgb_reps if modality == "rgb" else self.thermal_reps
        embeddings = []

        for rep in reps:
            x = batch["representations"][modality][view][rep]
            emb = self.encode_representation(modality, rep, x)
            embeddings.append(emb)

        stack = torch.stack(embeddings, dim=1)
        specialist, rep_weights = self.representation_attention(stack)
        return specialist, rep_weights

    def forward(self, batch):
        rgb_specs = []
        thermal_specs = []

        for view in self.views:
            rgb_spec, _ = self.encode_specialist(batch, "rgb", view)
            thermal_spec, _ = self.encode_specialist(batch, "thermal", view)

            rgb_specs.append(rgb_spec)
            thermal_specs.append(thermal_spec)

        rgb_branch, rgb_view_weights = self.rgb_view_attention(torch.stack(rgb_specs, dim=1))
        thermal_branch, thermal_view_weights = self.thermal_view_attention(torch.stack(thermal_specs, dim=1))

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
        }


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan,
        "pr_auc": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan,
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
    }


def run_epoch(model, loader, optimizer, criterion, device, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    labels_all = []
    probs_all = []
    ids_all = []

    for batch in loader:
        batch = move_batch(batch, device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            out = model(batch)
            loss = criterion(out["logits"], batch["label"])

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    1.0
                )
                optimizer.step()

        bs = batch["label"].shape[0]
        total_loss += float(loss.item()) * bs

        labels_all.extend(batch["label"].detach().cpu().numpy().tolist())
        probs_all.extend(out["probabilities"].detach().cpu().numpy().tolist())
        ids_all.extend(batch["participant_id"])

    metrics = compute_metrics(labels_all, probs_all)

    return {
        "loss": total_loss / len(loader.dataset),
        "metrics": metrics,
        "ids": ids_all,
        "labels": labels_all,
        "probs": probs_all
    }


def make_optimizer(model, base_lr, weight_decay, encoders_trainable):
    if encoders_trainable:
        return torch.optim.AdamW(
            [
                {
                    "params": list(model.non_encoder_parameters()),
                    "lr": base_lr,
                },
                {
                    "params": list(model.encoder_parameters()),
                    "lr": base_lr * ENCODER_LR_MULTIPLIER,
                },
            ],
            weight_decay=weight_decay
        )

    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=base_lr,
        weight_decay=weight_decay
    )


def main():
    config = load_json(CONFIG_FILE)
    loss_config = load_json(LOSS_CONFIG_FILE)
    checkpoint_map = load_json(ENCODER_MAP_FILE)

    set_seed(int(config["optimization"]["random_seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pm = pd.read_csv(config["paths"]["participant_manifest"])
    rm = pd.read_csv(config["paths"]["representation_manifest"])
    split = pd.read_csv(config["paths"]["holdout_split"])

    train_ds = ParticipantDataset(pm, rm, split, "train", config)
    val_ds = ParticipantDataset(pm, rm, split, "val", config)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(config["data"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["data"]["num_workers"]),
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=int(config["data"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"]["num_workers"]),
        collate_fn=collate_fn
    )

    model = ContrastiveFineTunedCPMRNet(config).to(device)
    model.load_contrastive_encoders(checkpoint_map, device)

    model.set_encoder_trainable(False)
    encoders_trainable = False

    base_lr = float(config["optimization"]["learning_rate"])
    weight_decay = float(config["optimization"]["weight_decay"])

    optimizer = make_optimizer(model, base_lr, weight_decay, encoders_trainable)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(config["optimization"]["scheduler_factor"]),
        patience=int(config["optimization"]["scheduler_patience"])
    )

    pos_weight = torch.tensor(
        [float(loss_config["positive_class_weight_for_bce"])],
        dtype=torch.float32,
        device=device
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    max_epochs = int(config["optimization"]["max_epochs"])
    patience = int(config["optimization"]["early_stopping_patience"])

    best_auc = -np.inf
    best_epoch = 0
    patience_counter = 0
    best_state = None
    best_val_predictions = None
    history = []

    parameter_summary = {
        "total_parameters": int(count_parameters(model, False)),
        "trainable_initial": int(count_parameters(model, True)),
    }

    for epoch in range(1, max_epochs + 1):
        if epoch == FREEZE_ENCODERS_FIRST_EPOCHS + 1:
            model.set_encoder_trainable(True)
            encoders_trainable = True
            optimizer = make_optimizer(model, base_lr, weight_decay, encoders_trainable)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=float(config["optimization"]["scheduler_factor"]),
                patience=int(config["optimization"]["scheduler_patience"])
            )
            print("Encoders unfrozen: progressive fine-tuning started.")

        train_out = run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        val_out = run_epoch(model, val_loader, optimizer, criterion, device, train=False)

        val_auc = val_out["metrics"]["roc_auc"]
        scheduler.step(0.0 if np.isnan(val_auc) else val_auc)

        row = {
            "epoch": epoch,
            "encoders_trainable": encoders_trainable,
            "learning_rate_main": optimizer.param_groups[0]["lr"],
            "learning_rate_encoder": optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else 0.0,
            "train_loss": train_out["loss"],
            "val_loss": val_out["loss"],
        }

        for k, v in train_out["metrics"].items():
            row[f"train_{k}"] = v
        for k, v in val_out["metrics"].items():
            row[f"val_{k}"] = v

        history.append(row)

        improved = not np.isnan(val_auc) and val_auc > best_auc

        if improved:
            best_auc = val_auc
            best_epoch = epoch
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())

            best_val_predictions = pd.DataFrame({
                "participant_id": val_out["ids"],
                "label": val_out["labels"],
                "probability": val_out["probs"],
                "prediction_0_5": (np.asarray(val_out["probs"]) >= 0.5).astype(int),
            })

            torch.save(best_state, MODELS_OUT / "ProgressiveContrastive_CPMRNet_best_val_auc.pt")
        else:
            patience_counter += 1

        torch.save(model.state_dict(), MODELS_OUT / "ProgressiveContrastive_CPMRNet_last.pt")

        print(
            f"Epoch {epoch:03d} | "
            f"enc_train={encoders_trainable} | "
            f"train_loss={train_out['loss']:.4f} | "
            f"val_loss={val_out['loss']:.4f} | "
            f"val_auc={val_auc:.4f} | "
            f"val_f1={val_out['metrics']['f1']:.4f} | "
            f"patience={patience_counter}/{patience}"
        )

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    history_df = pd.DataFrame(history)
    history_df.to_csv(TABLES_OUT / "progressive_contrastive_training_history.csv", index=False)

    if best_val_predictions is not None:
        best_val_predictions.to_csv(PRED_OUT / "progressive_contrastive_best_validation_predictions.csv", index=False)

    parameter_summary["trainable_after_unfreeze"] = int(count_parameters(model, True))
    pd.DataFrame([parameter_summary]).to_csv(TABLES_OUT / "progressive_contrastive_parameter_summary.csv", index=False)

    summary = {
        "stage": "Stage10X",
        "title": "Progressive Fine-Tuning of Contrastive-Pretrained CPMR-Net",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "train_participants": int(len(train_ds)),
        "val_participants": int(len(val_ds)),
        "encoder_initialization": "Stage10V contrastive pretrained encoders",
        "auxiliary_losses": False,
        "freeze_first_epochs": FREEZE_ENCODERS_FIRST_EPOCHS,
        "encoder_lr_multiplier": ENCODER_LR_MULTIPLIER,
        "epochs_completed": int(len(history_df)),
        "best_epoch": int(best_epoch),
        "best_val_roc_auc": float(best_auc),
        "best_model_file": str(MODELS_OUT / "ProgressiveContrastive_CPMRNet_best_val_auc.pt"),
        "last_model_file": str(MODELS_OUT / "ProgressiveContrastive_CPMRNet_last.pt"),
        "outputs_saved_to": str(STAGE_OUT)
    }

    with open(STAGE_OUT / "Stage10X_Progressive_FineTuning_Contrastive_CPMRNet_Summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    report = []
    report.append("# Stage 10X Progressive Fine-Tuning of Contrastive-Pretrained CPMR-Net\n")
    report.append(f"Generated at: {summary['created_at']}\n")
    report.append("## Purpose\n")
    report.append(
        "This stage trains CPMR-Net using Stage 10V contrastive-pretrained encoders without auxiliary losses. "
        "Encoders are frozen for a short warm-up period and then unfrozen with a lower encoder learning rate.\n"
    )
    report.append("## Summary\n")
    report.append(f"- Train participants: {summary['train_participants']}")
    report.append(f"- Validation participants: {summary['val_participants']}")
    report.append(f"- Freeze-first epochs: {summary['freeze_first_epochs']}")
    report.append(f"- Encoder LR multiplier: {summary['encoder_lr_multiplier']}")
    report.append(f"- Epochs completed: {summary['epochs_completed']}")
    report.append(f"- Best epoch: {summary['best_epoch']}")
    report.append(f"- Best validation ROC-AUC: {summary['best_val_roc_auc']:.4f}\n")
    report.append("## Output Files\n")
    report.append("- `tables/progressive_contrastive_training_history.csv`")
    report.append("- `tables/progressive_contrastive_parameter_summary.csv`")
    report.append("- `predictions/progressive_contrastive_best_validation_predictions.csv`")
    report.append("- `models/ProgressiveContrastive_CPMRNet_best_val_auc.pt`")
    report.append("- `models/ProgressiveContrastive_CPMRNet_last.pt`")
    report.append("- `Stage10X_Progressive_FineTuning_Contrastive_CPMRNet_Summary.json`")

    with open(REPORTS_OUT / "Stage10X_Progressive_FineTuning_Contrastive_CPMRNet_Report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("=" * 80)
    print("STAGE 10X PROGRESSIVE CONTRASTIVE FINE-TUNING COMPLETED")
    print("=" * 80)
    print(f"Train/Val participants: {len(train_ds)}/{len(val_ds)}")
    print(f"Epochs completed: {summary['epochs_completed']}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation ROC-AUC: {best_auc:.4f}")
    print(f"Outputs saved to: {STAGE_OUT}")
    print("=" * 80)


if __name__ == "__main__":
    main()