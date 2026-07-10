# -*- coding: utf-8 -*-
"""
Stage 10W - Multi-Task Supervised CPMR-Net Training

Main task:
- Binary anemia classification.

Auxiliary tasks:
- Modality prediction: RGB vs thermal.
- View prediction: left dorsal, left palmar, right dorsal, right palmar.
- Representation prediction: representation type.

Encoder initialization:
- Loads Stage 10V contrastive-pretrained encoders.
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

STAGE_OUT = OUTPUTS_DIR / "Stage10W_MultiTask_Supervised_CPMRNet"
TABLES_OUT = STAGE_OUT / "tables"
REPORTS_OUT = STAGE_OUT / "reports"
MODELS_OUT = STAGE_OUT / "models"
PRED_OUT = STAGE_OUT / "predictions"

for p in [TABLES_OUT, REPORTS_OUT, MODELS_OUT, PRED_OUT]:
    p.mkdir(parents=True, exist_ok=True)


AUX_WEIGHT_MODALITY = 0.10
AUX_WEIGHT_VIEW = 0.10
AUX_WEIGHT_REPRESENTATION = 0.10

FREEZE_ENCODERS_FIRST_EPOCHS = 5


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

    if group == "rgb_3ch":
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    else:
        img2 = img[:, :, 0]
        img2 = cv2.resize(img2, (224, 224), interpolation=cv2.INTER_AREA)
        img = img2[:, :, None]

    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return torch.tensor(img, dtype=torch.float32)


class ParticipantDataset(Dataset):
    def __init__(self, participant_manifest, representation_manifest, split_df, split_name, config, rep_to_idx):
        self.pm = participant_manifest.copy()
        self.rm = representation_manifest.copy()
        self.split_df = split_df[split_df["split"] == split_name].copy()
        self.config = config
        self.rep_to_idx = rep_to_idx

        self.modalities = config["data"]["modalities"]
        self.views = config["data"]["views"]
        self.rgb_reps = config["data"]["rgb_representations"]
        self.thermal_reps = config["data"]["thermal_representations"]

        self.view_to_idx = {v: i for i, v in enumerate(self.views)}
        self.modality_to_idx = {"rgb": 0, "thermal": 1}

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
            "representations": {"rgb": {}, "thermal": {}},
            "aux": {"rgb": {}, "thermal": {}}
        }

        for modality in self.modalities:
            reps = self.rgb_reps if modality == "rgb" else self.thermal_reps

            for view in self.views:
                sample["representations"][modality][view] = {}
                sample["aux"][modality][view] = {}

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
                        r.iloc[0]["path"], modality, rep
                    )

                    sample["aux"][modality][view][rep] = {
                        "modality": torch.tensor(self.modality_to_idx[modality], dtype=torch.long),
                        "view": torch.tensor(self.view_to_idx[view], dtype=torch.long),
                        "representation": torch.tensor(self.rep_to_idx[f"{modality}:{rep}"], dtype=torch.long),
                    }

        return sample


def collate_fn(batch):
    first = batch[0]
    out = {
        "participant_id": [b["participant_id"] for b in batch],
        "label": torch.stack([b["label"] for b in batch]),
        "representations": {"rgb": {}, "thermal": {}},
        "aux": {"rgb": {}, "thermal": {}}
    }

    for modality in ["rgb", "thermal"]:
        for view in first["representations"][modality]:
            out["representations"][modality][view] = {}
            out["aux"][modality][view] = {}

            for rep in first["representations"][modality][view]:
                out["representations"][modality][view][rep] = torch.stack(
                    [b["representations"][modality][view][rep] for b in batch],
                    dim=0
                )

                out["aux"][modality][view][rep] = {
                    "modality": torch.stack([b["aux"][modality][view][rep]["modality"] for b in batch]),
                    "view": torch.stack([b["aux"][modality][view][rep]["view"] for b in batch]),
                    "representation": torch.stack([b["aux"][modality][view][rep]["representation"] for b in batch]),
                }

    return out


def move_batch(batch, device):
    batch["label"] = batch["label"].to(device)

    for modality in ["rgb", "thermal"]:
        for view in batch["representations"][modality]:
            for rep in batch["representations"][modality][view]:
                batch["representations"][modality][view][rep] = batch["representations"][modality][view][rep].to(device)
                for key in ["modality", "view", "representation"]:
                    batch["aux"][modality][view][rep][key] = batch["aux"][modality][view][rep][key].to(device)

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
            nn.Linear(embedding_dim // 2, 1)
        )

    def forward(self, x):
        logits = self.score(x).squeeze(-1)
        weights = torch.softmax(logits, dim=1)
        return torch.sum(x * weights.unsqueeze(-1), dim=1), weights


class AdaptiveFusion(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, 2)
        )

    def forward(self, rgb_vec, thermal_vec):
        weights = torch.softmax(self.gate(torch.cat([rgb_vec, thermal_vec], dim=-1)), dim=-1)
        fused = weights[:, 0:1] * rgb_vec + weights[:, 1:2] * thermal_vec
        return fused, weights


class MultiTaskCPMRNet(nn.Module):
    def __init__(self, config, num_rep_classes):
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
            nn.Linear(self.embedding_dim, 1)
        )

        self.modality_head = nn.Linear(self.embedding_dim, 2)
        self.view_head = nn.Linear(self.embedding_dim, len(self.views))
        self.representation_head = nn.Linear(self.embedding_dim, num_rep_classes)

    def load_contrastive_encoders(self, checkpoint_map, device):
        mapping = {
            "rgb_3ch": self.rgb_3ch_encoder,
            "rgb_1ch": self.rgb_1ch_encoder,
            "thermal_1ch": self.thermal_encoder,
        }

        for key, module in mapping.items():
            state = torch.load(checkpoint_map[key], map_location=device)
            module.load_state_dict(state)

    def set_encoder_trainable(self, trainable):
        for module in [self.rgb_3ch_encoder, self.rgb_1ch_encoder, self.thermal_encoder]:
            for p in module.parameters():
                p.requires_grad = trainable
            if not trainable:
                module.eval()

    def encode_representation(self, modality, rep, x):
        if modality == "rgb":
            if rep == "rgb_texture":
                return self.rgb_1ch_encoder(x)
            return self.rgb_3ch_encoder(x)
        return self.thermal_encoder(x)

    def encode_specialist(self, batch, modality, view, aux_outputs):
        reps = self.rgb_reps if modality == "rgb" else self.thermal_reps
        embeddings = []

        for rep in reps:
            x = batch["representations"][modality][view][rep]
            emb = self.encode_representation(modality, rep, x)
            embeddings.append(emb)

            aux_outputs["modality_logits"].append(self.modality_head(emb))
            aux_outputs["view_logits"].append(self.view_head(emb))
            aux_outputs["representation_logits"].append(self.representation_head(emb))

            aux_outputs["modality_targets"].append(batch["aux"][modality][view][rep]["modality"])
            aux_outputs["view_targets"].append(batch["aux"][modality][view][rep]["view"])
            aux_outputs["representation_targets"].append(batch["aux"][modality][view][rep]["representation"])

        stack = torch.stack(embeddings, dim=1)
        specialist, rep_weights = self.representation_attention(stack)
        return specialist, rep_weights

    def forward(self, batch):
        rgb_specs = []
        thermal_specs = []

        aux_outputs = {
            "modality_logits": [],
            "view_logits": [],
            "representation_logits": [],
            "modality_targets": [],
            "view_targets": [],
            "representation_targets": []
        }

        for view in self.views:
            rgb_spec, _ = self.encode_specialist(batch, "rgb", view, aux_outputs)
            th_spec, _ = self.encode_specialist(batch, "thermal", view, aux_outputs)
            rgb_specs.append(rgb_spec)
            thermal_specs.append(th_spec)

        rgb_branch, rgb_view_weights = self.rgb_view_attention(torch.stack(rgb_specs, dim=1))
        thermal_branch, thermal_view_weights = self.thermal_view_attention(torch.stack(thermal_specs, dim=1))

        fused, modality_weights = self.fusion(rgb_branch, thermal_branch)

        logits = self.classifier(fused).squeeze(-1)
        probs = torch.sigmoid(logits)

        for k in ["modality_logits", "view_logits", "representation_logits"]:
            aux_outputs[k] = torch.cat(aux_outputs[k], dim=0)

        for k in ["modality_targets", "view_targets", "representation_targets"]:
            aux_outputs[k] = torch.cat(aux_outputs[k], dim=0)

        return {
            "logits": logits,
            "probabilities": probs,
            "modality_weights": modality_weights,
            "rgb_view_weights": rgb_view_weights,
            "thermal_view_weights": thermal_view_weights,
            "aux": aux_outputs
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


def run_epoch(model, loader, optimizer, device, bce_loss, train=True):
    ce = nn.CrossEntropyLoss()

    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_main = 0.0
    total_mod = 0.0
    total_view = 0.0
    total_rep = 0.0

    labels_all, probs_all, ids_all = [], [], []

    for batch in loader:
        batch = move_batch(batch, device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            out = model(batch)

            main_loss = bce_loss(out["logits"], batch["label"])
            modality_loss = ce(out["aux"]["modality_logits"], out["aux"]["modality_targets"])
            view_loss = ce(out["aux"]["view_logits"], out["aux"]["view_targets"])
            rep_loss = ce(out["aux"]["representation_logits"], out["aux"]["representation_targets"])

            loss = (
                main_loss
                + AUX_WEIGHT_MODALITY * modality_loss
                + AUX_WEIGHT_VIEW * view_loss
                + AUX_WEIGHT_REPRESENTATION * rep_loss
            )

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()

        bs = batch["label"].shape[0]
        total_loss += float(loss.item()) * bs
        total_main += float(main_loss.item()) * bs
        total_mod += float(modality_loss.item()) * bs
        total_view += float(view_loss.item()) * bs
        total_rep += float(rep_loss.item()) * bs

        labels_all.extend(batch["label"].detach().cpu().numpy().tolist())
        probs_all.extend(out["probabilities"].detach().cpu().numpy().tolist())
        ids_all.extend(batch["participant_id"])

    n = len(loader.dataset)
    metrics = compute_metrics(labels_all, probs_all)

    return {
        "loss": total_loss / n,
        "main_loss": total_main / n,
        "modality_loss": total_mod / n,
        "view_loss": total_view / n,
        "representation_loss": total_rep / n,
        "metrics": metrics,
        "ids": ids_all,
        "labels": labels_all,
        "probs": probs_all
    }


def main():
    config = load_json(CONFIG_FILE)
    loss_config = load_json(LOSS_CONFIG_FILE)
    checkpoint_map = load_json(ENCODER_MAP_FILE)

    set_seed(int(config["optimization"]["random_seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pm = pd.read_csv(config["paths"]["participant_manifest"])
    rm = pd.read_csv(config["paths"]["representation_manifest"])
    split = pd.read_csv(config["paths"]["holdout_split"])

    rep_keys = []
    for rep in config["data"]["rgb_representations"]:
        rep_keys.append(f"rgb:{rep}")
    for rep in config["data"]["thermal_representations"]:
        rep_keys.append(f"thermal:{rep}")

    rep_to_idx = {r: i for i, r in enumerate(sorted(rep_keys))}

    train_ds = ParticipantDataset(pm, rm, split, "train", config, rep_to_idx)
    val_ds = ParticipantDataset(pm, rm, split, "val", config, rep_to_idx)

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

    model = MultiTaskCPMRNet(config, num_rep_classes=len(rep_to_idx)).to(device)
    model.load_contrastive_encoders(checkpoint_map, device)

    model.set_encoder_trainable(False)

    pos_weight = torch.tensor(
        [float(loss_config["positive_class_weight_for_bce"])],
        dtype=torch.float32,
        device=device
    )
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(config["optimization"]["learning_rate"]),
        weight_decay=float(config["optimization"]["weight_decay"])
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(config["optimization"]["scheduler_factor"]),
        patience=int(config["optimization"]["scheduler_patience"])
    )

    max_epochs = int(config["optimization"]["max_epochs"])
    patience = int(config["optimization"]["early_stopping_patience"])

    best_auc = -np.inf
    best_epoch = 0
    patience_counter = 0
    best_state = None
    best_val_predictions = None
    history = []

    for epoch in range(1, max_epochs + 1):
        if epoch == FREEZE_ENCODERS_FIRST_EPOCHS + 1:
            model.set_encoder_trainable(True)
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=float(config["optimization"]["learning_rate"]) * 0.5,
                weight_decay=float(config["optimization"]["weight_decay"])
            )
            print("Encoders unfrozen for progressive fine-tuning.")

        train_out = run_epoch(model, train_loader, optimizer, device, bce_loss, train=True)
        val_out = run_epoch(model, val_loader, optimizer, device, bce_loss, train=False)

        val_auc = val_out["metrics"]["roc_auc"]
        scheduler.step(0.0 if np.isnan(val_auc) else val_auc)

        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "encoders_trainable": epoch > FREEZE_ENCODERS_FIRST_EPOCHS,
            "train_loss": train_out["loss"],
            "train_main_loss": train_out["main_loss"],
            "train_modality_loss": train_out["modality_loss"],
            "train_view_loss": train_out["view_loss"],
            "train_representation_loss": train_out["representation_loss"],
            "val_loss": val_out["loss"],
            "val_main_loss": val_out["main_loss"],
            "val_modality_loss": val_out["modality_loss"],
            "val_view_loss": val_out["view_loss"],
            "val_representation_loss": val_out["representation_loss"],
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

            torch.save(best_state, MODELS_OUT / "MultiTask_CPMRNet_best_val_auc.pt")
        else:
            patience_counter += 1

        torch.save(model.state_dict(), MODELS_OUT / "MultiTask_CPMRNet_last.pt")

        print(
            f"Epoch {epoch:03d} | "
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
    history_df.to_csv(TABLES_OUT / "multitask_training_history.csv", index=False)

    if best_val_predictions is not None:
        best_val_predictions.to_csv(PRED_OUT / "multitask_best_validation_predictions.csv", index=False)

    summary = {
        "stage": "Stage10W",
        "title": "Multi-Task Supervised CPMR-Net Training",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "train_participants": int(len(train_ds)),
        "val_participants": int(len(val_ds)),
        "encoder_initialization": "Stage10V contrastive pretrained encoders",
        "freeze_first_epochs": FREEZE_ENCODERS_FIRST_EPOCHS,
        "auxiliary_weights": {
            "modality": AUX_WEIGHT_MODALITY,
            "view": AUX_WEIGHT_VIEW,
            "representation": AUX_WEIGHT_REPRESENTATION
        },
        "num_representation_classes": len(rep_to_idx),
        "epochs_completed": int(len(history_df)),
        "best_epoch": int(best_epoch),
        "best_val_roc_auc": float(best_auc),
        "best_model_file": str(MODELS_OUT / "MultiTask_CPMRNet_best_val_auc.pt"),
        "last_model_file": str(MODELS_OUT / "MultiTask_CPMRNet_last.pt"),
        "outputs_saved_to": str(STAGE_OUT)
    }

    with open(STAGE_OUT / "Stage10W_MultiTask_Supervised_CPMRNet_Summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    report = []
    report.append("# Stage 10W Multi-Task Supervised CPMR-Net Training\n")
    report.append(f"Generated at: {summary['created_at']}\n")
    report.append("## Purpose\n")
    report.append(
        "This stage trains CPMR-Net with anemia classification as the main task and modality, view, "
        "and representation prediction as auxiliary tasks. Encoders are initialized from Stage 10V "
        "contrastive pretraining.\n"
    )
    report.append("## Summary\n")
    report.append(f"- Train participants: {summary['train_participants']}")
    report.append(f"- Validation participants: {summary['val_participants']}")
    report.append(f"- Freeze-first epochs: {summary['freeze_first_epochs']}")
    report.append(f"- Epochs completed: {summary['epochs_completed']}")
    report.append(f"- Best epoch: {summary['best_epoch']}")
    report.append(f"- Best validation ROC-AUC: {summary['best_val_roc_auc']:.4f}\n")
    report.append("## Output Files\n")
    report.append("- `tables/multitask_training_history.csv`")
    report.append("- `predictions/multitask_best_validation_predictions.csv`")
    report.append("- `models/MultiTask_CPMRNet_best_val_auc.pt`")
    report.append("- `models/MultiTask_CPMRNet_last.pt`")
    report.append("- `Stage10W_MultiTask_Supervised_CPMRNet_Summary.json`")

    with open(REPORTS_OUT / "Stage10W_MultiTask_Supervised_CPMRNet_Report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("=" * 80)
    print("STAGE 10W MULTI-TASK SUPERVISED CPMR-NET TRAINING COMPLETED")
    print("=" * 80)
    print(f"Train/Val participants: {len(train_ds)}/{len(val_ds)}")
    print(f"Epochs completed: {summary['epochs_completed']}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation ROC-AUC: {best_auc:.4f}")
    print(f"Outputs saved to: {STAGE_OUT}")
    print("=" * 80)


if __name__ == "__main__":
    main()