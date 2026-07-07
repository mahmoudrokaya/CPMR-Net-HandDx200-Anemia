# -*- coding: utf-8 -*-
"""
Stage 10T - Pretrained Encoder Baselines

Purpose:
Evaluate whether pretrained visual encoders improve CPMR-Net generalization.

Baseline variants:
1. ResNet-18 pretrained, frozen backbone
2. MobileNetV3-Small pretrained, frozen backbone

No repeated CV here. This is a holdout validation diagnostic stage.
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

from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, confusion_matrix, brier_score_loss

warnings.filterwarnings("ignore")

try:
    import torchvision.models as models
except Exception as e:
    raise ImportError("torchvision is required. Install it using: pip install torchvision") from e


BASE_DIR = Path(r"D:\47\472\New-Papers\Anemia_Paper")
OUTPUTS_DIR = BASE_DIR / "Outputs"

CONFIG_FILE = OUTPUTS_DIR / "Stage10I4_Training_Configuration_Experiment_Control" / "configs" / "CPMRNet_training_config_v1.json"
LOSS_CONFIG_FILE = OUTPUTS_DIR / "Stage10K_Loss_Functions_Class_Imbalance" / "configs" / "CPMRNet_loss_config_v1.json"

STAGE_OUT = OUTPUTS_DIR / "Stage10T_Pretrained_Encoder_Baselines"
TABLES_OUT = STAGE_OUT / "tables"
REPORTS_OUT = STAGE_OUT / "reports"
MODELS_OUT = STAGE_OUT / "models"
PRED_OUT = STAGE_OUT / "predictions"

for p in [TABLES_OUT, REPORTS_OUT, MODELS_OUT, PRED_OUT]:
    p.mkdir(parents=True, exist_ok=True)


BASELINES = [
    {"baseline_id": "T1", "name": "resnet18_pretrained_frozen", "backbone": "resnet18"},
    {"baseline_id": "T2", "name": "mobilenetv3small_pretrained_frozen", "backbone": "mobilenet_v3_small"},
]


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


def resize_to_224(img):
    return cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)


def read_image_tensor(path, modality, representation):
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
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img = resize_to_224(img)
    img = img.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std

    img = np.transpose(img, (2, 0, 1))
    return torch.tensor(img, dtype=torch.float32)


class ParticipantDataset(Dataset):
    def __init__(self, participant_manifest, representation_manifest, split_df, split_name, config):
        self.participant_manifest = participant_manifest.copy()
        self.representation_manifest = representation_manifest.copy()
        self.split_df = split_df[split_df["split"] == split_name].copy()
        self.config = config

        self.views = config["data"]["views"]
        self.rgb_reps = config["data"]["rgb_representations"]
        self.thermal_reps = config["data"]["thermal_representations"]
        self.modalities = config["data"]["modalities"]

        self.participant_ids = self.split_df["participant_id"].astype(str).tolist()
        self.participant_manifest["participant_id"] = self.participant_manifest["participant_id"].astype(str)
        self.representation_manifest["participant_id"] = self.representation_manifest["participant_id"].astype(str)

        self.participant_lookup = {
            str(r["participant_id"]): r for _, r in self.participant_manifest.iterrows()
        }

        self.rep_groups = {
            pid: g.copy() for pid, g in self.representation_manifest.groupby("participant_id")
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
            "representations": {"rgb": {}, "thermal": {}},
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
                    row = view_df[view_df["representation"] == rep]
                    if len(row) == 0:
                        raise RuntimeError(f"Missing representation: {pid} {modality} {view} {rep}")

                    sample["representations"][modality][view][rep] = read_image_tensor(
                        row.iloc[0]["path"], modality, rep
                    )

        return sample


def collate_fn(batch):
    first = batch[0]
    output = {
        "participant_id": [b["participant_id"] for b in batch],
        "label": torch.stack([b["label"] for b in batch]),
        "representations": {"rgb": {}, "thermal": {}},
    }

    for modality in ["rgb", "thermal"]:
        for view in first["representations"][modality]:
            output["representations"][modality][view] = {}
            for rep in first["representations"][modality][view]:
                output["representations"][modality][view][rep] = torch.stack(
                    [b["representations"][modality][view][rep] for b in batch], dim=0
                )

    return output


def move_batch_to_device(batch, device):
    batch["label"] = batch["label"].to(device)
    for modality in ["rgb", "thermal"]:
        for view in batch["representations"][modality]:
            for rep in batch["representations"][modality][view]:
                batch["representations"][modality][view][rep] = batch["representations"][modality][view][rep].to(device)
    return batch


class PretrainedImageEncoder(nn.Module):
    def __init__(self, backbone_name, embedding_dim=128, dropout=0.25):
        super().__init__()

        self.backbone_name = backbone_name

        if backbone_name == "resnet18":
            try:
                weights = models.ResNet18_Weights.DEFAULT
                model = models.resnet18(weights=weights)
            except Exception:
                model = models.resnet18(pretrained=True)

            feature_dim = model.fc.in_features
            model.fc = nn.Identity()
            self.backbone = model

        elif backbone_name == "mobilenet_v3_small":
            try:
                weights = models.MobileNet_V3_Small_Weights.DEFAULT
                model = models.mobilenet_v3_small(weights=weights)
            except Exception:
                model = models.mobilenet_v3_small(pretrained=True)

            feature_dim = model.classifier[0].in_features
            model.classifier = nn.Identity()
            self.backbone = model

        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
        return self.projection(features)


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
        return torch.sum(x * weights.unsqueeze(-1), dim=1), weights


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


class PretrainedCPMRNet(nn.Module):
    def __init__(self, config, backbone_name):
        super().__init__()

        self.embedding_dim = int(config["model"]["embedding_dim"])
        dropout = float(config["model"]["dropout"])

        self.views = config["data"]["views"]
        self.rgb_reps = config["data"]["rgb_representations"]
        self.thermal_reps = config["data"]["thermal_representations"]

        self.rgb_encoder = PretrainedImageEncoder(backbone_name, self.embedding_dim, dropout)
        self.thermal_encoder = PretrainedImageEncoder(backbone_name, self.embedding_dim, dropout)

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

    def encode_specialist(self, batch, modality, view):
        reps = self.rgb_reps if modality == "rgb" else self.thermal_reps
        encoder = self.rgb_encoder if modality == "rgb" else self.thermal_encoder

        embeddings = []
        for rep in reps:
            x = batch["representations"][modality][view][rep]
            embeddings.append(encoder(x))

        stack = torch.stack(embeddings, dim=1)
        return self.representation_attention(stack)

    def forward(self, batch):
        rgb_specs = []
        th_specs = []

        for view in self.views:
            rgb_spec, _ = self.encode_specialist(batch, "rgb", view)
            th_spec, _ = self.encode_specialist(batch, "thermal", view)
            rgb_specs.append(rgb_spec)
            th_specs.append(th_spec)

        rgb_branch, rgb_view_w = self.rgb_view_attention(torch.stack(rgb_specs, dim=1))
        thermal_branch, thermal_view_w = self.thermal_view_attention(torch.stack(th_specs, dim=1))

        fused, modality_w = self.fusion(rgb_branch, thermal_branch)

        logits = self.classifier(fused).squeeze(-1)
        probs = torch.sigmoid(logits)

        return {
            "logits": logits,
            "probabilities": probs,
            "modality_weights": modality_w,
            "rgb_view_weights": rgb_view_w,
            "thermal_view_weights": thermal_view_w,
        }


def safe_auc(y_true, y_prob):
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return np.nan


def safe_pr_auc(y_true, y_prob):
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return float(average_precision_score(y_true, y_prob))
    except Exception:
        return np.nan


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": safe_auc(y_true, y_prob),
        "pr_auc": safe_pr_auc(y_true, y_prob),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()

    total_loss = 0.0
    labels_all = []
    probs_all = []
    pids_all = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            out = model(batch)
            loss = criterion(out["logits"], batch["label"])

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()

        total_loss += float(loss.item()) * batch["label"].shape[0]
        labels_all.extend(batch["label"].detach().cpu().numpy().tolist())
        probs_all.extend(out["probabilities"].detach().cpu().numpy().tolist())
        pids_all.extend(batch["participant_id"])

    return total_loss / len(loader.dataset), compute_metrics(labels_all, probs_all), pids_all, labels_all, probs_all


def train_baseline(config, loss_config, baseline, train_loader, val_loader, device):
    model = PretrainedCPMRNet(config, baseline["backbone"]).to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [float(loss_config["positive_class_weight_for_bce"])],
            dtype=torch.float32,
            device=device
        )
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(config["optimization"]["learning_rate"]),
        weight_decay=float(config["optimization"]["weight_decay"]),
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(config["optimization"]["scheduler_factor"]),
        patience=int(config["optimization"]["scheduler_patience"]),
    )

    max_epochs = int(config["optimization"]["max_epochs"])
    patience = int(config["optimization"]["early_stopping_patience"])

    best_val_auc = -np.inf
    best_epoch = 0
    patience_counter = 0
    best_state = None
    best_val_predictions = None
    history = []

    for epoch in range(1, max_epochs + 1):
        train_loss, train_metrics, _, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_metrics, val_pids, val_labels, val_probs = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        val_auc = val_metrics["roc_auc"]
        scheduler.step(0.0 if np.isnan(val_auc) else val_auc)

        row = {
            "baseline_id": baseline["baseline_id"],
            "baseline_name": baseline["name"],
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "val_loss": val_loss,
        }

        for k, v in train_metrics.items():
            row[f"train_{k}"] = v
        for k, v in val_metrics.items():
            row[f"val_{k}"] = v

        history.append(row)

        improved = not np.isnan(val_auc) and val_auc > best_val_auc

        if improved:
            best_val_auc = val_auc
            best_epoch = epoch
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
            best_val_predictions = pd.DataFrame({
                "participant_id": val_pids,
                "label": val_labels,
                "probability": val_probs,
                "prediction_0_5": (np.asarray(val_probs) >= 0.5).astype(int),
            })

            torch.save(best_state, MODELS_OUT / f"{baseline['name']}_best_val_auc.pt")
        else:
            patience_counter += 1

        torch.save(model.state_dict(), MODELS_OUT / f"{baseline['name']}_last.pt")

        print(
            f"{baseline['name']} | Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_auc={val_auc:.4f} | val_f1={val_metrics['f1']:.4f} | "
            f"patience={patience_counter}/{patience}"
        )

        if patience_counter >= patience:
            print(f"Early stopping for {baseline['name']} at epoch {epoch}.")
            break

    history_df = pd.DataFrame(history)
    pred_path = PRED_OUT / f"{baseline['name']}_best_validation_predictions.csv"
    if best_val_predictions is not None:
        best_val_predictions.to_csv(pred_path, index=False)

    result = {
        "baseline_id": baseline["baseline_id"],
        "baseline_name": baseline["name"],
        "backbone": baseline["backbone"],
        "total_parameters": int(count_parameters(model)),
        "trainable_parameters": int(count_parameters(model, trainable_only=True)),
        "epochs_completed": int(len(history_df)),
        "best_epoch": int(best_epoch),
        "best_val_roc_auc": float(best_val_auc),
        "best_model_file": str(MODELS_OUT / f"{baseline['name']}_best_val_auc.pt"),
        "prediction_file": str(pred_path),
    }

    return result, history_df


def main():
    config = load_json(CONFIG_FILE)
    loss_config = load_json(LOSS_CONFIG_FILE)

    set_seed(int(config["optimization"]["random_seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    participant_manifest = pd.read_csv(config["paths"]["participant_manifest"])
    representation_manifest = pd.read_csv(config["paths"]["representation_manifest"])
    holdout_split = pd.read_csv(config["paths"]["holdout_split"])

    train_ds = ParticipantDataset(participant_manifest, representation_manifest, holdout_split, "train", config)
    val_ds = ParticipantDataset(participant_manifest, representation_manifest, holdout_split, "val", config)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(config["data"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["data"]["num_workers"]),
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=int(config["data"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"]["num_workers"]),
        collate_fn=collate_fn,
    )

    all_results = []
    all_history = []

    for baseline in BASELINES:
        print("=" * 80)
        print(f"Running pretrained baseline: {baseline['name']}")
        print("=" * 80)

        result, history_df = train_baseline(
            config,
            loss_config,
            baseline,
            train_loader,
            val_loader,
            device
        )

        all_results.append(result)
        all_history.append(history_df)

    results_df = pd.DataFrame(all_results)
    history_all_df = pd.concat(all_history, ignore_index=True)

    results_df.to_csv(TABLES_OUT / "pretrained_encoder_baseline_results.csv", index=False)
    history_all_df.to_csv(TABLES_OUT / "pretrained_encoder_training_history.csv", index=False)

    best_row = results_df.sort_values("best_val_roc_auc", ascending=False).iloc[0].to_dict()

    summary = {
        "stage": "Stage10T",
        "title": "Pretrained Encoder Baselines",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "baselines_completed": int(len(results_df)),
        "best_baseline": best_row["baseline_name"],
        "best_backbone": best_row["backbone"],
        "best_val_roc_auc": float(best_row["best_val_roc_auc"]),
        "results": results_df.to_dict(orient="records"),
        "outputs_saved_to": str(STAGE_OUT),
    }

    with open(STAGE_OUT / "Stage10T_Pretrained_Encoder_Baselines_Summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    report = []
    report.append("# Stage 10T Pretrained Encoder Baselines\n")
    report.append(f"Generated at: {summary['created_at']}\n")
    report.append("## Purpose\n")
    report.append(
        "This stage evaluates whether pretrained visual encoders improve CPMR-Net validation behavior compared with from-scratch lightweight encoders.\n"
    )
    report.append("## Results\n")
    for r in summary["results"]:
        report.append(f"- {r['baseline_name']}: best validation ROC-AUC = {r['best_val_roc_auc']:.4f}, trainable parameters = {r['trainable_parameters']}")
    report.append(f"\nBest baseline: **{summary['best_baseline']}** with validation ROC-AUC **{summary['best_val_roc_auc']:.4f}**.\n")
    report.append("## Output Files\n")
    report.append("- `tables/pretrained_encoder_baseline_results.csv`")
    report.append("- `tables/pretrained_encoder_training_history.csv`")
    report.append("- `predictions/*_best_validation_predictions.csv`")
    report.append("- `models/*_best_val_auc.pt`")
    report.append("- `Stage10T_Pretrained_Encoder_Baselines_Summary.json`")

    with open(REPORTS_OUT / "Stage10T_Pretrained_Encoder_Baselines_Report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("=" * 80)
    print("STAGE 10T PRETRAINED ENCODER BASELINES COMPLETED")
    print("=" * 80)
    print(f"Baselines completed: {summary['baselines_completed']}")
    print(f"Best baseline: {summary['best_baseline']}")
    print(f"Best validation ROC-AUC: {summary['best_val_roc_auc']:.4f}")
    print(f"Outputs saved to: {STAGE_OUT}")
    print("=" * 80)


if __name__ == "__main__":
    main()