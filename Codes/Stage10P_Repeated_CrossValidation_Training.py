# -*- coding: utf-8 -*-
"""
Stage 10P - Repeated Cross-Validation Training

Purpose:
Run repeated participant-level cross-validation training for CPMR-Net to determine
whether the weak independent test result reflects split sensitivity or systematic
generalization limitation.

Default:
- Uses Stage 10I1 repeated_stratified_5fold_train_val_test_splits.csv
- Runs 5 repeats × 5 folds
- Trains CPMR-Net per fold
- Selects best checkpoint by validation ROC-AUC
- Evaluates fold test set using validation-derived Youden threshold
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

STAGE_OUT = OUTPUTS_DIR / "Stage10P_Repeated_CrossValidation_Training"
TABLES_OUT = STAGE_OUT / "tables"
REPORTS_OUT = STAGE_OUT / "reports"
MODELS_OUT = STAGE_OUT / "models"
PRED_OUT = STAGE_OUT / "predictions"

for p in [TABLES_OUT, REPORTS_OUT, MODELS_OUT, PRED_OUT]:
    p.mkdir(parents=True, exist_ok=True)


# Runtime control
RUN_REPEATS = [1, 2, 3, 4, 5]
RUN_FOLDS = [1, 2, 3, 4, 5]
MAX_EPOCHS_OVERRIDE = None  # set e.g. 40 for faster diagnostic run
PATIENCE_OVERRIDE = None    # set e.g. 8 for faster diagnostic run


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_image_as_tensor(path, modality, representation):
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


class CPMRParticipantDataset(Dataset):
    def __init__(self, participant_manifest, representation_manifest, split_df, split_name, config):
        self.participant_manifest = participant_manifest.copy()
        self.representation_manifest = representation_manifest.copy()
        self.split_df = split_df[split_df["split"] == split_name].copy()
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
                    sample["representations"][modality][view][rep] = read_image_as_tensor(
                        row.iloc[0]["path"], modality, rep
                    )

        return sample


def cpmr_collate_fn(batch):
    first = batch[0]
    output = {
        "participant_id": [b["participant_id"] for b in batch],
        "label": torch.stack([b["label"] for b in batch]),
        "representations": {"rgb": {}, "thermal": {}},
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


def move_batch_to_device(batch, device):
    batch["label"] = batch["label"].to(device)
    for modality in ["rgb", "thermal"]:
        for view in batch["representations"][modality]:
            for rep in batch["representations"][modality][view]:
                batch["representations"][modality][view][rep] = batch["representations"][modality][view][rep].to(device)
    return batch


class LightweightEncoder(nn.Module):
    def __init__(self, in_channels, embedding_dim=128, dropout=0.25):
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
        return torch.sum(x * weights.unsqueeze(-1), dim=1), weights


class AdaptiveRGBThermalFusion(nn.Module):
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


class CPMRNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding_dim = int(config["model"]["embedding_dim"])
        dropout = float(config["model"]["dropout"])

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
            nn.Linear(self.embedding_dim, 1),
        )

    def encode_representation(self, modality, representation, tensor):
        if modality == "rgb":
            return self.rgb_1ch_encoder(tensor) if representation == "rgb_texture" else self.rgb_3ch_encoder(tensor)
        return self.thermal_encoder(tensor)

    def encode_specialist(self, batch, modality, view):
        reps = self.rgb_reps if modality == "rgb" else self.thermal_reps
        embeddings = [self.encode_representation(modality, rep, batch["representations"][modality][view][rep]) for rep in reps]
        return self.representation_attention(torch.stack(embeddings, dim=1))

    def forward(self, batch):
        rgb_specialists, thermal_specialists = [], []

        for view in self.views:
            rgb_spec, _ = self.encode_specialist(batch, "rgb", view)
            th_spec, _ = self.encode_specialist(batch, "thermal", view)
            rgb_specialists.append(rgb_spec)
            thermal_specialists.append(th_spec)

        rgb_branch, rgb_view_weights = self.rgb_view_attention(torch.stack(rgb_specialists, dim=1))
        thermal_branch, thermal_view_weights = self.thermal_view_attention(torch.stack(thermal_specialists, dim=1))
        fused, modality_weights = self.fusion(rgb_branch, thermal_branch)

        logits = self.classifier(fused).squeeze(-1)
        return {
            "logits": logits,
            "probabilities": torch.sigmoid(logits),
            "modality_weights": modality_weights,
            "rgb_view_weights": rgb_view_weights,
            "thermal_view_weights": thermal_view_weights,
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


def metrics_at_threshold(y_true, y_prob, threshold):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc": safe_auc(y_true, y_prob),
        "pr_auc": safe_pr_auc(y_true, y_prob),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
    }


def select_youden_threshold(y_true, y_prob):
    rows = []
    for th in np.round(np.arange(0.01, 1.00, 0.01), 2):
        row = metrics_at_threshold(y_true, y_prob, th)
        row["youden_j"] = row["recall"] + row["specificity"] - 1
        rows.append(row)
    df = pd.DataFrame(rows)
    best = df.sort_values(["youden_j", "balanced_accuracy", "f1"], ascending=False).iloc[0]
    return float(best["threshold"]), df


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train() if train else model.eval()
    total_loss = 0.0
    labels_all, probs_all, pids_all = [], [], []

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            out = model(batch)
            loss = criterion(out["logits"], batch["label"])
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        total_loss += float(loss.item()) * batch["label"].shape[0]
        labels_all.extend(batch["label"].detach().cpu().numpy().tolist())
        probs_all.extend(out["probabilities"].detach().cpu().numpy().tolist())
        pids_all.extend(batch["participant_id"])

    return total_loss / len(loader.dataset), labels_all, probs_all, pids_all


def train_one_fold(config, loss_config, participant_manifest, representation_manifest, split_df, repeat, fold, device):
    seed = int(config["optimization"]["random_seed"]) + repeat * 100 + fold
    set_seed(seed)

    batch_size = int(config["data"]["batch_size"])
    num_workers = int(config["data"]["num_workers"])

    train_ds = CPMRParticipantDataset(participant_manifest, representation_manifest, split_df, "train", config)
    val_ds = CPMRParticipantDataset(participant_manifest, representation_manifest, split_df, "val", config)
    test_ds = CPMRParticipantDataset(participant_manifest, representation_manifest, split_df, "test", config)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=cpmr_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=cpmr_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=cpmr_collate_fn)

    model = CPMRNet(config).to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([float(loss_config["positive_class_weight_for_bce"])], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["optimization"]["learning_rate"]),
        weight_decay=float(config["optimization"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(config["optimization"]["scheduler_factor"]),
        patience=int(config["optimization"]["scheduler_patience"]),
    )

    max_epochs = int(config["optimization"]["max_epochs"]) if MAX_EPOCHS_OVERRIDE is None else MAX_EPOCHS_OVERRIDE
    patience = int(config["optimization"]["early_stopping_patience"]) if PATIENCE_OVERRIDE is None else PATIENCE_OVERRIDE

    best_auc, best_epoch, patience_counter = -np.inf, 0, 0
    best_state = None
    history = []

    for epoch in range(1, max_epochs + 1):
        train_loss, train_y, train_p, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_y, val_p, _ = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        train_metrics = metrics_at_threshold(train_y, train_p, 0.5)
        val_metrics = metrics_at_threshold(val_y, val_p, 0.5)
        val_auc = val_metrics["roc_auc"]

        scheduler.step(0.0 if np.isnan(val_auc) else val_auc)

        history.append({
            "repeat": repeat, "fold": fold, "epoch": epoch,
            "train_loss": train_loss, "val_loss": val_loss,
            "train_roc_auc": train_metrics["roc_auc"], "val_roc_auc": val_auc,
            "train_f1": train_metrics["f1"], "val_f1": val_metrics["f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        })

        if not np.isnan(val_auc) and val_auc > best_auc:
            best_auc, best_epoch = val_auc, epoch
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    model.load_state_dict(best_state)

    _, val_y, val_p, val_ids = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
    _, test_y, test_p, test_ids = run_epoch(model, test_loader, criterion, optimizer, device, train=False)

    val_threshold, threshold_sweep = select_youden_threshold(val_y, val_p)
    val_final = metrics_at_threshold(val_y, val_p, val_threshold)
    test_final = metrics_at_threshold(test_y, test_p, val_threshold)
    test_default = metrics_at_threshold(test_y, test_p, 0.5)

    fold_tag = f"R{repeat}_F{fold}"
    torch.save(best_state, MODELS_OUT / f"CPMRNet_best_{fold_tag}.pt")

    pd.DataFrame({
        "participant_id": val_ids, "label": val_y, "probability": val_p,
        "prediction_youden": (np.asarray(val_p) >= val_threshold).astype(int)
    }).to_csv(PRED_OUT / f"validation_predictions_{fold_tag}.csv", index=False)

    pd.DataFrame({
        "participant_id": test_ids, "label": test_y, "probability": test_p,
        "prediction_youden_from_validation": (np.asarray(test_p) >= val_threshold).astype(int),
        "prediction_default_0_5": (np.asarray(test_p) >= 0.5).astype(int)
    }).to_csv(PRED_OUT / f"test_predictions_{fold_tag}.csv", index=False)

    threshold_sweep.insert(0, "fold", fold)
    threshold_sweep.insert(0, "repeat", repeat)

    fold_summary = {
        "repeat": repeat,
        "fold": fold,
        "train_n": len(train_ds),
        "val_n": len(val_ds),
        "test_n": len(test_ds),
        "best_epoch": best_epoch,
        "best_val_auc": best_auc,
        "validation_youden_threshold": val_threshold,
        **{f"val_{k}": v for k, v in val_final.items()},
        **{f"test_{k}": v for k, v in test_final.items()},
        **{f"test_default_{k}": v for k, v in test_default.items()},
    }

    return fold_summary, pd.DataFrame(history), threshold_sweep


def main():
    config = load_json(CONFIG_FILE)
    loss_config = load_json(LOSS_CONFIG_FILE)

    participant_manifest = pd.read_csv(config["paths"]["participant_manifest"])
    representation_manifest = pd.read_csv(config["paths"]["representation_manifest"])
    repeated_splits = pd.read_csv(config["paths"]["repeated_splits"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_fold_summaries = []
    all_histories = []
    all_threshold_sweeps = []

    for repeat in RUN_REPEATS:
        for fold in RUN_FOLDS:
            print("=" * 80)
            print(f"Training repeat {repeat}, fold {fold}")
            print("=" * 80)

            split_df = repeated_splits[
                (repeated_splits["repeat"] == repeat)
                & (repeated_splits["fold"] == fold)
            ].copy()

            fold_summary, history_df, threshold_df = train_one_fold(
                config, loss_config, participant_manifest, representation_manifest,
                split_df, repeat, fold, device
            )

            all_fold_summaries.append(fold_summary)
            all_histories.append(history_df)
            all_threshold_sweeps.append(threshold_df)

            print(
                f"R{repeat} F{fold} | best_val_auc={fold_summary['best_val_auc']:.4f} | "
                f"thr={fold_summary['validation_youden_threshold']:.2f} | "
                f"test_auc={fold_summary['test_roc_auc']:.4f} | "
                f"test_f1={fold_summary['test_f1']:.4f}"
            )

    fold_summary_df = pd.DataFrame(all_fold_summaries)
    history_all_df = pd.concat(all_histories, ignore_index=True)
    threshold_all_df = pd.concat(all_threshold_sweeps, ignore_index=True)

    fold_summary_df.to_csv(TABLES_OUT / "repeated_cv_fold_results.csv", index=False)
    history_all_df.to_csv(TABLES_OUT / "repeated_cv_training_history.csv", index=False)
    threshold_all_df.to_csv(TABLES_OUT / "repeated_cv_validation_threshold_sweeps.csv", index=False)

    metric_cols = [
        "test_accuracy", "test_balanced_accuracy", "test_precision", "test_recall",
        "test_specificity", "test_f1", "test_roc_auc", "test_pr_auc", "test_mcc",
        "val_roc_auc", "val_f1"
    ]

    aggregate_rows = []
    for metric in metric_cols:
        if metric in fold_summary_df.columns:
            vals = fold_summary_df[metric].dropna()
            aggregate_rows.append({
                "metric": metric,
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "min": float(vals.min()),
                "median": float(vals.median()),
                "max": float(vals.max()),
                "n": int(len(vals))
            })

    aggregate_df = pd.DataFrame(aggregate_rows)
    aggregate_df.to_csv(TABLES_OUT / "repeated_cv_aggregate_metrics.csv", index=False)

    summary = {
        "stage": "Stage10P",
        "title": "Repeated Cross-Validation Training",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "repeats_run": RUN_REPEATS,
        "folds_run": RUN_FOLDS,
        "fold_configurations_completed": int(len(fold_summary_df)),
        "mean_test_roc_auc": float(fold_summary_df["test_roc_auc"].mean()),
        "std_test_roc_auc": float(fold_summary_df["test_roc_auc"].std()),
        "mean_test_pr_auc": float(fold_summary_df["test_pr_auc"].mean()),
        "mean_test_balanced_accuracy": float(fold_summary_df["test_balanced_accuracy"].mean()),
        "mean_test_f1": float(fold_summary_df["test_f1"].mean()),
        "mean_test_recall": float(fold_summary_df["test_recall"].mean()),
        "mean_test_specificity": float(fold_summary_df["test_specificity"].mean()),
        "outputs_saved_to": str(STAGE_OUT),
    }

    with open(STAGE_OUT / "Stage10P_Repeated_CrossValidation_Training_Summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    report = []
    report.append("# Stage 10P Repeated Cross-Validation Training\n")
    report.append(f"Generated at: {summary['created_at']}\n")
    report.append("## Purpose\n")
    report.append(
        "This stage trains CPMR-Net under repeated participant-level cross-validation to diagnose whether "
        "the weak holdout test result reflects split sensitivity or a systematic limitation.\n"
    )
    report.append("## Summary\n")
    report.append(f"- Device: {summary['device']}")
    report.append(f"- Fold configurations completed: {summary['fold_configurations_completed']}")
    report.append(f"- Mean test ROC-AUC: {summary['mean_test_roc_auc']:.4f} ± {summary['std_test_roc_auc']:.4f}")
    report.append(f"- Mean test PR-AUC: {summary['mean_test_pr_auc']:.4f}")
    report.append(f"- Mean test balanced accuracy: {summary['mean_test_balanced_accuracy']:.4f}")
    report.append(f"- Mean test F1: {summary['mean_test_f1']:.4f}")
    report.append(f"- Mean test recall: {summary['mean_test_recall']:.4f}")
    report.append(f"- Mean test specificity: {summary['mean_test_specificity']:.4f}\n")
    report.append("## Output Files\n")
    report.append("- `tables/repeated_cv_fold_results.csv`")
    report.append("- `tables/repeated_cv_training_history.csv`")
    report.append("- `tables/repeated_cv_validation_threshold_sweeps.csv`")
    report.append("- `tables/repeated_cv_aggregate_metrics.csv`")
    report.append("- `predictions/validation_predictions_R*_F*.csv`")
    report.append("- `predictions/test_predictions_R*_F*.csv`")
    report.append("- `models/CPMRNet_best_R*_F*.pt`")
    report.append("- `Stage10P_Repeated_CrossValidation_Training_Summary.json`")

    with open(REPORTS_OUT / "Stage10P_Repeated_CrossValidation_Training_Report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("=" * 80)
    print("STAGE 10P REPEATED CROSS-VALIDATION TRAINING COMPLETED")
    print("=" * 80)
    print(f"Fold configurations completed: {summary['fold_configurations_completed']}")
    print(f"Mean test ROC-AUC: {summary['mean_test_roc_auc']:.4f} ± {summary['std_test_roc_auc']:.4f}")
    print(f"Mean test PR-AUC: {summary['mean_test_pr_auc']:.4f}")
    print(f"Mean test balanced accuracy: {summary['mean_test_balanced_accuracy']:.4f}")
    print(f"Mean test F1: {summary['mean_test_f1']:.4f}")
    print(f"Outputs saved to: {STAGE_OUT}")
    print("=" * 80)


if __name__ == "__main__":
    main()