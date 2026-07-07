# -*- coding: utf-8 -*-
"""
Stage 10N - Independent Holdout Test Evaluation

Purpose:
Evaluate the best CPMR-Net checkpoint from Stage 10L on the independent holdout test set.

Important:
- The test set is not used for model selection.
- The threshold is loaded from Stage 10M validation analysis.
- No threshold re-optimization is performed on the test set.
"""

from pathlib import Path
import json
from datetime import datetime
import warnings

import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix,
    brier_score_loss,
)

warnings.filterwarnings("ignore")


BASE_DIR = Path(r"D:\47\472\New-Papers\Anemia_Paper")
OUTPUTS_DIR = BASE_DIR / "Outputs"

CONFIG_FILE = OUTPUTS_DIR / "Stage10I4_Training_Configuration_Experiment_Control" / "configs" / "CPMRNet_training_config_v1.json"
THRESHOLD_FILE = OUTPUTS_DIR / "Stage10M_Validation_Engine_Threshold_Analysis" / "configs" / "CPMRNet_validation_threshold_config_v1.json"
BEST_MODEL_FILE = OUTPUTS_DIR / "Stage10L_CPMRNet_Training_Engine" / "models" / "CPMRNet_best_val_auc.pt"

STAGE_OUT = OUTPUTS_DIR / "Stage10N_Independent_Holdout_Test_Evaluation"
TABLES_OUT = STAGE_OUT / "tables"
REPORTS_OUT = STAGE_OUT / "reports"
PRED_OUT = STAGE_OUT / "predictions"

for p in [TABLES_OUT, REPORTS_OUT, PRED_OUT]:
    p.mkdir(parents=True, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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

                    path = row.iloc[0]["path"]
                    sample["representations"][modality][view][rep] = read_image_as_tensor(path, modality, rep)

        return sample


def cpmr_collate_fn(batch):
    first = batch[0]

    output = {
        "participant_id": [b["participant_id"] for b in batch],
        "label": torch.stack([b["label"] for b in batch]),
        "class_name": [b["class_name"] for b in batch],
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
                batch["representations"][modality][view][rep] = (
                    batch["representations"][modality][view][rep].to(device)
                )

    return batch


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
            nn.Linear(embedding_dim // 2, 1),
        )

    def forward(self, x):
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
            nn.Linear(embedding_dim, 2),
        )

    def forward(self, rgb_vec, thermal_vec):
        x = torch.cat([rgb_vec, thermal_vec], dim=-1)
        weights = torch.softmax(self.gate(x), dim=-1)
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

        stack = torch.stack(embeddings, dim=1)
        specialist, weights = self.representation_attention(stack)
        return specialist, weights

    def forward(self, batch):
        rgb_specialists = []
        thermal_specialists = []

        for view in self.views:
            rgb_spec, _ = self.encode_specialist(batch, "rgb", view)
            th_spec, _ = self.encode_specialist(batch, "thermal", view)

            rgb_specialists.append(rgb_spec)
            thermal_specialists.append(th_spec)

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


def compute_metrics(y_true, y_prob, threshold):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "ppv": float(ppv),
        "npv": float(npv),
        "roc_auc": safe_auc(y_true, y_prob),
        "pr_auc": safe_pr_auc(y_true, y_prob),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate_model(model, loader, device):
    model.eval()

    rows = []

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            output = model(batch)

            probs = output["probabilities"].detach().cpu().numpy()
            labels = batch["label"].detach().cpu().numpy()

            modality_weights = output["modality_weights"].detach().cpu().numpy()

            for i, pid in enumerate(batch["participant_id"]):
                rows.append({
                    "participant_id": pid,
                    "label": int(labels[i]),
                    "probability": float(probs[i]),
                    "rgb_modality_weight": float(modality_weights[i, 0]),
                    "thermal_modality_weight": float(modality_weights[i, 1]),
                })

    return pd.DataFrame(rows)


def main():
    for path in [CONFIG_FILE, THRESHOLD_FILE, BEST_MODEL_FILE]:
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing required input: {path}")

    config = load_json(CONFIG_FILE)
    threshold_config = load_json(THRESHOLD_FILE)

    participant_manifest = pd.read_csv(config["paths"]["participant_manifest"])
    representation_manifest = pd.read_csv(config["paths"]["representation_manifest"])
    holdout_split = pd.read_csv(config["paths"]["holdout_split"])

    test_ds = CPMRParticipantDataset(
        participant_manifest,
        representation_manifest,
        holdout_split,
        "test",
        config
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=int(config["data"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"]["num_workers"]),
        collate_fn=cpmr_collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CPMRNet(config).to(device)
    model.load_state_dict(torch.load(BEST_MODEL_FILE, map_location=device))

    pred_df = evaluate_model(model, test_loader, device)

    validation_threshold = float(threshold_config["recommended_threshold"])
    default_threshold = 0.50

    pred_df["prediction_default_0_50"] = (pred_df["probability"] >= default_threshold).astype(int)
    pred_df["prediction_validation_threshold"] = (pred_df["probability"] >= validation_threshold).astype(int)

    pred_df.to_csv(PRED_OUT / "independent_test_predictions.csv", index=False)

    y_true = pred_df["label"].values
    y_prob = pred_df["probability"].values

    default_metrics = compute_metrics(y_true, y_prob, default_threshold)
    default_metrics["threshold_rule"] = "default_0_50"

    validation_threshold_metrics = compute_metrics(y_true, y_prob, validation_threshold)
    validation_threshold_metrics["threshold_rule"] = "validation_derived_youden_threshold"

    metrics_df = pd.DataFrame([default_metrics, validation_threshold_metrics])
    metrics_df.to_csv(TABLES_OUT / "independent_test_metrics.csv", index=False)

    modality_weight_summary = pred_df[["rgb_modality_weight", "thermal_modality_weight"]].agg(
        ["mean", "std", "min", "max"]
    ).reset_index().rename(columns={"index": "statistic"})

    modality_weight_summary.to_csv(TABLES_OUT / "test_modality_weight_summary.csv", index=False)

    summary = {
        "stage": "Stage10N",
        "title": "Independent Holdout Test Evaluation",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "test_participants": int(len(test_ds)),
        "test_positive_anemia": int(np.sum(y_true == 1)),
        "test_negative_normal": int(np.sum(y_true == 0)),
        "best_model_file": str(BEST_MODEL_FILE),
        "validation_threshold_file": str(THRESHOLD_FILE),
        "default_threshold": default_threshold,
        "validation_derived_threshold": validation_threshold,
        "default_threshold_metrics": default_metrics,
        "validation_threshold_metrics": validation_threshold_metrics,
        "outputs_saved_to": str(STAGE_OUT),
        "important_note": (
            "The validation-derived threshold was applied directly to the independent test set. "
            "No test-set threshold optimization was performed."
        )
    }

    with open(STAGE_OUT / "Stage10N_Independent_Holdout_Test_Evaluation_Summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    report = []
    report.append("# Stage 10N Independent Holdout Test Evaluation\n")
    report.append(f"Generated at: {summary['created_at']}\n")

    report.append("## Purpose\n")
    report.append(
        "This stage evaluates the best Stage 10L CPMR-Net checkpoint on the independent holdout test set. "
        "The validation-derived threshold from Stage 10M is applied directly without re-optimizing on the test set.\n"
    )

    report.append("## Test Set\n")
    report.append(f"- Test participants: {summary['test_participants']}")
    report.append(f"- Anemia: {summary['test_positive_anemia']}")
    report.append(f"- Normal: {summary['test_negative_normal']}\n")

    report.append("## Thresholds Evaluated\n")
    report.append(f"- Default threshold: {default_threshold:.2f}")
    report.append(f"- Validation-derived threshold: {validation_threshold:.2f}\n")

    report.append("## Test Metrics at Validation-Derived Threshold\n")
    for k, v in validation_threshold_metrics.items():
        if isinstance(v, float):
            report.append(f"- {k}: {v:.4f}")
        else:
            report.append(f"- {k}: {v}")

    report.append("\n## Test Metrics at Default 0.50 Threshold\n")
    for k, v in default_metrics.items():
        if isinstance(v, float):
            report.append(f"- {k}: {v:.4f}")
        else:
            report.append(f"- {k}: {v}")

    report.append("\n## Output Files\n")
    report.append("- `predictions/independent_test_predictions.csv`")
    report.append("- `tables/independent_test_metrics.csv`")
    report.append("- `tables/test_modality_weight_summary.csv`")
    report.append("- `Stage10N_Independent_Holdout_Test_Evaluation_Summary.json`\n")

    report.append("## Important Note\n")
    report.append(summary["important_note"])

    with open(REPORTS_OUT / "Stage10N_Independent_Holdout_Test_Evaluation_Report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("=" * 80)
    print("STAGE 10N INDEPENDENT HOLDOUT TEST EVALUATION COMPLETED")
    print("=" * 80)
    print(f"Test participants: {summary['test_participants']}")
    print(f"Anemia/Normal: {summary['test_positive_anemia']}/{summary['test_negative_normal']}")
    print(f"Validation-derived threshold: {validation_threshold:.2f}")
    print("Metrics at validation-derived threshold:")
    print(f"  Accuracy: {validation_threshold_metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {validation_threshold_metrics['balanced_accuracy']:.4f}")
    print(f"  Precision: {validation_threshold_metrics['precision']:.4f}")
    print(f"  Recall: {validation_threshold_metrics['recall']:.4f}")
    print(f"  Specificity: {validation_threshold_metrics['specificity']:.4f}")
    print(f"  F1: {validation_threshold_metrics['f1']:.4f}")
    print(f"  ROC-AUC: {validation_threshold_metrics['roc_auc']:.4f}")
    print(f"  PR-AUC: {validation_threshold_metrics['pr_auc']:.4f}")
    print(f"  MCC: {validation_threshold_metrics['mcc']:.4f}")
    print(f"  Confusion: TN={validation_threshold_metrics['tn']}, FP={validation_threshold_metrics['fp']}, FN={validation_threshold_metrics['fn']}, TP={validation_threshold_metrics['tp']}")
    print(f"Outputs saved to: {STAGE_OUT}")
    print("=" * 80)


if __name__ == "__main__":
    main()