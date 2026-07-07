# -*- coding: utf-8 -*-
"""
Stage 10K - Loss Functions and Class-Imbalance Handling

Purpose:
Define and validate loss functions for CPMR-Net supervised training.

This stage:
- Loads Stage 10I4 experiment config.
- Loads Stage 10I1 participant-level holdout split.
- Computes class imbalance statistics from the training split only.
- Defines weighted BCEWithLogitsLoss.
- Defines optional focal loss.
- Validates loss behavior on synthetic logits.
- Saves loss configuration for training.

No model training is performed.
"""

from pathlib import Path
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

BASE_DIR = Path(r"D:\47\472\New-Papers\Anemia_Paper")
OUTPUTS_DIR = BASE_DIR / "Outputs"

STAGE10I4_DIR = OUTPUTS_DIR / "Stage10I4_Training_Configuration_Experiment_Control"
CONFIG_FILE = STAGE10I4_DIR / "configs" / "CPMRNet_training_config_v1.json"

STAGE_OUT = OUTPUTS_DIR / "Stage10K_Loss_Functions_Class_Imbalance"
TABLES_OUT = STAGE_OUT / "tables"
REPORTS_OUT = STAGE_OUT / "reports"
CONFIG_OUT = STAGE_OUT / "configs"

for p in [TABLES_OUT, REPORTS_OUT, CONFIG_OUT]:
    p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------

class BinaryFocalLossWithLogits(nn.Module):
    """
    Binary focal loss using logits.

    This is optional and not the default loss.
    """

    def __init__(self, alpha=1.0, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        targets = targets.float()

        bce = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none"
        )

        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)

        focal_factor = (1 - pt) ** self.gamma
        loss = self.alpha * focal_factor * bce

        if self.reduction == "mean":
            return loss.mean()

        if self.reduction == "sum":
            return loss.sum()

        return loss


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_binary_metrics_for_logits(logits, labels):
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).long()

    labels_long = labels.long()

    tp = int(((preds == 1) & (labels_long == 1)).sum().item())
    tn = int(((preds == 0) & (labels_long == 0)).sum().item())
    fp = int(((preds == 1) & (labels_long == 0)).sum().item())
    fn = int(((preds == 0) & (labels_long == 1)).sum().item())

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "mean_probability": float(probs.mean().item()),
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Missing Stage 10I4 config: {CONFIG_FILE}")

    config = load_json(CONFIG_FILE)

    holdout_split_file = Path(config["paths"]["holdout_split"])

    if not holdout_split_file.exists():
        raise FileNotFoundError(f"Missing holdout split file: {holdout_split_file}")

    split_df = pd.read_csv(holdout_split_file)

    train_df = split_df[split_df["split"] == "train"].copy()
    val_df = split_df[split_df["split"] == "val"].copy()
    test_df = split_df[split_df["split"] == "test"].copy()

    if train_df.empty:
        raise ValueError("Training split is empty.")

    train_counts = train_df["label"].value_counts().to_dict()
    val_counts = val_df["label"].value_counts().to_dict()
    test_counts = test_df["label"].value_counts().to_dict()

    train_neg = int(train_counts.get(0, 0))
    train_pos = int(train_counts.get(1, 0))

    if train_pos == 0 or train_neg == 0:
        raise ValueError("Training split must contain both classes.")

    pos_weight = train_neg / train_pos

    # Balanced class weights for record keeping
    total_train = train_neg + train_pos
    weight_for_negative = total_train / (2 * train_neg)
    weight_for_positive = total_train / (2 * train_pos)

    weighted_bce = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32)
    )

    focal_loss = BinaryFocalLossWithLogits(
        alpha=weight_for_positive,
        gamma=float(config["loss"]["focal_gamma"]),
        reduction="mean"
    )

    # -----------------------------------------------------------------
    # Synthetic validation
    # -----------------------------------------------------------------

    synthetic_logits = torch.tensor(
        [-3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0, -2.0, 2.0, 0.2],
        dtype=torch.float32
    )

    synthetic_labels = torch.tensor(
        [0, 0, 0, 1, 1, 1, 1, 0, 1, 0],
        dtype=torch.float32
    )

    bce_value = float(weighted_bce(synthetic_logits, synthetic_labels).item())
    focal_value = float(focal_loss(synthetic_logits, synthetic_labels).item())

    unweighted_bce = nn.BCEWithLogitsLoss()
    unweighted_bce_value = float(unweighted_bce(synthetic_logits, synthetic_labels).item())

    synthetic_metrics = compute_binary_metrics_for_logits(
        synthetic_logits,
        synthetic_labels
    )

    validation_rows = [
        {
            "loss_name": "unweighted_bce_with_logits",
            "loss_value": unweighted_bce_value,
            "description": "Reference unweighted BCE loss."
        },
        {
            "loss_name": "weighted_bce_with_logits",
            "loss_value": bce_value,
            "description": "Default recommended loss for CPMR-Net."
        },
        {
            "loss_name": "binary_focal_loss_with_logits",
            "loss_value": focal_value,
            "description": "Optional alternative if recall remains weak."
        }
    ]

    validation_df = pd.DataFrame(validation_rows)
    validation_df.to_csv(TABLES_OUT / "loss_function_synthetic_validation.csv", index=False)

    class_balance_rows = [
        {
            "split": "train",
            "negative_normal": train_neg,
            "positive_anemia": train_pos,
            "total": int(len(train_df)),
            "positive_rate": float(train_pos / len(train_df))
        },
        {
            "split": "val",
            "negative_normal": int(val_counts.get(0, 0)),
            "positive_anemia": int(val_counts.get(1, 0)),
            "total": int(len(val_df)),
            "positive_rate": float(val_counts.get(1, 0) / len(val_df))
        },
        {
            "split": "test",
            "negative_normal": int(test_counts.get(0, 0)),
            "positive_anemia": int(test_counts.get(1, 0)),
            "total": int(len(test_df)),
            "positive_rate": float(test_counts.get(1, 0) / len(test_df))
        }
    ]

    class_balance_df = pd.DataFrame(class_balance_rows)
    class_balance_df.to_csv(TABLES_OUT / "loss_class_balance_summary.csv", index=False)

    loss_config = {
        "stage": "Stage10K",
        "title": "Loss Functions and Class-Imbalance Handling",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "default_loss": "weighted_bce_with_logits",
        "optional_loss": "binary_focal_loss_with_logits",
        "train_negative_normal": train_neg,
        "train_positive_anemia": train_pos,
        "train_total": int(len(train_df)),
        "positive_class_weight_for_bce": float(pos_weight),
        "balanced_weight_negative": float(weight_for_negative),
        "balanced_weight_positive": float(weight_for_positive),
        "focal_gamma": float(config["loss"]["focal_gamma"]),
        "focal_alpha": float(weight_for_positive),
        "threshold_policy": config["metrics"]["threshold_policy"],
        "primary_selection_metric": config["metrics"]["primary_selection_metric"],
        "synthetic_validation": {
            "unweighted_bce": unweighted_bce_value,
            "weighted_bce": bce_value,
            "focal_loss": focal_value,
            "synthetic_metrics_at_0_5": synthetic_metrics,
        },
        "recommendation": (
            "Use weighted BCEWithLogitsLoss as the default CPMR-Net training loss. "
            "Use focal loss only as a secondary experiment if anemia recall remains weak."
        ),
        "outputs_saved_to": str(STAGE_OUT),
    }

    with open(CONFIG_OUT / "CPMRNet_loss_config_v1.json", "w", encoding="utf-8") as f:
        json.dump(loss_config, f, indent=4, ensure_ascii=False)

    with open(STAGE_OUT / "Stage10K_Loss_Functions_Class_Imbalance_Summary.json", "w", encoding="utf-8") as f:
        json.dump(loss_config, f, indent=4, ensure_ascii=False)

    pd.DataFrame([
        {"setting": k, "value": json.dumps(v) if isinstance(v, (dict, list)) else v}
        for k, v in loss_config.items()
    ]).to_csv(TABLES_OUT / "loss_config_flattened.csv", index=False)

    report = []
    report.append("# Stage 10K Loss Functions and Class-Imbalance Handling\n")
    report.append(f"Generated at: {loss_config['created_at']}\n")

    report.append("## Purpose\n")
    report.append(
        "This stage defines the loss functions and class-imbalance handling strategy for CPMR-Net. "
        "Class weights are computed from the training split only to avoid information leakage from validation or test data.\n"
    )

    report.append("## Training Class Balance\n")
    report.append(f"- Normal participants in training: {train_neg}")
    report.append(f"- Anemia participants in training: {train_pos}")
    report.append(f"- Positive class weight for BCE: {pos_weight:.4f}")
    report.append(f"- Balanced negative weight: {weight_for_negative:.4f}")
    report.append(f"- Balanced positive weight: {weight_for_positive:.4f}\n")

    report.append("## Loss Strategy\n")
    report.append("- Default: weighted BCEWithLogitsLoss.")
    report.append("- Optional: binary focal loss if anemia recall remains weak.")
    report.append("- Threshold policy: default 0.5 and validation-derived Youden threshold.\n")

    report.append("## Synthetic Loss Validation\n")
    report.append(f"- Unweighted BCE: {unweighted_bce_value:.6f}")
    report.append(f"- Weighted BCE: {bce_value:.6f}")
    report.append(f"- Focal loss: {focal_value:.6f}\n")

    report.append("## Output Files\n")
    report.append("- `configs/CPMRNet_loss_config_v1.json`")
    report.append("- `loss_class_balance_summary.csv`")
    report.append("- `loss_function_synthetic_validation.csv`")
    report.append("- `loss_config_flattened.csv`")
    report.append("- `Stage10K_Loss_Functions_Class_Imbalance_Summary.json`\n")

    report.append("## Implementation Role\n")
    report.append(
        "Future training scripts should load `CPMRNet_loss_config_v1.json` and use weighted BCEWithLogitsLoss "
        "as the default supervised loss."
    )

    with open(REPORTS_OUT / "Stage10K_Loss_Functions_Class_Imbalance_Report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("=" * 80)
    print("STAGE 10K LOSS FUNCTIONS AND CLASS-IMBALANCE HANDLING COMPLETED")
    print("=" * 80)
    print(f"Training normal/anemia: {train_neg}/{train_pos}")
    print(f"Positive class weight for BCE: {pos_weight:.4f}")
    print(f"Default loss: weighted BCEWithLogitsLoss")
    print(f"Optional loss: binary focal loss")
    print(f"Synthetic unweighted BCE: {unweighted_bce_value:.6f}")
    print(f"Synthetic weighted BCE: {bce_value:.6f}")
    print(f"Synthetic focal loss: {focal_value:.6f}")
    print(f"Outputs saved to: {STAGE_OUT}")
    print("=" * 80)


if __name__ == "__main__":
    main()