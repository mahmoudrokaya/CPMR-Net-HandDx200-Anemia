# -*- coding: utf-8 -*-
"""
Stage 10M - Validation Engine and Threshold Analysis

Purpose:
Analyze Stage 10L validation predictions and derive threshold-dependent metrics.

This stage:
- Loads best_validation_predictions.csv from Stage 10L
- Computes threshold-independent metrics
- Searches thresholds from 0.01 to 0.99
- Selects:
    1) default 0.50 threshold
    2) Youden J threshold
    3) F1-optimal threshold
    4) balanced-accuracy-optimal threshold
    5) recall-oriented threshold
- Saves threshold analysis tables and final recommended validation threshold

No model training is performed.
"""

from pathlib import Path
import json
from datetime import datetime

import numpy as np
import pandas as pd

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


BASE_DIR = Path(r"D:\47\472\New-Papers\Anemia_Paper")
OUTPUTS_DIR = BASE_DIR / "Outputs"

STAGE10L_DIR = OUTPUTS_DIR / "Stage10L_CPMRNet_Training_Engine"
PRED_FILE = STAGE10L_DIR / "predictions" / "best_validation_predictions.csv"

STAGE_OUT = OUTPUTS_DIR / "Stage10M_Validation_Engine_Threshold_Analysis"
TABLES_OUT = STAGE_OUT / "tables"
REPORTS_OUT = STAGE_OUT / "reports"
CONFIG_OUT = STAGE_OUT / "configs"

for p in [TABLES_OUT, REPORTS_OUT, CONFIG_OUT]:
    p.mkdir(parents=True, exist_ok=True)


def compute_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "ppv": float(ppv),
        "npv": float(npv),
        "youden_j": float(sensitivity + specificity - 1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def safe_auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_prob))


def safe_pr_auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(average_precision_score(y_true, y_prob))


def main():
    if not PRED_FILE.exists():
        raise FileNotFoundError(f"Missing Stage 10L validation predictions: {PRED_FILE}")

    pred_df = pd.read_csv(PRED_FILE)

    required_cols = ["participant_id", "label", "probability"]

    missing_cols = [c for c in required_cols if c not in pred_df.columns]

    if missing_cols:
        raise ValueError(f"Missing required prediction columns: {missing_cols}")

    y_true = pred_df["label"].astype(int).values
    y_prob = pred_df["probability"].astype(float).values

    roc_auc = safe_auc(y_true, y_prob)
    pr_auc = safe_pr_auc(y_true, y_prob)
    brier = float(brier_score_loss(y_true, y_prob))

    thresholds = np.round(np.arange(0.01, 1.00, 0.01), 2)

    threshold_rows = []

    for threshold in thresholds:
        row = compute_metrics(y_true, y_prob, threshold)
        row["roc_auc"] = roc_auc
        row["pr_auc"] = pr_auc
        row["brier_score"] = brier
        threshold_rows.append(row)

    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df.to_csv(TABLES_OUT / "validation_threshold_sweep.csv", index=False)

    default_row = compute_metrics(y_true, y_prob, 0.50)
    default_row["selection_rule"] = "default_0_50"

    youden_row = threshold_df.sort_values(
        ["youden_j", "balanced_accuracy", "f1"],
        ascending=False
    ).iloc[0].to_dict()
    youden_row["selection_rule"] = "youden_j_max"

    f1_row = threshold_df.sort_values(
        ["f1", "balanced_accuracy", "youden_j"],
        ascending=False
    ).iloc[0].to_dict()
    f1_row["selection_rule"] = "f1_max"

    balacc_row = threshold_df.sort_values(
        ["balanced_accuracy", "youden_j", "f1"],
        ascending=False
    ).iloc[0].to_dict()
    balacc_row["selection_rule"] = "balanced_accuracy_max"

    recall_candidates = threshold_df[threshold_df["recall"] >= 0.75].copy()

    if len(recall_candidates) > 0:
        recall_row = recall_candidates.sort_values(
            ["specificity", "f1", "balanced_accuracy"],
            ascending=False
        ).iloc[0].to_dict()
        recall_row["selection_rule"] = "recall_at_least_0_75_best_specificity"
    else:
        recall_row = threshold_df.sort_values(
            ["recall", "specificity", "f1"],
            ascending=False
        ).iloc[0].to_dict()
        recall_row["selection_rule"] = "max_recall_available"

    selected_rows = pd.DataFrame([
        default_row,
        youden_row,
        f1_row,
        balacc_row,
        recall_row,
    ])

    selected_rows.to_csv(TABLES_OUT / "selected_validation_thresholds.csv", index=False)

    recommended_threshold = float(youden_row["threshold"])

    pred_df["prediction_default_0_50"] = (y_prob >= 0.50).astype(int)
    pred_df["prediction_youden"] = (y_prob >= recommended_threshold).astype(int)
    pred_df.to_csv(TABLES_OUT / "validation_predictions_with_thresholds.csv", index=False)

    final_threshold_config = {
        "stage": "Stage10M",
        "title": "Validation Engine and Threshold Analysis",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "prediction_file": str(PRED_FILE),
        "validation_participants": int(len(pred_df)),
        "validation_positive_anemia": int(np.sum(y_true == 1)),
        "validation_negative_normal": int(np.sum(y_true == 0)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "recommended_threshold_rule": "youden_j_max",
        "recommended_threshold": recommended_threshold,
        "default_threshold": 0.50,
        "selected_thresholds": selected_rows.to_dict(orient="records"),
        "outputs_saved_to": str(STAGE_OUT),
    }

    with open(CONFIG_OUT / "CPMRNet_validation_threshold_config_v1.json", "w", encoding="utf-8") as f:
        json.dump(final_threshold_config, f, indent=4, ensure_ascii=False)

    with open(STAGE_OUT / "Stage10M_Validation_Engine_Threshold_Analysis_Summary.json", "w", encoding="utf-8") as f:
        json.dump(final_threshold_config, f, indent=4, ensure_ascii=False)

    report = []
    report.append("# Stage 10M Validation Engine and Threshold Analysis\n")
    report.append(f"Generated at: {final_threshold_config['created_at']}\n")

    report.append("## Purpose\n")
    report.append(
        "This stage analyzes validation predictions from the best Stage 10L checkpoint and derives "
        "threshold-dependent operating points for CPMR-Net.\n"
    )

    report.append("## Validation Set\n")
    report.append(f"- Participants: {final_threshold_config['validation_participants']}")
    report.append(f"- Anemia: {final_threshold_config['validation_positive_anemia']}")
    report.append(f"- Normal: {final_threshold_config['validation_negative_normal']}\n")

    report.append("## Threshold-Independent Metrics\n")
    report.append(f"- ROC-AUC: {roc_auc:.4f}")
    report.append(f"- PR-AUC: {pr_auc:.4f}")
    report.append(f"- Brier score: {brier:.4f}\n")

    report.append("## Selected Thresholds\n")
    for _, row in selected_rows.iterrows():
        report.append(f"### {row['selection_rule']}")
        report.append(f"- Threshold: {row['threshold']:.2f}")
        report.append(f"- Accuracy: {row['accuracy']:.4f}")
        report.append(f"- Balanced accuracy: {row['balanced_accuracy']:.4f}")
        report.append(f"- Precision: {row['precision']:.4f}")
        report.append(f"- Recall: {row['recall']:.4f}")
        report.append(f"- Specificity: {row['specificity']:.4f}")
        report.append(f"- F1: {row['f1']:.4f}")
        report.append(f"- MCC: {row['mcc']:.4f}")
        report.append(f"- Confusion matrix: TN={int(row['tn'])}, FP={int(row['fp'])}, FN={int(row['fn'])}, TP={int(row['tp'])}\n")

    report.append("## Recommended Threshold\n")
    report.append(f"The recommended validation-derived threshold is **{recommended_threshold:.2f}**, selected by Youden J.\n")

    report.append("## Output Files\n")
    report.append("- `validation_threshold_sweep.csv`")
    report.append("- `selected_validation_thresholds.csv`")
    report.append("- `validation_predictions_with_thresholds.csv`")
    report.append("- `configs/CPMRNet_validation_threshold_config_v1.json`")
    report.append("- `Stage10M_Validation_Engine_Threshold_Analysis_Summary.json`\n")

    report.append("## Implementation Role\n")
    report.append(
        "The threshold selected here should be applied to the independent holdout test set in the next testing stage, "
        "without re-optimizing on the test set."
    )

    with open(REPORTS_OUT / "Stage10M_Validation_Engine_Threshold_Analysis_Report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("=" * 80)
    print("STAGE 10M VALIDATION ENGINE AND THRESHOLD ANALYSIS COMPLETED")
    print("=" * 80)
    print(f"Validation participants: {len(pred_df)}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"Brier score: {brier:.4f}")
    print(f"Recommended threshold: {recommended_threshold:.2f} via Youden J")
    print(f"Outputs saved to: {STAGE_OUT}")
    print("=" * 80)


if __name__ == "__main__":
    main()