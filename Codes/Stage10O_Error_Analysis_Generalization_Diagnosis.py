# -*- coding: utf-8 -*-
"""
Stage 10O - Error Analysis and Generalization Diagnosis

Purpose:
Diagnose why CPMR-Net validation performance was strong but independent test
performance dropped.

No model training is performed.
"""

from pathlib import Path
import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix


BASE_DIR = Path(r"D:\47\472\New-Papers\Anemia_Paper")
OUTPUTS_DIR = BASE_DIR / "Outputs"

STAGE10M_DIR = OUTPUTS_DIR / "Stage10M_Validation_Engine_Threshold_Analysis"
STAGE10N_DIR = OUTPUTS_DIR / "Stage10N_Independent_Holdout_Test_Evaluation"

VAL_PRED_FILE = STAGE10M_DIR / "tables" / "validation_predictions_with_thresholds.csv"
VAL_SUMMARY_FILE = STAGE10M_DIR / "Stage10M_Validation_Engine_Threshold_Analysis_Summary.json"

TEST_PRED_FILE = STAGE10N_DIR / "predictions" / "independent_test_predictions.csv"
TEST_SUMMARY_FILE = STAGE10N_DIR / "Stage10N_Independent_Holdout_Test_Evaluation_Summary.json"

STAGE_OUT = OUTPUTS_DIR / "Stage10O_Error_Analysis_Generalization_Diagnosis"
TABLES_OUT = STAGE_OUT / "tables"
REPORTS_OUT = STAGE_OUT / "reports"

TABLES_OUT.mkdir(parents=True, exist_ok=True)
REPORTS_OUT.mkdir(parents=True, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def probability_summary(df, split_name):
    rows = []

    for label_value, label_name in [(0, "Normal"), (1, "Anemia")]:
        g = df[df["label"] == label_value]

        rows.append({
            "split": split_name,
            "class_label": label_value,
            "class_name": label_name,
            "count": int(len(g)),
            "mean_probability": float(g["probability"].mean()) if len(g) else np.nan,
            "std_probability": float(g["probability"].std()) if len(g) else np.nan,
            "min_probability": float(g["probability"].min()) if len(g) else np.nan,
            "q25_probability": float(g["probability"].quantile(0.25)) if len(g) else np.nan,
            "median_probability": float(g["probability"].median()) if len(g) else np.nan,
            "q75_probability": float(g["probability"].quantile(0.75)) if len(g) else np.nan,
            "max_probability": float(g["probability"].max()) if len(g) else np.nan,
        })

    return rows


def classify_errors(df, threshold, split_name):
    out = df.copy()
    out["prediction"] = (out["probability"] >= threshold).astype(int)

    conditions = []
    for _, row in out.iterrows():
        if row["label"] == 1 and row["prediction"] == 1:
            conditions.append("TP")
        elif row["label"] == 0 and row["prediction"] == 0:
            conditions.append("TN")
        elif row["label"] == 0 and row["prediction"] == 1:
            conditions.append("FP")
        elif row["label"] == 1 and row["prediction"] == 0:
            conditions.append("FN")
        else:
            conditions.append("Unknown")

    out["confusion_group"] = conditions
    out["split"] = split_name
    out["distance_from_threshold"] = np.abs(out["probability"] - threshold)

    return out


def metric_summary(df, threshold, split_name):
    y_true = df["label"].astype(int).values
    y_prob = df["probability"].astype(float).values
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    precision = tp / (tp + fp) if (tp + fp) else 0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    balanced_accuracy = (sensitivity + specificity) / 2

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = np.nan

    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except Exception:
        pr_auc = np.nan

    return {
        "split": split_name,
        "threshold": threshold,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main():
    for p in [VAL_PRED_FILE, VAL_SUMMARY_FILE, TEST_PRED_FILE, TEST_SUMMARY_FILE]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required input: {p}")

    val_df = pd.read_csv(VAL_PRED_FILE)
    test_df = pd.read_csv(TEST_PRED_FILE)

    val_summary = load_json(VAL_SUMMARY_FILE)
    test_summary = load_json(TEST_SUMMARY_FILE)

    threshold = float(val_summary["recommended_threshold"])

    val_errors = classify_errors(val_df, threshold, "validation")
    test_errors = classify_errors(test_df, threshold, "test")

    val_errors.to_csv(TABLES_OUT / "validation_error_cases.csv", index=False)
    test_errors.to_csv(TABLES_OUT / "test_error_cases.csv", index=False)

    combined_errors = pd.concat([val_errors, test_errors], ignore_index=True)
    combined_errors.to_csv(TABLES_OUT / "combined_validation_test_error_cases.csv", index=False)

    prob_summary_df = pd.DataFrame(
        probability_summary(val_df, "validation") +
        probability_summary(test_df, "test")
    )
    prob_summary_df.to_csv(TABLES_OUT / "probability_distribution_by_class.csv", index=False)

    error_group_summary = (
        combined_errors.groupby(["split", "confusion_group"])
        .agg(
            count=("participant_id", "count"),
            mean_probability=("probability", "mean"),
            std_probability=("probability", "std"),
            mean_distance_from_threshold=("distance_from_threshold", "mean"),
        )
        .reset_index()
    )
    error_group_summary.to_csv(TABLES_OUT / "confusion_group_probability_summary.csv", index=False)

    threshold_rows = []
    for th in np.round(np.arange(0.05, 0.96, 0.05), 2):
        threshold_rows.append(metric_summary(val_df, th, "validation"))
        threshold_rows.append(metric_summary(test_df, th, "test"))

    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df.to_csv(TABLES_OUT / "validation_test_threshold_sensitivity.csv", index=False)

    val_metrics = metric_summary(val_df, threshold, "validation")
    test_metrics = metric_summary(test_df, threshold, "test")

    gap_rows = []
    for metric in ["accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc"]:
        gap_rows.append({
            "metric": metric,
            "validation": val_metrics[metric],
            "test": test_metrics[metric],
            "test_minus_validation": test_metrics[metric] - val_metrics[metric],
        })

    gap_df = pd.DataFrame(gap_rows)
    gap_df.to_csv(TABLES_OUT / "validation_test_generalization_gap.csv", index=False)

    if "rgb_modality_weight" in test_df.columns and "thermal_modality_weight" in test_df.columns:
        modality_error_summary = (
            test_errors.groupby("confusion_group")
            .agg(
                count=("participant_id", "count"),
                mean_rgb_weight=("rgb_modality_weight", "mean"),
                mean_thermal_weight=("thermal_modality_weight", "mean"),
                std_rgb_weight=("rgb_modality_weight", "std"),
                std_thermal_weight=("thermal_modality_weight", "std"),
            )
            .reset_index()
        )
        modality_error_summary.to_csv(TABLES_OUT / "test_modality_weight_by_error_group.csv", index=False)
    else:
        modality_error_summary = pd.DataFrame()
        modality_error_summary.to_csv(TABLES_OUT / "test_modality_weight_by_error_group.csv", index=False)

    diagnosis_points = []

    if test_metrics["roc_auc"] < val_metrics["roc_auc"] - 0.15:
        diagnosis_points.append(
            "Large validation-to-test ROC-AUC drop suggests limited generalization or high split sensitivity."
        )

    if test_metrics["recall"] < val_metrics["recall"] - 0.25:
        diagnosis_points.append(
            "Anemia recall dropped substantially on the test set, indicating missed positive cases."
        )

    if test_metrics["pr_auc"] < val_metrics["pr_auc"] - 0.20:
        diagnosis_points.append(
            "PR-AUC dropped substantially, suggesting weaker positive-class ranking on the test split."
        )

    if test_metrics["specificity"] > val_metrics["specificity"]:
        diagnosis_points.append(
            "Specificity improved relative to validation, but recall declined, indicating a stricter operating behavior on test data."
        )

    if not diagnosis_points:
        diagnosis_points.append(
            "No single dominant failure pattern was detected, but test performance remains weaker than validation."
        )

    summary = {
        "stage": "Stage10O",
        "title": "Error Analysis and Generalization Diagnosis",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "validation_threshold": threshold,
        "validation_metrics_at_threshold": val_metrics,
        "test_metrics_at_threshold": test_metrics,
        "roc_auc_gap_test_minus_validation": test_metrics["roc_auc"] - val_metrics["roc_auc"],
        "pr_auc_gap_test_minus_validation": test_metrics["pr_auc"] - val_metrics["pr_auc"],
        "balanced_accuracy_gap_test_minus_validation": test_metrics["balanced_accuracy"] - val_metrics["balanced_accuracy"],
        "diagnosis_points": diagnosis_points,
        "outputs_saved_to": str(STAGE_OUT),
    }

    with open(STAGE_OUT / "Stage10O_Error_Analysis_Generalization_Diagnosis_Summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    report = []
    report.append("# Stage 10O Error Analysis and Generalization Diagnosis\n")
    report.append(f"Generated at: {summary['created_at']}\n")

    report.append("## Purpose\n")
    report.append(
        "This stage diagnoses the generalization gap between the validation and independent test results of CPMR-Net.\n"
    )

    report.append("## Threshold\n")
    report.append(f"- Validation-derived threshold: {threshold:.2f}\n")

    report.append("## Validation vs Test Metrics\n")
    for row in gap_rows:
        report.append(
            f"- {row['metric']}: validation={row['validation']:.4f}, "
            f"test={row['test']:.4f}, gap={row['test_minus_validation']:.4f}"
        )

    report.append("\n## Diagnostic Interpretation\n")
    for point in diagnosis_points:
        report.append(f"- {point}")

    report.append("\n## Output Files\n")
    report.append("- `validation_error_cases.csv`")
    report.append("- `test_error_cases.csv`")
    report.append("- `combined_validation_test_error_cases.csv`")
    report.append("- `probability_distribution_by_class.csv`")
    report.append("- `confusion_group_probability_summary.csv`")
    report.append("- `validation_test_threshold_sensitivity.csv`")
    report.append("- `validation_test_generalization_gap.csv`")
    report.append("- `test_modality_weight_by_error_group.csv`")
    report.append("- `Stage10O_Error_Analysis_Generalization_Diagnosis_Summary.json`\n")

    report.append("## Recommended Next Step\n")
    report.append(
        "The next stage should not immediately claim final superiority. Instead, Stage 10P should run repeated "
        "cross-validation training to determine whether the weak independent test result reflects split sensitivity "
        "or a systematic limitation of the current CPMR-Net training strategy."
    )

    with open(REPORTS_OUT / "Stage10O_Error_Analysis_Generalization_Diagnosis_Report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("=" * 80)
    print("STAGE 10O ERROR ANALYSIS AND GENERALIZATION DIAGNOSIS COMPLETED")
    print("=" * 80)
    print(f"Validation threshold: {threshold:.2f}")
    print(f"Validation ROC-AUC: {val_metrics['roc_auc']:.4f}")
    print(f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"ROC-AUC gap: {summary['roc_auc_gap_test_minus_validation']:.4f}")
    print(f"Validation balanced accuracy: {val_metrics['balanced_accuracy']:.4f}")
    print(f"Test balanced accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print("Diagnosis:")
    for point in diagnosis_points:
        print(f"  - {point}")
    print(f"Outputs saved to: {STAGE_OUT}")
    print("=" * 80)


if __name__ == "__main__":
    main()