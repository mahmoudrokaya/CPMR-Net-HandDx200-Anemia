# -*- coding: utf-8 -*-
"""
Stage 10Q - Training Strategy Revision and Stabilization Plan

Purpose:
Convert Stage 10N/10O/10P findings into a concrete stabilization plan.

No model training is performed.
"""

from pathlib import Path
import json
from datetime import datetime
import pandas as pd

BASE_DIR = Path(r"D:\47\472\New-Papers\Anemia_Paper")
OUTPUTS_DIR = BASE_DIR / "Outputs"

STAGE10N_SUMMARY = OUTPUTS_DIR / "Stage10N_Independent_Holdout_Test_Evaluation" / "Stage10N_Independent_Holdout_Test_Evaluation_Summary.json"
STAGE10O_SUMMARY = OUTPUTS_DIR / "Stage10O_Error_Analysis_Generalization_Diagnosis" / "Stage10O_Error_Analysis_Generalization_Diagnosis_Summary.json"
STAGE10P_SUMMARY = OUTPUTS_DIR / "Stage10P_Repeated_CrossValidation_Training" / "Stage10P_Repeated_CrossValidation_Training_Summary.json"

STAGE_OUT = OUTPUTS_DIR / "Stage10Q_Training_Strategy_Revision_Stabilization_Plan"
TABLES_OUT = STAGE_OUT / "tables"
REPORTS_OUT = STAGE_OUT / "reports"
CONFIG_OUT = STAGE_OUT / "configs"

for p in [TABLES_OUT, REPORTS_OUT, CONFIG_OUT]:
    p.mkdir(parents=True, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    for p in [STAGE10N_SUMMARY, STAGE10O_SUMMARY, STAGE10P_SUMMARY]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required input: {p}")

    n = load_json(STAGE10N_SUMMARY)
    o = load_json(STAGE10O_SUMMARY)
    p = load_json(STAGE10P_SUMMARY)

    diagnosis = [
        {
            "finding_id": "F1",
            "finding": "Validation-to-test generalization gap",
            "evidence": f"Validation ROC-AUC {o['validation_metrics_at_threshold']['roc_auc']:.4f} vs test ROC-AUC {o['test_metrics_at_threshold']['roc_auc']:.4f}",
            "interpretation": "The first holdout result overestimated generalization.",
            "severity": "High"
        },
        {
            "finding_id": "F2",
            "finding": "Repeated-CV confirms weak generalization",
            "evidence": f"Mean repeated-CV test ROC-AUC {p['mean_test_roc_auc']:.4f} ± {p['std_test_roc_auc']:.4f}",
            "interpretation": "The problem is systematic, not only one unlucky test split.",
            "severity": "High"
        },
        {
            "finding_id": "F3",
            "finding": "Positive-class recall remains unstable",
            "evidence": f"Repeated-CV mean recall {p['mean_test_recall']:.4f}",
            "interpretation": "The model is still missing anemia cases.",
            "severity": "High"
        },
        {
            "finding_id": "F4",
            "finding": "Current end-to-end image training may be too data-hungry",
            "evidence": "198 participants but 730,918 trainable parameters.",
            "interpretation": "The architecture is structurally valid but likely underconstrained for the sample size.",
            "severity": "High"
        }
    ]

    interventions = [
        {
            "priority": 1,
            "intervention_id": "S1",
            "name": "Freeze representation encoders initially",
            "action": "Train only attention, fusion, and classifier layers first; optionally unfreeze upper encoder blocks later.",
            "reason": "Reduces overfitting and stabilizes learning with only 198 participants.",
            "risk": "Low",
            "expected_effect": "Improved generalization and reduced validation-test instability."
        },
        {
            "priority": 2,
            "intervention_id": "S2",
            "name": "Use pretrained encoders as controlled baselines",
            "action": "Add ResNet-18 or MobileNetV3-Small pretrained encoder variants.",
            "reason": "From-scratch encoders may not learn robust visual features from 126 training participants.",
            "risk": "Medium",
            "expected_effect": "Potentially stronger feature extraction and better test ranking."
        },
        {
            "priority": 3,
            "intervention_id": "S3",
            "name": "Train RGB-only model before full multimodal fusion",
            "action": "Run RGB-only CPMR-Net branch as the primary stabilized model.",
            "reason": "Handcrafted stages showed RGB signals are stronger than thermal signals.",
            "risk": "Low",
            "expected_effect": "Clarifies whether thermal branch is adding noise."
        },
        {
            "priority": 4,
            "intervention_id": "S4",
            "name": "Reduce representation burden",
            "action": "Start with strongest RGB representations: RGB original, HSV, LAB, texture, and selected patches.",
            "reason": "Too many representation paths may amplify noise and overfitting.",
            "risk": "Low",
            "expected_effect": "Simpler model with better bias-variance balance."
        },
        {
            "priority": 5,
            "intervention_id": "S5",
            "name": "Add validation-calibrated threshold reporting only",
            "action": "Keep threshold selection on validation, but report ROC-AUC/PR-AUC as primary test metrics.",
            "reason": "Threshold metrics are unstable with small test folds.",
            "risk": "Low",
            "expected_effect": "More reliable performance interpretation."
        },
        {
            "priority": 6,
            "intervention_id": "S6",
            "name": "Compare against embedding-level classical models",
            "action": "Use Stage 10D/10E/10F embeddings with SVM/Logistic Regression as a low-variance comparator.",
            "reason": "May reveal whether learned encoders or classifier/fusion training is the main bottleneck.",
            "risk": "Low",
            "expected_effect": "Provides diagnostic bridge between handcrafted ML and CPMR-Net."
        }
    ]

    revised_training_plan = [
        {
            "stage": "10R",
            "name": "Frozen-Encoder CPMR-Net Training",
            "objective": "Train attention/fusion/classifier while freezing representation encoders.",
            "success_criterion": "Repeated-CV ROC-AUC improves over Stage 10P and recall stabilizes."
        },
        {
            "stage": "10S",
            "name": "RGB-Only Stabilized CPMR-Net",
            "objective": "Train the strongest RGB-centered model without thermal branch.",
            "success_criterion": "Demonstrates whether RGB-only generalizes better than full fusion."
        },
        {
            "stage": "10T",
            "name": "Pretrained Encoder Baselines",
            "objective": "Evaluate ResNet-18/MobileNetV3-Small frozen or partially frozen encoders.",
            "success_criterion": "Improves ranking metrics versus from-scratch lightweight encoders."
        },
        {
            "stage": "10U",
            "name": "Embedding-Level Classical Diagnostic Models",
            "objective": "Train SVM/Logistic models on learned validation embeddings.",
            "success_criterion": "Identifies whether neural classifier/fusion or representation learning is the bottleneck."
        }
    ]

    diagnosis_df = pd.DataFrame(diagnosis)
    interventions_df = pd.DataFrame(interventions)
    revised_plan_df = pd.DataFrame(revised_training_plan)

    diagnosis_df.to_csv(TABLES_OUT / "generalization_failure_diagnosis.csv", index=False)
    interventions_df.to_csv(TABLES_OUT / "stabilization_intervention_priority_table.csv", index=False)
    revised_plan_df.to_csv(TABLES_OUT / "revised_training_stage_plan.csv", index=False)

    recommended_config = {
        "stage": "Stage10Q",
        "title": "Training Strategy Revision and Stabilization Plan",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "core_conclusion": "Current CPMR-Net is structurally valid but not yet generalizing sufficiently.",
        "primary_failure_mode": "Systematic weak generalization with unstable anemia recall.",
        "recommended_next_stage": "Stage10R_Frozen_Encoder_CPMRNet_Training",
        "recommended_changes": {
            "freeze_representation_encoders": True,
            "train_attention_fusion_classifier_only_first": True,
            "keep_weighted_bce": True,
            "keep_validation_only_threshold_selection": True,
            "prioritize_rgb_branch_diagnostics": True,
            "avoid_claiming_final_superiority": True
        },
        "evidence": {
            "holdout_test_roc_auc": n["validation_threshold_metrics"]["roc_auc"],
            "repeated_cv_mean_test_roc_auc": p["mean_test_roc_auc"],
            "repeated_cv_std_test_roc_auc": p["std_test_roc_auc"],
            "repeated_cv_mean_recall": p["mean_test_recall"],
            "repeated_cv_mean_f1": p["mean_test_f1"]
        },
        "outputs_saved_to": str(STAGE_OUT)
    }

    with open(CONFIG_OUT / "CPMRNet_stabilization_plan_v1.json", "w", encoding="utf-8") as f:
        json.dump(recommended_config, f, indent=4, ensure_ascii=False)

    with open(STAGE_OUT / "Stage10Q_Training_Strategy_Revision_Stabilization_Plan_Summary.json", "w", encoding="utf-8") as f:
        json.dump(recommended_config, f, indent=4, ensure_ascii=False)

    report = []
    report.append("# Stage 10Q Training Strategy Revision and Stabilization Plan\n")
    report.append(f"Generated at: {recommended_config['created_at']}\n")

    report.append("## Purpose\n")
    report.append(
        "This stage converts the validation, independent test, error-analysis, and repeated cross-validation findings "
        "into a concrete stabilization plan before further training.\n"
    )

    report.append("## Core Conclusion\n")
    report.append(
        "The current CPMR-Net architecture is structurally valid, but the first supervised training strategy does not yet "
        "generalize sufficiently. Repeated cross-validation confirms that the weak performance is systematic rather than only "
        "a single split artifact.\n"
    )

    report.append("## Key Evidence\n")
    report.append(f"- Holdout test ROC-AUC: {n['validation_threshold_metrics']['roc_auc']:.4f}")
    report.append(f"- Repeated-CV mean test ROC-AUC: {p['mean_test_roc_auc']:.4f} ± {p['std_test_roc_auc']:.4f}")
    report.append(f"- Repeated-CV mean recall: {p['mean_test_recall']:.4f}")
    report.append(f"- Repeated-CV mean F1: {p['mean_test_f1']:.4f}\n")

    report.append("## Prioritized Stabilization Actions\n")
    for item in interventions:
        report.append(f"### {item['priority']}. {item['name']}")
        report.append(f"- Action: {item['action']}")
        report.append(f"- Reason: {item['reason']}")
        report.append(f"- Expected effect: {item['expected_effect']}\n")

    report.append("## Recommended Next Stage\n")
    report.append("**Stage 10R — Frozen-Encoder CPMR-Net Training**\n")
    report.append(
        "The next experiment should freeze the representation encoders and train only the representation attention, "
        "view attention, RGB/thermal fusion, and classification head. This is the safest first stabilization step.\n"
    )

    report.append("## Output Files\n")
    report.append("- `generalization_failure_diagnosis.csv`")
    report.append("- `stabilization_intervention_priority_table.csv`")
    report.append("- `revised_training_stage_plan.csv`")
    report.append("- `configs/CPMRNet_stabilization_plan_v1.json`")
    report.append("- `Stage10Q_Training_Strategy_Revision_Stabilization_Plan_Summary.json`")

    with open(REPORTS_OUT / "Stage10Q_Training_Strategy_Revision_Stabilization_Plan_Report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("=" * 80)
    print("STAGE 10Q TRAINING STRATEGY REVISION AND STABILIZATION PLAN COMPLETED")
    print("=" * 80)
    print("Core conclusion: current CPMR-Net is structurally valid but not yet generalizing sufficiently.")
    print(f"Repeated-CV mean test ROC-AUC: {p['mean_test_roc_auc']:.4f} ± {p['std_test_roc_auc']:.4f}")
    print(f"Repeated-CV mean recall: {p['mean_test_recall']:.4f}")
    print("Recommended next stage: Stage 10R - Frozen-Encoder CPMR-Net Training")
    print(f"Outputs saved to: {STAGE_OUT}")
    print("=" * 80)


if __name__ == "__main__":
    main()