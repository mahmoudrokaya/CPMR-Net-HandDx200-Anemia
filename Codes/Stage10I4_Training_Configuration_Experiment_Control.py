# -*- coding: utf-8 -*-
"""
Stage 10I4 - Training Configuration and Experiment Control File

Purpose:
Create reproducible experiment-control files for CPMR-Net training.

This stage:
- Defines official paths
- Defines data protocol
- Defines model configuration
- Defines augmentation policy
- Defines optimization settings
- Defines metrics
- Defines checkpointing and early stopping
- Defines ablation experiment IDs

No model training is performed.
"""

from pathlib import Path
import json
from datetime import datetime
import pandas as pd


BASE_DIR = Path(r"D:\47\472\New-Papers\Anemia_Paper")
OUTPUTS_DIR = BASE_DIR / "Outputs"

STAGE10A_DIR = OUTPUTS_DIR / "Stage10A_ParticipantLevel_Dataset_Loader"
STAGE10B_DIR = OUTPUTS_DIR / "Stage10B_MultiRepresentation_Generator"
STAGE10I1_DIR = OUTPUTS_DIR / "Stage10I1_Participant_Level_Split_Strategy"
STAGE10I3_DIR = OUTPUTS_DIR / "Stage10I3_Training_Time_Data_Augmentation_Strategy"

STAGE_OUT = OUTPUTS_DIR / "Stage10I4_Training_Configuration_Experiment_Control"
TABLES_OUT = STAGE_OUT / "tables"
REPORTS_OUT = STAGE_OUT / "reports"
CONFIG_OUT = STAGE_OUT / "configs"

for p in [TABLES_OUT, REPORTS_OUT, CONFIG_OUT]:
    p.mkdir(parents=True, exist_ok=True)


created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
experiment_id = "CPMRNet_HandDx200_Stage10I4_Config_v1"


config = {
    "experiment_id": experiment_id,
    "created_at": created_at,
    "project": {
        "name": "HandDx-200 Binary Anemia Classification",
        "architecture": "Cooperative Physiological Multi-Representation Network",
        "short_name": "CPMR-Net",
        "task": "Binary anemia classification",
        "diagnostic_unit": "participant",
        "num_participants": 198,
        "num_classes": 2,
        "class_names": ["Normal", "Anemia"],
    },
    "paths": {
        "base_dir": str(BASE_DIR),
        "outputs_dir": str(OUTPUTS_DIR),
        "participant_manifest": str(STAGE10A_DIR / "tables" / "valid_participant_level_manifest.csv"),
        "representation_manifest": str(STAGE10B_DIR / "tables" / "multi_representation_manifest.csv"),
        "holdout_split": str(STAGE10I1_DIR / "tables" / "holdout_train_val_test_split.csv"),
        "repeated_splits": str(STAGE10I1_DIR / "tables" / "repeated_stratified_5fold_train_val_test_splits.csv"),
        "augmentation_rules": str(STAGE10I3_DIR / "tables" / "augmentation_strategy_rules.csv"),
        "training_output_root": str(OUTPUTS_DIR / "Stage10J_CPMRNet_Training"),
    },
    "data": {
        "batch_size": 4,
        "num_workers": 0,
        "image_size": 224,
        "patch_size": 112,
        "representations_per_participant": 64,
        "modalities": ["rgb", "thermal"],
        "views": ["l_dorsal", "l_palmar", "r_dorsal", "r_palmar"],
        "rgb_representations": [
            "rgb_original",
            "rgb_hsv",
            "rgb_lab",
            "rgb_texture",
            "patch_center",
            "patch_upper",
            "patch_lower",
            "patch_left",
            "patch_right",
        ],
        "thermal_representations": [
            "thermal_normalized",
            "thermal_texture",
            "patch_center",
            "patch_upper",
            "patch_lower",
            "patch_left",
            "patch_right",
        ],
        "split_protocol": "participant_level_holdout_and_repeated_5x5_cv",
        "primary_training_protocol": "holdout_train_val_test",
        "final_validation_protocol": "repeated_stratified_5fold_5repeats",
    },
    "model": {
        "encoder_type": "custom_lightweight_cnn",
        "embedding_dim": 128,
        "rgb_encoder": "shared_rgb_encoder",
        "rgb_texture_encoder": "shared_rgb_1channel_encoder",
        "thermal_encoder": "shared_thermal_encoder",
        "specialist_aggregation": "trainable_representation_attention",
        "view_branch_aggregation": "trainable_view_attention",
        "fusion": "trainable_adaptive_rgb_thermal_cooperative_fusion",
        "classification_head": "mlp_binary_classifier",
        "dropout": 0.25,
        "use_symmetry_loss": False,
        "symmetry_loss_weight": 0.0,
    },
    "augmentation": {
        "policy": "mild_train_only",
        "apply_to_train": True,
        "apply_to_val": False,
        "apply_to_test": False,
        "small_rotation_degrees": 7,
        "small_translation_fraction": 0.05,
        "scale_range": [0.95, 1.05],
        "rgb_brightness_contrast_jitter": 0.08,
        "gaussian_noise_probability": 0.15,
        "thermal_intensity_jitter": 0.05,
        "disabled": [
            "hue_shift",
            "strong_saturation_jitter",
            "cutout",
            "random_erasing",
            "large_geometric_distortions",
        ],
    },
    "optimization": {
        "optimizer": "AdamW",
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "max_epochs": 100,
        "early_stopping_patience": 15,
        "lr_scheduler": "ReduceLROnPlateau",
        "scheduler_monitor": "val_roc_auc",
        "scheduler_factor": 0.5,
        "scheduler_patience": 5,
        "gradient_clip_norm": 1.0,
        "random_seed": 42,
    },
    "loss": {
        "primary_loss": "weighted_binary_cross_entropy",
        "class_imbalance_handling": "positive_class_weight",
        "focal_loss_enabled": False,
        "focal_gamma": 2.0,
        "auxiliary_losses": {
            "symmetry_loss": False,
            "attention_entropy_regularization": False,
        },
    },
    "metrics": {
        "primary_selection_metric": "val_roc_auc",
        "classification_metrics": [
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "specificity",
            "f1",
            "roc_auc",
            "pr_auc",
            "mcc",
        ],
        "calibration_metrics": [
            "brier_score",
            "expected_calibration_error",
        ],
        "report_confusion_matrix": True,
        "threshold_policy": "default_0.5_and_youden_validation_threshold",
    },
    "checkpointing": {
        "save_best_model": True,
        "save_last_model": True,
        "monitor": "val_roc_auc",
        "mode": "max",
        "save_attention_outputs": True,
        "save_predictions": True,
        "save_training_curves": True,
    },
    "ablation_plan": [
        {
            "ablation_id": "A1",
            "name": "RGB_only_participant_model",
            "use_rgb": True,
            "use_thermal": False,
            "use_multi_representation": True,
            "use_specialists": True,
            "use_adaptive_fusion": False,
            "use_symmetry_loss": False,
        },
        {
            "ablation_id": "A2",
            "name": "Thermal_only_participant_model",
            "use_rgb": False,
            "use_thermal": True,
            "use_multi_representation": True,
            "use_specialists": True,
            "use_adaptive_fusion": False,
            "use_symmetry_loss": False,
        },
        {
            "ablation_id": "A3",
            "name": "RGB_Thermal_equal_fusion",
            "use_rgb": True,
            "use_thermal": True,
            "use_multi_representation": True,
            "use_specialists": True,
            "use_adaptive_fusion": False,
            "fusion_type": "equal_mean",
            "use_symmetry_loss": False,
        },
        {
            "ablation_id": "A4",
            "name": "RGB_centered_auxiliary_thermal_fusion",
            "use_rgb": True,
            "use_thermal": True,
            "use_multi_representation": True,
            "use_specialists": True,
            "use_adaptive_fusion": False,
            "fusion_type": "fixed_rgb_70_thermal_30",
            "use_symmetry_loss": False,
        },
        {
            "ablation_id": "A5",
            "name": "Without_multi_representation",
            "use_rgb": True,
            "use_thermal": True,
            "use_multi_representation": False,
            "use_specialists": True,
            "use_adaptive_fusion": True,
            "use_symmetry_loss": False,
        },
        {
            "ablation_id": "A6",
            "name": "Without_anatomical_specialists",
            "use_rgb": True,
            "use_thermal": True,
            "use_multi_representation": True,
            "use_specialists": False,
            "use_adaptive_fusion": True,
            "use_symmetry_loss": False,
        },
        {
            "ablation_id": "A7",
            "name": "Without_adaptive_cooperative_fusion",
            "use_rgb": True,
            "use_thermal": True,
            "use_multi_representation": True,
            "use_specialists": True,
            "use_adaptive_fusion": False,
            "fusion_type": "concatenation",
            "use_symmetry_loss": False,
        },
        {
            "ablation_id": "A8",
            "name": "With_bilateral_symmetry_loss",
            "use_rgb": True,
            "use_thermal": True,
            "use_multi_representation": True,
            "use_specialists": True,
            "use_adaptive_fusion": True,
            "use_symmetry_loss": True,
            "symmetry_loss_weight": 0.05,
        },
        {
            "ablation_id": "A9",
            "name": "Full_CPMR_Net",
            "use_rgb": True,
            "use_thermal": True,
            "use_multi_representation": True,
            "use_specialists": True,
            "use_adaptive_fusion": True,
            "use_symmetry_loss": False,
        },
    ],
}


# ---------------------------------------------------------------------
# Save JSON config
# ---------------------------------------------------------------------

json_path = CONFIG_OUT / "CPMRNet_training_config_v1.json"

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=4, ensure_ascii=False)


# ---------------------------------------------------------------------
# Save flattened tables
# ---------------------------------------------------------------------

config_sections = []

for section_name, section_value in config.items():
    if isinstance(section_value, dict):
        for key, value in section_value.items():
            config_sections.append({
                "section": section_name,
                "key": key,
                "value": json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
            })
    elif isinstance(section_value, list):
        config_sections.append({
            "section": section_name,
            "key": "list_value",
            "value": json.dumps(section_value, ensure_ascii=False)
        })
    else:
        config_sections.append({
            "section": "root",
            "key": section_name,
            "value": section_value
        })

pd.DataFrame(config_sections).to_csv(TABLES_OUT / "training_config_flattened.csv", index=False)

pd.DataFrame(config["ablation_plan"]).to_csv(TABLES_OUT / "ablation_experiment_control_table.csv", index=False)

metric_rows = []
for metric_group, metrics in config["metrics"].items():
    if isinstance(metrics, list):
        for metric in metrics:
            metric_rows.append({
                "metric_group": metric_group,
                "metric": metric
            })
    else:
        metric_rows.append({
            "metric_group": metric_group,
            "metric": metrics
        })

pd.DataFrame(metric_rows).to_csv(TABLES_OUT / "training_metric_plan.csv", index=False)

optimization_df = pd.DataFrame([
    {"setting": k, "value": v}
    for k, v in config["optimization"].items()
])
optimization_df.to_csv(TABLES_OUT / "optimization_settings.csv", index=False)

model_df = pd.DataFrame([
    {"setting": k, "value": json.dumps(v) if isinstance(v, (dict, list)) else v}
    for k, v in config["model"].items()
])
model_df.to_csv(TABLES_OUT / "model_settings.csv", index=False)

augmentation_df = pd.DataFrame([
    {"setting": k, "value": json.dumps(v) if isinstance(v, (dict, list)) else v}
    for k, v in config["augmentation"].items()
])
augmentation_df.to_csv(TABLES_OUT / "augmentation_settings.csv", index=False)


# ---------------------------------------------------------------------
# Summary and report
# ---------------------------------------------------------------------

summary = {
    "stage": "Stage10I4",
    "title": "Training Configuration and Experiment Control File",
    "created_at": created_at,
    "experiment_id": experiment_id,
    "architecture": config["project"]["architecture"],
    "batch_size": config["data"]["batch_size"],
    "embedding_dim": config["model"]["embedding_dim"],
    "optimizer": config["optimization"]["optimizer"],
    "learning_rate": config["optimization"]["learning_rate"],
    "max_epochs": config["optimization"]["max_epochs"],
    "early_stopping_patience": config["optimization"]["early_stopping_patience"],
    "primary_metric": config["metrics"]["primary_selection_metric"],
    "ablation_experiments": len(config["ablation_plan"]),
    "config_file": str(json_path),
    "outputs_saved_to": str(STAGE_OUT),
}

with open(STAGE_OUT / "Stage10I4_Training_Configuration_Experiment_Control_Summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4, ensure_ascii=False)

report = []
report.append("# Stage 10I4 Training Configuration and Experiment Control File\n")
report.append(f"Generated at: {created_at}\n")
report.append("## Purpose\n")
report.append(
    "This stage creates the official experiment-control configuration for CPMR-Net training. "
    "It defines the dataset paths, split protocol, model settings, augmentation rules, optimization settings, "
    "metrics, checkpointing policy, and ablation experiment identifiers.\n"
)

report.append("## Main Configuration\n")
report.append(f"- Experiment ID: {experiment_id}")
report.append(f"- Architecture: {config['project']['architecture']}")
report.append(f"- Diagnostic unit: {config['project']['diagnostic_unit']}")
report.append(f"- Batch size: {config['data']['batch_size']}")
report.append(f"- Embedding dimension: {config['model']['embedding_dim']}")
report.append(f"- Optimizer: {config['optimization']['optimizer']}")
report.append(f"- Learning rate: {config['optimization']['learning_rate']}")
report.append(f"- Max epochs: {config['optimization']['max_epochs']}")
report.append(f"- Early stopping patience: {config['optimization']['early_stopping_patience']}")
report.append(f"- Primary selection metric: {config['metrics']['primary_selection_metric']}")
report.append(f"- Ablation experiments: {len(config['ablation_plan'])}\n")

report.append("## Official Config File\n")
report.append(f"- `{json_path}`\n")

report.append("## Output Files\n")
report.append("- `configs/CPMRNet_training_config_v1.json`")
report.append("- `tables/training_config_flattened.csv`")
report.append("- `tables/ablation_experiment_control_table.csv`")
report.append("- `tables/training_metric_plan.csv`")
report.append("- `tables/optimization_settings.csv`")
report.append("- `tables/model_settings.csv`")
report.append("- `tables/augmentation_settings.csv`")
report.append("- `Stage10I4_Training_Configuration_Experiment_Control_Summary.json`\n")

report.append("## Implementation Role\n")
report.append(
    "All subsequent CPMR-Net training, validation, ablation, and evaluation scripts should load this JSON configuration "
    "instead of hard-coding experimental settings."
)

with open(REPORTS_OUT / "Stage10I4_Training_Configuration_Experiment_Control_Report.md", "w", encoding="utf-8") as f:
    f.write("\n".join(report))

print("=" * 80)
print("STAGE 10I4 TRAINING CONFIGURATION AND EXPERIMENT CONTROL COMPLETED")
print("=" * 80)
print(f"Experiment ID: {experiment_id}")
print(f"Architecture: {config['project']['architecture']}")
print(f"Batch size: {config['data']['batch_size']}")
print(f"Embedding dim: {config['model']['embedding_dim']}")
print(f"Optimizer: {config['optimization']['optimizer']}")
print(f"Learning rate: {config['optimization']['learning_rate']}")
print(f"Max epochs: {config['optimization']['max_epochs']}")
print(f"Primary metric: {config['metrics']['primary_selection_metric']}")
print(f"Ablation experiments: {len(config['ablation_plan'])}")
print(f"Config saved to: {json_path}")
print(f"Outputs saved to: {STAGE_OUT}")
print("=" * 80)