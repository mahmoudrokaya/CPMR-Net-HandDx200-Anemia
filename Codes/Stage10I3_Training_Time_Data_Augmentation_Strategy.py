# -*- coding: utf-8 -*-
"""
Stage 10I3 - Training-Time Data Augmentation Strategy

Purpose:
Define and validate safe augmentation rules for CPMR-Net training.

No model training is performed.
"""

from pathlib import Path
import json
from datetime import datetime
import pandas as pd

BASE_DIR = Path(r"D:\47\472\New-Papers\Anemia_Paper")
OUTPUTS_DIR = BASE_DIR / "Outputs"

STAGE10I2_DIR = OUTPUTS_DIR / "Stage10I2_PyTorch_Participant_Dataset_Dataloader"
STAGE_OUT = OUTPUTS_DIR / "Stage10I3_Training_Time_Data_Augmentation_Strategy"
TABLES_OUT = STAGE_OUT / "tables"
REPORTS_OUT = STAGE_OUT / "reports"

TABLES_OUT.mkdir(parents=True, exist_ok=True)
REPORTS_OUT.mkdir(parents=True, exist_ok=True)

augmentation_rules = [
    {
        "rule_id": "AUG1",
        "augmentation": "Horizontal flip",
        "applies_to": "RGB and thermal",
        "probability": 0.25,
        "strength": "Mild",
        "allowed_in_train": True,
        "allowed_in_val_test": False,
        "rationale": "Improves robustness while preserving global hand structure.",
        "caution": "Must not break left-right view labels if anatomical laterality is explicitly modeled."
    },
    {
        "rule_id": "AUG2",
        "augmentation": "Small rotation",
        "applies_to": "RGB and thermal",
        "probability": 0.30,
        "strength": "±7 degrees",
        "allowed_in_train": True,
        "allowed_in_val_test": False,
        "rationale": "Accounts for minor acquisition-angle variation.",
        "caution": "Avoid large rotations that distort anatomical orientation."
    },
    {
        "rule_id": "AUG3",
        "augmentation": "Small translation/scale",
        "applies_to": "RGB and thermal",
        "probability": 0.30,
        "strength": "scale 0.95–1.05, translation ≤5%",
        "allowed_in_train": True,
        "allowed_in_val_test": False,
        "rationale": "Improves robustness to hand positioning.",
        "caution": "Avoid cropping diagnostically relevant palm or dorsal regions."
    },
    {
        "rule_id": "AUG4",
        "augmentation": "Mild brightness/contrast jitter",
        "applies_to": "RGB only",
        "probability": 0.20,
        "strength": "brightness/contrast ±8%",
        "allowed_in_train": True,
        "allowed_in_val_test": False,
        "rationale": "Accounts for illumination variation.",
        "caution": "Must remain mild because pallor/color is diagnostically important."
    },
    {
        "rule_id": "AUG5",
        "augmentation": "Mild Gaussian noise",
        "applies_to": "RGB and thermal",
        "probability": 0.15,
        "strength": "low variance",
        "allowed_in_train": True,
        "allowed_in_val_test": False,
        "rationale": "Improves robustness to sensor noise.",
        "caution": "Avoid noise that destroys texture or thermal gradients."
    },
    {
        "rule_id": "AUG6",
        "augmentation": "Thermal intensity jitter",
        "applies_to": "Thermal only",
        "probability": 0.15,
        "strength": "±5%",
        "allowed_in_train": True,
        "allowed_in_val_test": False,
        "rationale": "Accounts for mild thermal normalization variability.",
        "caution": "Must not invert or heavily distort relative temperature patterns."
    },
    {
        "rule_id": "AUG7",
        "augmentation": "Color hue shift",
        "applies_to": "RGB",
        "probability": 0.0,
        "strength": "Disabled",
        "allowed_in_train": False,
        "allowed_in_val_test": False,
        "rationale": "Hue is a key anemia-related signal from handcrafted analysis.",
        "caution": "Do not use."
    },
    {
        "rule_id": "AUG8",
        "augmentation": "Strong saturation jitter",
        "applies_to": "RGB",
        "probability": 0.0,
        "strength": "Disabled",
        "allowed_in_train": False,
        "allowed_in_val_test": False,
        "rationale": "May corrupt pallor-related information.",
        "caution": "Do not use."
    },
    {
        "rule_id": "AUG9",
        "augmentation": "Cutout/random erasing",
        "applies_to": "RGB and thermal",
        "probability": 0.0,
        "strength": "Disabled",
        "allowed_in_train": False,
        "allowed_in_val_test": False,
        "rationale": "May remove localized anatomical evidence that is central to CPMR-Net.",
        "caution": "Do not use in the first implementation."
    }
]

augmentation_df = pd.DataFrame(augmentation_rules)
augmentation_df.to_csv(TABLES_OUT / "augmentation_strategy_rules.csv", index=False)

training_policy = {
    "stage": "Stage10I3",
    "title": "Training-Time Data Augmentation Strategy",
    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "policy": "Use mild train-only augmentation. No augmentation in validation or test.",
    "core_principle": (
        "Augmentation must improve robustness without corrupting anemia-related pallor, hue, "
        "local anatomical texture, or thermal intensity cues."
    ),
    "recommended_train_augmentations": [
        "small rotation",
        "small translation/scale",
        "mild RGB brightness/contrast jitter",
        "mild Gaussian noise",
        "mild thermal intensity jitter"
    ],
    "disabled_augmentations": [
        "hue shift",
        "strong saturation jitter",
        "cutout/random erasing",
        "large geometric distortions"
    ],
    "validation_test_policy": "No stochastic augmentation; deterministic resizing/loading only.",
    "outputs_saved_to": str(STAGE_OUT)
}

with open(STAGE_OUT / "Stage10I3_Training_Time_Data_Augmentation_Strategy_Summary.json", "w", encoding="utf-8") as f:
    json.dump(training_policy, f, indent=4, ensure_ascii=False)

report = []
report.append("# Stage 10I3 Training-Time Data Augmentation Strategy\n")
report.append(f"Generated at: {training_policy['created_at']}\n")
report.append("## Purpose\n")
report.append(
    "This stage defines safe augmentation rules for CPMR-Net supervised training. "
    "The strategy is intentionally conservative because color, pallor, local texture, and thermal intensity are central diagnostic cues.\n"
)
report.append("## Core Policy\n")
report.append(training_policy["policy"] + "\n")
report.append("## Recommended Augmentations\n")
for item in training_policy["recommended_train_augmentations"]:
    report.append(f"- {item}")
report.append("\n## Disabled Augmentations\n")
for item in training_policy["disabled_augmentations"]:
    report.append(f"- {item}")
report.append("\n## Output Files\n")
report.append("- `augmentation_strategy_rules.csv`")
report.append("- `Stage10I3_Training_Time_Data_Augmentation_Strategy_Summary.json`\n")
report.append("## Implementation Role\n")
report.append(
    "These rules should be implemented in the training dataset only. "
    "Validation and test dataloaders must remain deterministic."
)

with open(REPORTS_OUT / "Stage10I3_Training_Time_Data_Augmentation_Strategy_Report.md", "w", encoding="utf-8") as f:
    f.write("\n".join(report))

print("=" * 80)
print("STAGE 10I3 TRAINING-TIME DATA AUGMENTATION STRATEGY COMPLETED")
print("=" * 80)
print("Policy: mild train-only augmentation; no validation/test augmentation.")
print(f"Rules saved: {len(augmentation_rules)}")
print(f"Outputs saved to: {STAGE_OUT}")
print("=" * 80)