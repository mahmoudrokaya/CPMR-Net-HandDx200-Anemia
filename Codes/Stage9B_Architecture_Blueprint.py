# -*- coding: utf-8 -*-
"""
Stage 9B - Architecture Blueprint

Converts Stage 9A architecture decisions into a concrete model blueprint:
inputs, representations, branches, specialist encoders, fusion modules,
losses, explainability outputs, and ablation plan.
"""

from pathlib import Path
import json
import pandas as pd
from datetime import datetime

BASE_DIR = Path(r"D:\47\472\New-Papers\Anemia_Paper")
OUTPUTS_DIR = BASE_DIR / "Outputs"

STAGE9A_DIR = OUTPUTS_DIR / "Stage9A_Architecture_Decision_Matrix"

STAGE_OUT = OUTPUTS_DIR / "Stage9B_Architecture_Blueprint"
TABLES_OUT = STAGE_OUT / "tables"
REPORTS_OUT = STAGE_OUT / "reports"

TABLES_OUT.mkdir(parents=True, exist_ok=True)
REPORTS_OUT.mkdir(parents=True, exist_ok=True)


def save_csv(rows, path):
    pd.DataFrame(rows).to_csv(path, index=False)


# ---------------------------------------------------------------------
# Architecture identity
# ---------------------------------------------------------------------

architecture_identity = {
    "architecture_name": "Cooperative Physiological Multi-Representation Network",
    "short_name": "CPMR-Net",
    "task": "Binary anemia classification",
    "diagnostic_unit": "Participant",
    "primary_modality": "RGB",
    "auxiliary_modality": "Thermal",
    "main_principle": (
        "Each participant is represented by multiple anatomical observations, "
        "each image generates multiple evidence representations, and final "
        "diagnosis emerges through adaptive cooperation among anatomical specialists."
    )
}


# ---------------------------------------------------------------------
# Input blueprint
# ---------------------------------------------------------------------

input_blueprint = [
    {
        "input_id": "I1",
        "input_name": "RGB left dorsal",
        "modality": "RGB",
        "view": "left_dorsal",
        "role": "Primary visual evidence"
    },
    {
        "input_id": "I2",
        "input_name": "RGB right dorsal",
        "modality": "RGB",
        "view": "right_dorsal",
        "role": "Primary visual evidence"
    },
    {
        "input_id": "I3",
        "input_name": "RGB left palmar",
        "modality": "RGB",
        "view": "left_palmar",
        "role": "Primary pallor/color evidence"
    },
    {
        "input_id": "I4",
        "input_name": "RGB right palmar",
        "modality": "RGB",
        "view": "right_palmar",
        "role": "Primary pallor/color evidence"
    },
    {
        "input_id": "I5",
        "input_name": "Thermal left dorsal",
        "modality": "Thermal",
        "view": "left_dorsal",
        "role": "Auxiliary perfusion evidence"
    },
    {
        "input_id": "I6",
        "input_name": "Thermal right dorsal",
        "modality": "Thermal",
        "view": "right_dorsal",
        "role": "Auxiliary perfusion evidence"
    },
    {
        "input_id": "I7",
        "input_name": "Thermal left palmar",
        "modality": "Thermal",
        "view": "left_palmar",
        "role": "Auxiliary perfusion evidence"
    },
    {
        "input_id": "I8",
        "input_name": "Thermal right palmar",
        "modality": "Thermal",
        "view": "right_palmar",
        "role": "Auxiliary perfusion evidence"
    }
]


# ---------------------------------------------------------------------
# Representation blueprint
# ---------------------------------------------------------------------

representation_blueprint = [
    {
        "representation_id": "R1",
        "representation": "Original RGB",
        "applies_to": "RGB",
        "purpose": "Preserve raw colorimetric and structural information.",
        "priority": "Core"
    },
    {
        "representation_id": "R2",
        "representation": "HSV representation",
        "applies_to": "RGB",
        "purpose": "Capture hue and saturation changes related to pallor.",
        "priority": "Core"
    },
    {
        "representation_id": "R3",
        "representation": "LAB representation",
        "applies_to": "RGB",
        "purpose": "Capture perceptual lightness and color-opponent channels.",
        "priority": "Core"
    },
    {
        "representation_id": "R4",
        "representation": "Local anatomical patches",
        "applies_to": "RGB and Thermal",
        "purpose": "Prioritize local evidence because handcrafted patches outperformed global descriptors.",
        "priority": "Core"
    },
    {
        "representation_id": "R5",
        "representation": "Texture-enhanced representation",
        "applies_to": "RGB and Thermal",
        "purpose": "Capture vascular, skin-surface, and thermal texture patterns.",
        "priority": "Core"
    },
    {
        "representation_id": "R6",
        "representation": "Thermal normalized intensity map",
        "applies_to": "Thermal",
        "purpose": "Capture relative spatial temperature/perfusion patterns.",
        "priority": "Core"
    },
    {
        "representation_id": "R7",
        "representation": "Saliency-guided or attention-guided crop",
        "applies_to": "RGB and Thermal",
        "purpose": "Focus learning on diagnostically informative regions.",
        "priority": "Secondary"
    },
    {
        "representation_id": "R8",
        "representation": "Frequency-domain representation",
        "applies_to": "RGB and Thermal",
        "purpose": "Explore texture and spatial-frequency information.",
        "priority": "Exploratory"
    }
]


# ---------------------------------------------------------------------
# Model modules
# ---------------------------------------------------------------------

module_blueprint = [
    {
        "module_id": "M1",
        "module_name": "Participant-level input container",
        "input": "Eight images per participant",
        "output": "Structured participant tensor/dictionary",
        "linked_novelty": "Participant-level multi-view anemia reasoning",
        "status": "Mandatory"
    },
    {
        "module_id": "M2",
        "module_name": "Multi-representation generator",
        "input": "Each RGB or thermal image",
        "output": "Multiple representation tensors per image",
        "linked_novelty": "Multi-representation anatomical evidence learning",
        "status": "Mandatory"
    },
    {
        "module_id": "M3",
        "module_name": "Representation encoders",
        "input": "Representation tensors",
        "output": "Representation embeddings",
        "linked_novelty": "Handcrafted-to-learned continuity",
        "status": "Mandatory"
    },
    {
        "module_id": "M4",
        "module_name": "Anatomical specialist encoders",
        "input": "Embeddings grouped by anatomical view",
        "output": "Specialist embeddings",
        "linked_novelty": "Adaptive cooperative specialist fusion",
        "status": "Mandatory"
    },
    {
        "module_id": "M5",
        "module_name": "RGB-centered branch",
        "input": "RGB specialist embeddings",
        "output": "RGB participant evidence vector",
        "linked_novelty": "RGB-centered multimodal physiological fusion",
        "status": "Mandatory"
    },
    {
        "module_id": "M6",
        "module_name": "Thermal auxiliary branch",
        "input": "Thermal specialist embeddings",
        "output": "Thermal auxiliary evidence vector",
        "linked_novelty": "RGB-centered multimodal physiological fusion",
        "status": "Mandatory but auxiliary"
    },
    {
        "module_id": "M7",
        "module_name": "Adaptive cooperative fusion",
        "input": "RGB and thermal specialist evidence vectors",
        "output": "Fused participant evidence vector plus contribution weights",
        "linked_novelty": "Adaptive cooperative specialist fusion and explainable evidence pathway",
        "status": "Mandatory"
    },
    {
        "module_id": "M8",
        "module_name": "Bilateral symmetry consistency module",
        "input": "Left-right paired embeddings",
        "output": "Symmetry regularization loss",
        "linked_novelty": "Bilateral physiological consistency",
        "status": "Secondary ablation"
    },
    {
        "module_id": "M9",
        "module_name": "Classification head",
        "input": "Fused participant evidence vector",
        "output": "Binary anemia probability",
        "linked_novelty": "Participant-level diagnosis",
        "status": "Mandatory"
    },
    {
        "module_id": "M10",
        "module_name": "Explainability output layer",
        "input": "Fusion weights and attention scores",
        "output": "Representation-level, view-level, and modality-level importance",
        "linked_novelty": "Explainable evidence pathway",
        "status": "Mandatory"
    }
]


# ---------------------------------------------------------------------
# Loss blueprint
# ---------------------------------------------------------------------

loss_blueprint = [
    {
        "loss_id": "L1",
        "loss_name": "Binary classification loss",
        "formula_type": "Weighted BCE or focal loss",
        "purpose": "Optimize anemia versus normal classification.",
        "priority": "Core"
    },
    {
        "loss_id": "L2",
        "loss_name": "Class imbalance handling",
        "formula_type": "Class weighting or focal modulation",
        "purpose": "Handle 65 anemia versus 133 normal participants.",
        "priority": "Core"
    },
    {
        "loss_id": "L3",
        "loss_name": "Bilateral symmetry consistency loss",
        "formula_type": "Embedding distance between left-right paired views",
        "purpose": "Encourage systemic consistency across bilateral hand observations.",
        "priority": "Secondary"
    },
    {
        "loss_id": "L4",
        "loss_name": "Fusion entropy or sparsity regularization",
        "formula_type": "Optional entropy control over attention/trust weights",
        "purpose": "Avoid unstable or meaningless specialist weighting.",
        "priority": "Exploratory"
    }
]


# ---------------------------------------------------------------------
# Output blueprint
# ---------------------------------------------------------------------

output_blueprint = [
    {
        "output_id": "O1",
        "output_name": "Anemia probability",
        "description": "Final participant-level probability of anemia.",
        "required": "Yes"
    },
    {
        "output_id": "O2",
        "output_name": "Predicted class",
        "description": "Binary class label: anemia or normal.",
        "required": "Yes"
    },
    {
        "output_id": "O3",
        "output_name": "Modality contribution weights",
        "description": "Relative contribution of RGB and thermal evidence.",
        "required": "Yes"
    },
    {
        "output_id": "O4",
        "output_name": "View contribution weights",
        "description": "Relative contribution of left/right dorsal/palmar specialists.",
        "required": "Yes"
    },
    {
        "output_id": "O5",
        "output_name": "Representation contribution weights",
        "description": "Relative contribution of RGB, HSV, LAB, patches, texture, and thermal maps.",
        "required": "Yes"
    },
    {
        "output_id": "O6",
        "output_name": "Symmetry consistency score",
        "description": "Participant-level bilateral consistency score.",
        "required": "For symmetry ablation"
    }
]


# ---------------------------------------------------------------------
# Ablation blueprint
# ---------------------------------------------------------------------

ablation_blueprint = [
    {
        "ablation_id": "A0",
        "experiment": "Handcrafted baseline",
        "description": "Use S6D6B nonredundant handcrafted benchmark as reference.",
        "purpose": "Confirm improvement beyond ROC-AUC 0.7447 baseline."
    },
    {
        "ablation_id": "A1",
        "experiment": "RGB-only participant model",
        "description": "Use RGB views only.",
        "purpose": "Test primary signal branch."
    },
    {
        "ablation_id": "A2",
        "experiment": "Thermal-only participant model",
        "description": "Use thermal views only.",
        "purpose": "Confirm weak standalone thermal contribution."
    },
    {
        "ablation_id": "A3",
        "experiment": "RGB + thermal equal fusion",
        "description": "Fuse RGB and thermal without adaptive weighting.",
        "purpose": "Baseline for multimodal fusion."
    },
    {
        "ablation_id": "A4",
        "experiment": "RGB-centered auxiliary thermal fusion",
        "description": "RGB-dominant branch with thermal auxiliary evidence.",
        "purpose": "Test core multimodal design."
    },
    {
        "ablation_id": "A5",
        "experiment": "Without multi-representation generator",
        "description": "Use only original images.",
        "purpose": "Measure value of multi-representation learning."
    },
    {
        "ablation_id": "A6",
        "experiment": "Without anatomical specialists",
        "description": "Pool all view embeddings directly.",
        "purpose": "Measure value of specialist organization."
    },
    {
        "ablation_id": "A7",
        "experiment": "Without adaptive cooperative fusion",
        "description": "Replace adaptive fusion with concatenation.",
        "purpose": "Measure value of cooperative fusion."
    },
    {
        "ablation_id": "A8",
        "experiment": "With bilateral symmetry loss",
        "description": "Add left-right consistency regularization.",
        "purpose": "Test secondary physiological consistency contribution."
    },
    {
        "ablation_id": "A9",
        "experiment": "Full CPMR-Net",
        "description": "All core modules plus selected secondary regularization.",
        "purpose": "Final proposed framework."
    }
]


# ---------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------

save_csv([architecture_identity], TABLES_OUT / "architecture_identity.csv")
save_csv(input_blueprint, TABLES_OUT / "input_blueprint.csv")
save_csv(representation_blueprint, TABLES_OUT / "representation_blueprint.csv")
save_csv(module_blueprint, TABLES_OUT / "module_blueprint.csv")
save_csv(loss_blueprint, TABLES_OUT / "loss_blueprint.csv")
save_csv(output_blueprint, TABLES_OUT / "output_blueprint.csv")
save_csv(ablation_blueprint, TABLES_OUT / "ablation_blueprint.csv")


knowledge_base = {
    "stage": "Stage9B",
    "title": "Architecture Blueprint",
    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "architecture_identity": architecture_identity,
    "input_blueprint": input_blueprint,
    "representation_blueprint": representation_blueprint,
    "module_blueprint": module_blueprint,
    "loss_blueprint": loss_blueprint,
    "output_blueprint": output_blueprint,
    "ablation_blueprint": ablation_blueprint
}

with open(STAGE_OUT / "Stage9B_Architecture_Blueprint.json", "w", encoding="utf-8") as f:
    json.dump(knowledge_base, f, indent=4, ensure_ascii=False)


report = []
report.append("# Stage 9B Architecture Blueprint\n")
report.append(f"Generated at: {knowledge_base['created_at']}\n")

report.append("## Architecture Identity\n")
report.append(f"**Name:** {architecture_identity['architecture_name']} ({architecture_identity['short_name']})")
report.append(f"**Task:** {architecture_identity['task']}")
report.append(f"**Diagnostic unit:** {architecture_identity['diagnostic_unit']}")
report.append(f"**Primary modality:** {architecture_identity['primary_modality']}")
report.append(f"**Auxiliary modality:** {architecture_identity['auxiliary_modality']}")
report.append(f"**Main principle:** {architecture_identity['main_principle']}\n")

report.append("## Input Blueprint\n")
for x in input_blueprint:
    report.append(f"- **{x['input_name']}**: {x['role']}")

report.append("\n## Representation Blueprint\n")
for x in representation_blueprint:
    report.append(f"- **{x['representation']}** ({x['priority']}): {x['purpose']}")

report.append("\n## Model Module Blueprint\n")
for x in module_blueprint:
    report.append(f"### {x['module_id']}. {x['module_name']}")
    report.append(f"- Input: {x['input']}")
    report.append(f"- Output: {x['output']}")
    report.append(f"- Linked novelty: {x['linked_novelty']}")
    report.append(f"- Status: {x['status']}\n")

report.append("## Loss Blueprint\n")
for x in loss_blueprint:
    report.append(f"- **{x['loss_name']}** ({x['priority']}): {x['purpose']}")

report.append("\n## Output Blueprint\n")
for x in output_blueprint:
    report.append(f"- **{x['output_name']}**: {x['description']}")

report.append("\n## Ablation Blueprint\n")
for x in ablation_blueprint:
    report.append(f"### {x['ablation_id']}. {x['experiment']}")
    report.append(f"- Description: {x['description']}")
    report.append(f"- Purpose: {x['purpose']}\n")

report.append("## Generated Output Files\n")
report.append("- `Stage9B_Architecture_Blueprint.json`")
report.append("- `tables/architecture_identity.csv`")
report.append("- `tables/input_blueprint.csv`")
report.append("- `tables/representation_blueprint.csv`")
report.append("- `tables/module_blueprint.csv`")
report.append("- `tables/loss_blueprint.csv`")
report.append("- `tables/output_blueprint.csv`")
report.append("- `tables/ablation_blueprint.csv`")

with open(REPORTS_OUT / "Stage9B_Architecture_Blueprint_Report.md", "w", encoding="utf-8") as f:
    f.write("\n".join(report))


print("=" * 80)
print("STAGE 9B ARCHITECTURE BLUEPRINT COMPLETED")
print("=" * 80)
print(f"Architecture: {architecture_identity['architecture_name']} ({architecture_identity['short_name']})")
print(f"Inputs: {len(input_blueprint)}")
print(f"Representations: {len(representation_blueprint)}")
print(f"Modules: {len(module_blueprint)}")
print(f"Losses: {len(loss_blueprint)}")
print(f"Outputs: {len(output_blueprint)}")
print(f"Ablations: {len(ablation_blueprint)}")
print(f"Outputs saved to: {STAGE_OUT}")
print("=" * 80)