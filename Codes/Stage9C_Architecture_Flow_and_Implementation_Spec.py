# -*- coding: utf-8 -*-
"""
Stage 9C - Architecture Flow Diagram and Implementation Specification

Purpose:
Convert Stage 9B CPMR-Net blueprint into an implementation-ready specification:
- architecture flow
- module interfaces
- tensor/data contracts
- diagram edges
- implementation stages
"""

from pathlib import Path
import json
import pandas as pd
from datetime import datetime

BASE_DIR = Path(r"D:\47\472\New-Papers\Anemia_Paper")
OUTPUTS_DIR = BASE_DIR / "Outputs"

STAGE_OUT = OUTPUTS_DIR / "Stage9C_Architecture_Flow_and_Implementation_Spec"
TABLES_OUT = STAGE_OUT / "tables"
REPORTS_OUT = STAGE_OUT / "reports"

TABLES_OUT.mkdir(parents=True, exist_ok=True)
REPORTS_OUT.mkdir(parents=True, exist_ok=True)


def save_csv(rows, path):
    pd.DataFrame(rows).to_csv(path, index=False)


architecture_flow = [
    ["F1", "Participant folder / metadata", "Participant-level input container"],
    ["F2", "Participant-level input container", "Eight-view image loader"],
    ["F3", "Eight-view image loader", "Multi-representation generator"],
    ["F4", "Multi-representation generator", "Representation encoders"],
    ["F5", "Representation encoders", "Anatomical specialist encoders"],
    ["F6", "Anatomical specialist encoders", "RGB-centered branch"],
    ["F7", "Anatomical specialist encoders", "Thermal auxiliary branch"],
    ["F8", "RGB-centered branch", "Adaptive cooperative fusion"],
    ["F9", "Thermal auxiliary branch", "Adaptive cooperative fusion"],
    ["F10", "Adaptive cooperative fusion", "Classification head"],
    ["F11", "Adaptive cooperative fusion", "Explainability output layer"],
    ["F12", "Anatomical specialist encoders", "Bilateral symmetry consistency module"],
    ["F13", "Classification head", "Anemia probability"],
    ["F14", "Explainability output layer", "Contribution reports"],
]

flow_rows = [
    {"edge_id": e[0], "source": e[1], "target": e[2]}
    for e in architecture_flow
]

module_specs = [
    {
        "module_id": "M1",
        "module_name": "ParticipantLevelDataset",
        "implementation_type": "PyTorch Dataset",
        "input_contract": "participant_dataset.csv with participant_id, label, sex, age, and eight image paths",
        "output_contract": "dict with participant_id, label, rgb views, thermal views, metadata",
        "notes": "Must split by participant only."
    },
    {
        "module_id": "M2",
        "module_name": "MultiRepresentationGenerator",
        "implementation_type": "Preprocessing / transform module",
        "input_contract": "Single RGB or thermal image tensor",
        "output_contract": "dictionary of representation tensors",
        "notes": "Core representations: RGB, HSV, LAB, patches, texture, thermal map."
    },
    {
        "module_id": "M3",
        "module_name": "RepresentationEncoder",
        "implementation_type": "Lightweight CNN / pretrained backbone",
        "input_contract": "representation tensor",
        "output_contract": "fixed-length representation embedding",
        "notes": "Shared or modality-specific encoders can be tested."
    },
    {
        "module_id": "M4",
        "module_name": "AnatomicalSpecialistEncoder",
        "implementation_type": "Embedding aggregator + attention",
        "input_contract": "all representation embeddings for one anatomical view",
        "output_contract": "one specialist embedding per view",
        "notes": "Produces interpretable representation-level weights."
    },
    {
        "module_id": "M5",
        "module_name": "RGBEvidenceBranch",
        "implementation_type": "View-level fusion branch",
        "input_contract": "four RGB specialist embeddings",
        "output_contract": "RGB participant evidence vector",
        "notes": "Primary branch."
    },
    {
        "module_id": "M6",
        "module_name": "ThermalEvidenceBranch",
        "implementation_type": "View-level fusion branch",
        "input_contract": "four thermal specialist embeddings",
        "output_contract": "thermal auxiliary evidence vector",
        "notes": "Auxiliary branch."
    },
    {
        "module_id": "M7",
        "module_name": "AdaptiveCooperativeFusion",
        "implementation_type": "Gating / attention fusion",
        "input_contract": "RGB evidence vector + thermal evidence vector",
        "output_contract": "fused participant vector + modality/view weights",
        "notes": "Avoid naive concatenation."
    },
    {
        "module_id": "M8",
        "module_name": "SymmetryConsistencyLoss",
        "implementation_type": "Auxiliary loss",
        "input_contract": "left/right paired specialist embeddings",
        "output_contract": "scalar symmetry loss",
        "notes": "Secondary ablation only."
    },
    {
        "module_id": "M9",
        "module_name": "ClassificationHead",
        "implementation_type": "MLP classifier",
        "input_contract": "fused participant evidence vector",
        "output_contract": "binary anemia probability",
        "notes": "Use weighted BCE or focal loss."
    },
    {
        "module_id": "M10",
        "module_name": "ExplainabilityReporter",
        "implementation_type": "Post-processing logger",
        "input_contract": "attention/fusion weights",
        "output_contract": "CSV tables of modality, view, and representation contributions",
        "notes": "Required for interpretability."
    }
]

tensor_contracts = [
    {
        "object": "RGB image",
        "suggested_shape": "[3, H, W]",
        "source": "RGB PNG",
        "consumer": "MultiRepresentationGenerator"
    },
    {
        "object": "Thermal image",
        "suggested_shape": "[1, H, W]",
        "source": "decoded thermal PNG",
        "consumer": "MultiRepresentationGenerator"
    },
    {
        "object": "Representation tensor",
        "suggested_shape": "[C, H, W] or patch batch [P, C, h, w]",
        "source": "MultiRepresentationGenerator",
        "consumer": "RepresentationEncoder"
    },
    {
        "object": "Representation embedding",
        "suggested_shape": "[D]",
        "source": "RepresentationEncoder",
        "consumer": "AnatomicalSpecialistEncoder"
    },
    {
        "object": "Specialist embedding",
        "suggested_shape": "[D]",
        "source": "AnatomicalSpecialistEncoder",
        "consumer": "RGBEvidenceBranch / ThermalEvidenceBranch"
    },
    {
        "object": "Modality evidence vector",
        "suggested_shape": "[D]",
        "source": "RGBEvidenceBranch / ThermalEvidenceBranch",
        "consumer": "AdaptiveCooperativeFusion"
    },
    {
        "object": "Fused participant vector",
        "suggested_shape": "[D]",
        "source": "AdaptiveCooperativeFusion",
        "consumer": "ClassificationHead"
    }
]

implementation_steps = [
    ["S9C-1", "Build participant-level dataset loader", "Required before any model training"],
    ["S9C-2", "Implement multi-representation generator", "Start with core representations only"],
    ["S9C-3", "Implement lightweight representation encoders", "Use small backbones due to n=198"],
    ["S9C-4", "Implement anatomical specialist aggregation", "One specialist per modality-view pair"],
    ["S9C-5", "Implement RGB and thermal evidence branches", "RGB primary, thermal auxiliary"],
    ["S9C-6", "Implement adaptive cooperative fusion", "Return both fused vector and weights"],
    ["S9C-7", "Implement classification head", "Weighted BCE or focal loss"],
    ["S9C-8", "Implement explainability logging", "Save weights per participant"],
    ["S9C-9", "Add optional symmetry loss", "Use only as ablation"],
    ["S9C-10", "Run ablation sequence A1-A9", "Compare against handcrafted baseline"],
]

implementation_rows = [
    {"step_id": x[0], "step": x[1], "note": x[2]}
    for x in implementation_steps
]

save_csv(flow_rows, TABLES_OUT / "architecture_flow_edges.csv")
save_csv(module_specs, TABLES_OUT / "module_implementation_specs.csv")
save_csv(tensor_contracts, TABLES_OUT / "tensor_data_contracts.csv")
save_csv(implementation_rows, TABLES_OUT / "implementation_sequence.csv")

knowledge_base = {
    "stage": "Stage9C",
    "title": "Architecture Flow Diagram and Implementation Specification",
    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "architecture_name": "Cooperative Physiological Multi-Representation Network",
    "short_name": "CPMR-Net",
    "architecture_flow_edges": flow_rows,
    "module_implementation_specs": module_specs,
    "tensor_data_contracts": tensor_contracts,
    "implementation_sequence": implementation_rows
}

with open(STAGE_OUT / "Stage9C_Architecture_Flow_and_Implementation_Spec.json", "w", encoding="utf-8") as f:
    json.dump(knowledge_base, f, indent=4, ensure_ascii=False)

report = []
report.append("# Stage 9C Architecture Flow Diagram and Implementation Specification\n")
report.append(f"Generated at: {knowledge_base['created_at']}\n")
report.append("## Architecture\n")
report.append("**Cooperative Physiological Multi-Representation Network (CPMR-Net)**\n")

report.append("## Architecture Flow\n")
for row in flow_rows:
    report.append(f"- {row['edge_id']}: {row['source']} → {row['target']}")

report.append("\n## Module Implementation Specifications\n")
for row in module_specs:
    report.append(f"### {row['module_id']}. {row['module_name']}")
    report.append(f"- Type: {row['implementation_type']}")
    report.append(f"- Input: {row['input_contract']}")
    report.append(f"- Output: {row['output_contract']}")
    report.append(f"- Notes: {row['notes']}\n")

report.append("## Tensor/Data Contracts\n")
for row in tensor_contracts:
    report.append(f"- **{row['object']}**: {row['suggested_shape']} | {row['source']} → {row['consumer']}")

report.append("\n## Implementation Sequence\n")
for row in implementation_rows:
    report.append(f"- **{row['step_id']}**: {row['step']} — {row['note']}")

report.append("\n## Generated Output Files\n")
report.append("- `Stage9C_Architecture_Flow_and_Implementation_Spec.json`")
report.append("- `tables/architecture_flow_edges.csv`")
report.append("- `tables/module_implementation_specs.csv`")
report.append("- `tables/tensor_data_contracts.csv`")
report.append("- `tables/implementation_sequence.csv`")

with open(REPORTS_OUT / "Stage9C_Architecture_Flow_and_Implementation_Spec_Report.md", "w", encoding="utf-8") as f:
    f.write("\n".join(report))

print("=" * 80)
print("STAGE 9C ARCHITECTURE FLOW AND IMPLEMENTATION SPEC COMPLETED")
print("=" * 80)
print(f"Flow edges: {len(flow_rows)}")
print(f"Module specs: {len(module_specs)}")
print(f"Tensor contracts: {len(tensor_contracts)}")
print(f"Implementation steps: {len(implementation_rows)}")
print(f"Outputs saved to: {STAGE_OUT}")
print("=" * 80)