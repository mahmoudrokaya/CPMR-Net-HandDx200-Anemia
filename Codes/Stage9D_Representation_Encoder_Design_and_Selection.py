# -*- coding: utf-8 -*-
"""
Stage 9D - Representation Encoder Design and Selection

Purpose:
Select the most appropriate representation encoder strategy for CPMR-Net before
implementing Stage 10D.

This stage evaluates:
- backbone architecture
- weight sharing
- embedding design
- regularization
- computational feasibility
- publishability

No model training is performed.
"""

from pathlib import Path
import json
import pandas as pd
from datetime import datetime

BASE_DIR = Path(r"D:\47\472\New-Papers\Anemia_Paper")
OUTPUTS_DIR = BASE_DIR / "Outputs"

STAGE_OUT = OUTPUTS_DIR / "Stage9D_Representation_Encoder_Design_and_Selection"
TABLES_OUT = STAGE_OUT / "tables"
REPORTS_OUT = STAGE_OUT / "reports"

TABLES_OUT.mkdir(parents=True, exist_ok=True)
REPORTS_OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Candidate encoder families
# ---------------------------------------------------------------------

encoder_candidates = [
    {
        "encoder_id": "E1",
        "encoder_family": "Custom lightweight CNN",
        "description": "Small CNN trained from scratch for 224x224 full images and 112x112 patches.",
        "parameter_risk": 2,
        "overfitting_risk": 2,
        "small_dataset_suitability": 5,
        "computational_efficiency": 5,
        "interpretability": 4,
        "implementation_complexity": 2,
        "publishability": 4,
        "recommended_role": "Primary recommended encoder"
    },
    {
        "encoder_id": "E2",
        "encoder_family": "ResNet-18 pretrained",
        "description": "ImageNet-pretrained ResNet-18 with frozen or partially fine-tuned layers.",
        "parameter_risk": 3,
        "overfitting_risk": 3,
        "small_dataset_suitability": 4,
        "computational_efficiency": 4,
        "interpretability": 4,
        "implementation_complexity": 3,
        "publishability": 4,
        "recommended_role": "Strong baseline encoder"
    },
    {
        "encoder_id": "E3",
        "encoder_family": "MobileNetV3-Small pretrained",
        "description": "Efficient pretrained lightweight CNN suitable for low-resource deployment.",
        "parameter_risk": 2,
        "overfitting_risk": 3,
        "small_dataset_suitability": 4,
        "computational_efficiency": 5,
        "interpretability": 3,
        "implementation_complexity": 3,
        "publishability": 4,
        "recommended_role": "Efficiency-oriented baseline"
    },
    {
        "encoder_id": "E4",
        "encoder_family": "EfficientNet-B0 pretrained",
        "description": "Compact pretrained CNN with strong feature extraction capacity.",
        "parameter_risk": 4,
        "overfitting_risk": 4,
        "small_dataset_suitability": 3,
        "computational_efficiency": 3,
        "interpretability": 3,
        "implementation_complexity": 3,
        "publishability": 4,
        "recommended_role": "Secondary comparison only"
    },
    {
        "encoder_id": "E5",
        "encoder_family": "ViT-Tiny / DeiT-Tiny",
        "description": "Transformer-based image encoder.",
        "parameter_risk": 5,
        "overfitting_risk": 5,
        "small_dataset_suitability": 2,
        "computational_efficiency": 2,
        "interpretability": 3,
        "implementation_complexity": 4,
        "publishability": 4,
        "recommended_role": "Not recommended for first implementation"
    },
    {
        "encoder_id": "E6",
        "encoder_family": "Hybrid CNN-Transformer",
        "description": "CNN encoder followed by lightweight transformer/attention pooling.",
        "parameter_risk": 4,
        "overfitting_risk": 4,
        "small_dataset_suitability": 3,
        "computational_efficiency": 3,
        "interpretability": 4,
        "implementation_complexity": 5,
        "publishability": 5,
        "recommended_role": "Future extension"
    }
]

enc_df = pd.DataFrame(encoder_candidates)

# Higher is better for positive criteria; lower is better for risk/complexity.
enc_df["selection_score"] = (
    enc_df["small_dataset_suitability"] * 2
    + enc_df["computational_efficiency"]
    + enc_df["interpretability"]
    + enc_df["publishability"]
    - enc_df["parameter_risk"]
    - enc_df["overfitting_risk"]
    - enc_df["implementation_complexity"] * 0.5
).round(2)

enc_df = enc_df.sort_values("selection_score", ascending=False)
enc_df.to_csv(TABLES_OUT / "encoder_family_selection_matrix.csv", index=False)


# ---------------------------------------------------------------------
# Weight-sharing strategies
# ---------------------------------------------------------------------

weight_sharing = [
    {
        "strategy_id": "W1",
        "strategy": "Single shared encoder for all representations",
        "advantage": "Lowest parameter count.",
        "disadvantage": "Ignores modality and representation differences.",
        "risk": "Medium",
        "decision": "Reject"
    },
    {
        "strategy_id": "W2",
        "strategy": "Separate encoder for each representation type",
        "advantage": "Maximum specialization.",
        "disadvantage": "Too many parameters for 198 participants.",
        "risk": "High",
        "decision": "Reject for first implementation"
    },
    {
        "strategy_id": "W3",
        "strategy": "Shared RGB encoder and shared thermal encoder",
        "advantage": "Balances parameter efficiency and modality specialization.",
        "disadvantage": "Representation-specific variation may be under-modeled.",
        "risk": "Low",
        "decision": "Recommended"
    },
    {
        "strategy_id": "W4",
        "strategy": "Shared backbone with representation-specific projection heads",
        "advantage": "Efficient and allows representation adaptation.",
        "disadvantage": "Slightly more complex.",
        "risk": "Low-Medium",
        "decision": "Recommended advanced option"
    },
    {
        "strategy_id": "W5",
        "strategy": "Independent RGB, HSV, LAB, texture, patch, and thermal encoders",
        "advantage": "Strong specialization.",
        "disadvantage": "High overfitting risk and high computational cost.",
        "risk": "High",
        "decision": "Not recommended"
    }
]

weight_df = pd.DataFrame(weight_sharing)
weight_df.to_csv(TABLES_OUT / "weight_sharing_strategy_matrix.csv", index=False)


# ---------------------------------------------------------------------
# Embedding design
# ---------------------------------------------------------------------

embedding_design = [
    {
        "design_id": "D1",
        "component": "Embedding dimension",
        "recommendation": "128",
        "reason": "Small enough for n=198, sufficient for multimodal representation learning."
    },
    {
        "design_id": "D2",
        "component": "Projection head",
        "recommendation": "Linear -> ReLU -> Dropout -> Linear",
        "reason": "Provides controlled adaptation without large parameter growth."
    },
    {
        "design_id": "D3",
        "component": "Normalization",
        "recommendation": "LayerNorm after projection",
        "reason": "Stabilizes embeddings before specialist aggregation."
    },
    {
        "design_id": "D4",
        "component": "Pooling",
        "recommendation": "Global average pooling for CNN feature maps",
        "reason": "Parameter-efficient and robust for small datasets."
    },
    {
        "design_id": "D5",
        "component": "Patch handling",
        "recommendation": "Encode each patch independently, then average/attention-pool patch embeddings",
        "reason": "Preserves local anatomical evidence while avoiding large concatenation."
    }
]

embed_df = pd.DataFrame(embedding_design)
embed_df.to_csv(TABLES_OUT / "embedding_design_recommendations.csv", index=False)


# ---------------------------------------------------------------------
# Regularization strategy
# ---------------------------------------------------------------------

regularization = [
    {
        "regularization_id": "R1",
        "method": "Data augmentation",
        "recommendation": "Mild geometric and photometric augmentation",
        "reason": "Improves generalization without destroying pallor/color signals."
    },
    {
        "regularization_id": "R2",
        "method": "Dropout",
        "recommendation": "0.20 to 0.40 in projection and fusion layers",
        "reason": "Controls overfitting in the participant-level model."
    },
    {
        "regularization_id": "R3",
        "method": "Weight decay",
        "recommendation": "1e-4 to 5e-4",
        "reason": "Stabilizes encoder and classifier parameters."
    },
    {
        "regularization_id": "R4",
        "method": "Frozen or partially frozen encoder",
        "recommendation": "Use frozen pretrained encoders for baseline; fine-tune only upper layers if needed.",
        "reason": "Reduces overfitting risk."
    },
    {
        "regularization_id": "R5",
        "method": "Early stopping",
        "recommendation": "Monitor validation ROC-AUC and F1",
        "reason": "Prevents overtraining on only 198 participants."
    },
    {
        "regularization_id": "R6",
        "method": "Class weighting or focal loss",
        "recommendation": "Required",
        "reason": "Dataset is imbalanced: 65 anemia and 133 normal participants."
    },
    {
        "regularization_id": "R7",
        "method": "Symmetry loss",
        "recommendation": "Use only in ablation after stable baseline",
        "reason": "Scientifically meaningful but should not destabilize first implementation."
    }
]

reg_df = pd.DataFrame(regularization)
reg_df.to_csv(TABLES_OUT / "regularization_strategy.csv", index=False)


# ---------------------------------------------------------------------
# Computational feasibility
# ---------------------------------------------------------------------

computational_plan = [
    {
        "item": "Dataset size",
        "value": "198 participants, 12,672 validated representations",
        "implication": "The bottleneck is participant count, not representation count."
    },
    {
        "item": "Encoder risk",
        "value": "High-capacity models may overfit",
        "implication": "Prefer lightweight CNNs or frozen pretrained encoders."
    },
    {
        "item": "Batching strategy",
        "value": "Participant-level batching",
        "implication": "Each batch item contains multiple images/representations."
    },
    {
        "item": "Preferred image size",
        "value": "224x224 full images, 112x112 patches",
        "implication": "Compatible with lightweight CNN encoders."
    },
    {
        "item": "Embedding dimension",
        "value": "128",
        "implication": "Balances compactness and representation capacity."
    },
    {
        "item": "Deployment direction",
        "value": "Lightweight and interpretable",
        "implication": "Avoid large ViT or heavy ensemble encoders in first paper."
    }
]

comp_df = pd.DataFrame(computational_plan)
comp_df.to_csv(TABLES_OUT / "computational_feasibility_plan.csv", index=False)


# ---------------------------------------------------------------------
# Final recommendation
# ---------------------------------------------------------------------

final_recommendation = {
    "recommended_encoder_family": "Custom lightweight CNN as primary encoder, with ResNet-18 or MobileNetV3-Small as comparative pretrained baselines",
    "recommended_weight_sharing": "Shared RGB encoder and shared thermal encoder, with optional representation-specific projection heads",
    "recommended_embedding_dimension": 128,
    "recommended_pooling": "Global average pooling followed by projection head",
    "recommended_patch_strategy": "Encode patches independently and aggregate them through mean or attention pooling",
    "recommended_regularization": "Mild augmentation, dropout, weight decay, class weighting/focal loss, early stopping",
    "not_recommended_for_first_implementation": "ViT, heavy hybrid CNN-transformer models, semantic memory, independent encoder for every representation",
    "scientific_rationale": (
        "The dataset has only 198 participants, so the encoder must be compact and strongly regularized. "
        "Because Stage 10C produced 12,672 encoder-ready representations, the model can exploit multi-representation evidence, "
        "but participant-level sample size remains the limiting factor. A lightweight CNN with shared modality encoders provides "
        "the best balance between feasibility, interpretability, and publishability."
    )
}

with open(STAGE_OUT / "Stage9D_Final_Encoder_Recommendation.json", "w", encoding="utf-8") as f:
    json.dump(final_recommendation, f, indent=4, ensure_ascii=False)


# ---------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------

report = []
report.append("# Stage 9D Representation Encoder Design and Selection\n")
report.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

report.append("## Purpose\n")
report.append(
    "This stage selects the representation encoder strategy for CPMR-Net before implementing neural encoders. "
    "The selection accounts for dataset size, representation count, overfitting risk, computational feasibility, "
    "interpretability, and publishability.\n"
)

report.append("## Encoder Family Ranking\n")
for _, row in enc_df.iterrows():
    report.append(f"### {row['encoder_id']}. {row['encoder_family']}")
    report.append(f"- Description: {row['description']}")
    report.append(f"- Selection score: {row['selection_score']}")
    report.append(f"- Recommended role: {row['recommended_role']}\n")

report.append("## Recommended Weight-Sharing Strategy\n")
report.append(
    "The recommended first implementation is to use a shared RGB encoder and a shared thermal encoder. "
    "This balances modality specialization with parameter efficiency. A more advanced variant can add "
    "representation-specific projection heads without creating a separate encoder for every representation.\n"
)

report.append("## Embedding Design\n")
for _, row in embed_df.iterrows():
    report.append(f"- **{row['component']}**: {row['recommendation']} — {row['reason']}")

report.append("\n## Regularization Strategy\n")
for _, row in reg_df.iterrows():
    report.append(f"- **{row['method']}**: {row['recommendation']} — {row['reason']}")

report.append("\n## Final Recommendation\n")
for k, v in final_recommendation.items():
    report.append(f"- **{k}**: {v}")

report.append("\n## Generated Output Files\n")
report.append("- `encoder_family_selection_matrix.csv`")
report.append("- `weight_sharing_strategy_matrix.csv`")
report.append("- `embedding_design_recommendations.csv`")
report.append("- `regularization_strategy.csv`")
report.append("- `computational_feasibility_plan.csv`")
report.append("- `Stage9D_Final_Encoder_Recommendation.json`")

with open(REPORTS_OUT / "Stage9D_Representation_Encoder_Design_and_Selection_Report.md", "w", encoding="utf-8") as f:
    f.write("\n".join(report))


summary = {
    "stage": "Stage9D",
    "title": "Representation Encoder Design and Selection",
    "candidate_encoder_families": len(enc_df),
    "recommended_primary_encoder": "Custom lightweight CNN",
    "recommended_baselines": ["ResNet-18 pretrained", "MobileNetV3-Small pretrained"],
    "recommended_embedding_dimension": 128,
    "recommended_weight_sharing": "Shared RGB encoder and shared thermal encoder",
    "outputs_saved_to": str(STAGE_OUT)
}

with open(STAGE_OUT / "Stage9D_Representation_Encoder_Design_and_Selection_Summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4, ensure_ascii=False)

print("=" * 80)
print("STAGE 9D REPRESENTATION ENCODER DESIGN AND SELECTION COMPLETED")
print("=" * 80)
print(f"Candidate encoder families evaluated: {summary['candidate_encoder_families']}")
print(f"Recommended primary encoder: {summary['recommended_primary_encoder']}")
print(f"Recommended baselines: {', '.join(summary['recommended_baselines'])}")
print(f"Recommended embedding dimension: {summary['recommended_embedding_dimension']}")
print(f"Recommended weight sharing: {summary['recommended_weight_sharing']}")
print(f"Outputs saved to: {STAGE_OUT}")
print("=" * 80)