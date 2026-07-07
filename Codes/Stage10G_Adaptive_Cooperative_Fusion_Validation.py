# -*- coding: utf-8 -*-
"""
Stage 10G - Adaptive Cooperative Fusion Validation

Purpose:
Validate the CPMR-Net adaptive cooperative fusion pathway.

Input:
- RGB-centered evidence embeddings from Stage 10F
- Thermal auxiliary evidence embeddings from Stage 10F
- Participant modality branch map

Output:
- Fused participant-level embeddings
- Deterministic modality contribution weights
- Fusion validation summaries

Important:
This is a validation-stage fusion using deterministic norm-based weights.
It is NOT the final trainable adaptive fusion layer.
"""

from pathlib import Path
import json
from datetime import datetime

import numpy as np
import pandas as pd


BASE_DIR = Path(r"D:\47\472\New-Papers\Anemia_Paper")
OUTPUTS_DIR = BASE_DIR / "Outputs"

STAGE10F_DIR = OUTPUTS_DIR / "Stage10F_RGB_Thermal_Evidence_Branches"

INPUT_BRANCH_MAP = STAGE10F_DIR / "tables" / "participant_modality_branch_map.csv"
RGB_EMBEDDINGS_FILE = STAGE10F_DIR / "embeddings" / "rgb_centered_weighted_evidence_embeddings.npy"
THERMAL_EMBEDDINGS_FILE = STAGE10F_DIR / "embeddings" / "thermal_auxiliary_weighted_evidence_embeddings.npy"

RGB_BRANCH_MANIFEST = STAGE10F_DIR / "tables" / "rgb_centered_evidence_branch_manifest.csv"
THERMAL_BRANCH_MANIFEST = STAGE10F_DIR / "tables" / "thermal_auxiliary_evidence_branch_manifest.csv"

STAGE_OUT = OUTPUTS_DIR / "Stage10G_Adaptive_Cooperative_Fusion_Validation"
TABLES_OUT = STAGE_OUT / "tables"
REPORTS_OUT = STAGE_OUT / "reports"
EMBEDDINGS_OUT = STAGE_OUT / "embeddings"

for p in [TABLES_OUT, REPORTS_OUT, EMBEDDINGS_OUT]:
    p.mkdir(parents=True, exist_ok=True)

EMBEDDING_DIM = 128


def softmax_weights(values):
    values = np.asarray(values, dtype=np.float32)
    values = values - np.max(values)
    exp_values = np.exp(values)
    denom = np.sum(exp_values)
    if denom <= 0:
        return np.ones_like(values) / len(values)
    return exp_values / denom


def main():
    for path in [INPUT_BRANCH_MAP, RGB_EMBEDDINGS_FILE, THERMAL_EMBEDDINGS_FILE, RGB_BRANCH_MANIFEST, THERMAL_BRANCH_MANIFEST]:
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing required input: {path}")

    branch_map = pd.read_csv(INPUT_BRANCH_MAP)
    rgb_manifest = pd.read_csv(RGB_BRANCH_MANIFEST)
    thermal_manifest = pd.read_csv(THERMAL_BRANCH_MANIFEST)

    rgb_embeddings = np.load(RGB_EMBEDDINGS_FILE)
    thermal_embeddings = np.load(THERMAL_EMBEDDINGS_FILE)

    if rgb_embeddings.shape[1] != EMBEDDING_DIM:
        raise ValueError(f"RGB embedding dimension mismatch: {rgb_embeddings.shape}")

    if thermal_embeddings.shape[1] != EMBEDDING_DIM:
        raise ValueError(f"Thermal embedding dimension mismatch: {thermal_embeddings.shape}")

    if len(rgb_manifest) != rgb_embeddings.shape[0]:
        raise ValueError("RGB manifest rows do not match RGB embedding rows.")

    if len(thermal_manifest) != thermal_embeddings.shape[0]:
        raise ValueError("Thermal manifest rows do not match thermal embedding rows.")

    rgb_index_map = {
        int(row["branch_index"]): i
        for i, row in rgb_manifest.reset_index(drop=True).iterrows()
    }

    thermal_index_map = {
        int(row["branch_index"]): i
        for i, row in thermal_manifest.reset_index(drop=True).iterrows()
    }

    fusion_records = []
    contribution_records = []
    fused_mean_embeddings = []
    fused_weighted_embeddings = []
    fused_rgb_dominant_embeddings = []

    for fusion_index, row in branch_map.sort_values("participant_id").reset_index(drop=True).iterrows():
        participant_id = row["participant_id"]
        label = int(row["label"])

        rgb_branch_index = int(row["rgb_branch_index"])
        thermal_branch_index = int(row["thermal_branch_index"])

        if rgb_branch_index not in rgb_index_map:
            raise ValueError(f"RGB branch index not found: {rgb_branch_index}")

        if thermal_branch_index not in thermal_index_map:
            raise ValueError(f"Thermal branch index not found: {thermal_branch_index}")

        rgb_vec = rgb_embeddings[rgb_index_map[rgb_branch_index]]
        thermal_vec = thermal_embeddings[thermal_index_map[thermal_branch_index]]

        rgb_norm = float(np.linalg.norm(rgb_vec))
        thermal_norm = float(np.linalg.norm(thermal_vec))

        # Deterministic validation fusion
        modality_weights = softmax_weights([rgb_norm, thermal_norm])
        rgb_weight = float(modality_weights[0])
        thermal_weight = float(modality_weights[1])

        fused_mean = ((rgb_vec + thermal_vec) / 2.0).astype(np.float32)
        fused_weighted = (rgb_weight * rgb_vec + thermal_weight * thermal_vec).astype(np.float32)

        # RGB-centered variant: enforce RGB as primary branch in validation
        rgb_centered_weight = 0.70
        thermal_aux_weight = 0.30
        fused_rgb_dominant = (
            rgb_centered_weight * rgb_vec + thermal_aux_weight * thermal_vec
        ).astype(np.float32)

        fused_mean_embeddings.append(fused_mean)
        fused_weighted_embeddings.append(fused_weighted)
        fused_rgb_dominant_embeddings.append(fused_rgb_dominant)

        fusion_records.append({
            "fusion_index": fusion_index,
            "participant_id": participant_id,
            "label": label,
            "rgb_branch_index": rgb_branch_index,
            "thermal_branch_index": thermal_branch_index,
            "rgb_norm": rgb_norm,
            "thermal_norm": thermal_norm,
            "rgb_weight_norm_based": rgb_weight,
            "thermal_weight_norm_based": thermal_weight,
            "rgb_weight_rgb_centered": rgb_centered_weight,
            "thermal_weight_rgb_centered": thermal_aux_weight,
            "mean_fused_norm": float(np.linalg.norm(fused_mean)),
            "weighted_fused_norm": float(np.linalg.norm(fused_weighted)),
            "rgb_centered_fused_norm": float(np.linalg.norm(fused_rgb_dominant)),
            "complete_fusion_pair": bool(row["complete_modality_pair"]),
        })

        contribution_records.append({
            "fusion_index": fusion_index,
            "participant_id": participant_id,
            "label": label,
            "modality": "rgb",
            "branch_index": rgb_branch_index,
            "norm": rgb_norm,
            "norm_based_weight": rgb_weight,
            "rgb_centered_weight": rgb_centered_weight,
        })

        contribution_records.append({
            "fusion_index": fusion_index,
            "participant_id": participant_id,
            "label": label,
            "modality": "thermal",
            "branch_index": thermal_branch_index,
            "norm": thermal_norm,
            "norm_based_weight": thermal_weight,
            "rgb_centered_weight": thermal_aux_weight,
        })

    fused_mean_embeddings = np.vstack(fused_mean_embeddings).astype(np.float32)
    fused_weighted_embeddings = np.vstack(fused_weighted_embeddings).astype(np.float32)
    fused_rgb_dominant_embeddings = np.vstack(fused_rgb_dominant_embeddings).astype(np.float32)

    fusion_manifest = pd.DataFrame(fusion_records)
    contribution_df = pd.DataFrame(contribution_records)

    fusion_manifest.to_csv(TABLES_OUT / "adaptive_cooperative_fusion_manifest.csv", index=False)
    contribution_df.to_csv(TABLES_OUT / "modality_contribution_weights.csv", index=False)

    np.save(EMBEDDINGS_OUT / "mean_fused_participant_embeddings.npy", fused_mean_embeddings)
    np.save(EMBEDDINGS_OUT / "norm_weighted_fused_participant_embeddings.npy", fused_weighted_embeddings)
    np.save(EMBEDDINGS_OUT / "rgb_centered_fused_participant_embeddings.npy", fused_rgb_dominant_embeddings)

    contribution_summary = (
        contribution_df.groupby("modality")
        .agg(
            records=("modality", "count"),
            mean_norm=("norm", "mean"),
            std_norm=("norm", "std"),
            mean_norm_based_weight=("norm_based_weight", "mean"),
            std_norm_based_weight=("norm_based_weight", "std"),
            mean_rgb_centered_weight=("rgb_centered_weight", "mean"),
        )
        .reset_index()
    )
    contribution_summary.to_csv(TABLES_OUT / "modality_contribution_summary.csv", index=False)

    embedding_validation = pd.DataFrame([
        {
            "embedding_type": "mean_fused",
            "shape": str(list(fused_mean_embeddings.shape)),
            "has_nan": bool(np.isnan(fused_mean_embeddings).any()),
            "has_inf": bool(np.isinf(fused_mean_embeddings).any()),
            "mean": float(np.mean(fused_mean_embeddings)),
            "std": float(np.std(fused_mean_embeddings)),
            "min": float(np.min(fused_mean_embeddings)),
            "max": float(np.max(fused_mean_embeddings)),
        },
        {
            "embedding_type": "norm_weighted_fused",
            "shape": str(list(fused_weighted_embeddings.shape)),
            "has_nan": bool(np.isnan(fused_weighted_embeddings).any()),
            "has_inf": bool(np.isinf(fused_weighted_embeddings).any()),
            "mean": float(np.mean(fused_weighted_embeddings)),
            "std": float(np.std(fused_weighted_embeddings)),
            "min": float(np.min(fused_weighted_embeddings)),
            "max": float(np.max(fused_weighted_embeddings)),
        },
        {
            "embedding_type": "rgb_centered_fused",
            "shape": str(list(fused_rgb_dominant_embeddings.shape)),
            "has_nan": bool(np.isnan(fused_rgb_dominant_embeddings).any()),
            "has_inf": bool(np.isinf(fused_rgb_dominant_embeddings).any()),
            "mean": float(np.mean(fused_rgb_dominant_embeddings)),
            "std": float(np.std(fused_rgb_dominant_embeddings)),
            "min": float(np.min(fused_rgb_dominant_embeddings)),
            "max": float(np.max(fused_rgb_dominant_embeddings)),
        },
    ])
    embedding_validation.to_csv(TABLES_OUT / "fusion_embedding_validation_summary.csv", index=False)

    summary = {
        "stage": "Stage10G",
        "title": "Adaptive Cooperative Fusion Validation",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "participants_fused": int(len(fusion_manifest)),
        "embedding_dim": EMBEDDING_DIM,
        "complete_fusion_pairs": int(fusion_manifest["complete_fusion_pair"].sum()),
        "incomplete_fusion_pairs": int((~fusion_manifest["complete_fusion_pair"]).sum()),
        "mean_fused_shape": list(fused_mean_embeddings.shape),
        "norm_weighted_fused_shape": list(fused_weighted_embeddings.shape),
        "rgb_centered_fused_shape": list(fused_rgb_dominant_embeddings.shape),
        "has_nan_any": bool(
            np.isnan(fused_mean_embeddings).any()
            or np.isnan(fused_weighted_embeddings).any()
            or np.isnan(fused_rgb_dominant_embeddings).any()
        ),
        "has_inf_any": bool(
            np.isinf(fused_mean_embeddings).any()
            or np.isinf(fused_weighted_embeddings).any()
            or np.isinf(fused_rgb_dominant_embeddings).any()
        ),
        "fusion_variants": [
            "mean_fusion",
            "norm_weighted_validation_fusion",
            "rgb_centered_70_30_validation_fusion",
        ],
        "note": (
            "This stage validates fusion mechanics only. The final CPMR-Net will use trainable "
            "adaptive cooperative fusion during supervised learning."
        ),
        "outputs_saved_to": str(STAGE_OUT),
    }

    with open(STAGE_OUT / "Stage10G_Adaptive_Cooperative_Fusion_Validation_Summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    report = []
    report.append("# Stage 10G Adaptive Cooperative Fusion Validation\n")
    report.append(f"Generated at: {summary['created_at']}\n")

    report.append("## Purpose\n")
    report.append(
        "This stage validates the CPMR-Net cooperative fusion pathway by combining the RGB-centered "
        "and thermal-auxiliary participant evidence vectors into fused participant-level embeddings.\n"
    )

    report.append("## Fusion Variants\n")
    report.append("- Mean fusion: equal RGB and thermal contribution.")
    report.append("- Norm-weighted fusion: deterministic validation weighting based on branch embedding norms.")
    report.append("- RGB-centered fusion: fixed 70% RGB and 30% thermal weighting to reflect the RGB-centered design.\n")

    report.append("## Summary\n")
    report.append(f"- Participants fused: {summary['participants_fused']}")
    report.append(f"- Complete fusion pairs: {summary['complete_fusion_pairs']}")
    report.append(f"- Incomplete fusion pairs: {summary['incomplete_fusion_pairs']}")
    report.append(f"- Mean fused shape: {summary['mean_fused_shape']}")
    report.append(f"- Norm-weighted fused shape: {summary['norm_weighted_fused_shape']}")
    report.append(f"- RGB-centered fused shape: {summary['rgb_centered_fused_shape']}")
    report.append(f"- NaN detected: {summary['has_nan_any']}")
    report.append(f"- Inf detected: {summary['has_inf_any']}\n")

    report.append("## Output Files\n")
    report.append("- `tables/adaptive_cooperative_fusion_manifest.csv`")
    report.append("- `tables/modality_contribution_weights.csv`")
    report.append("- `tables/modality_contribution_summary.csv`")
    report.append("- `tables/fusion_embedding_validation_summary.csv`")
    report.append("- `embeddings/mean_fused_participant_embeddings.npy`")
    report.append("- `embeddings/norm_weighted_fused_participant_embeddings.npy`")
    report.append("- `embeddings/rgb_centered_fused_participant_embeddings.npy`")
    report.append("- `Stage10G_Adaptive_Cooperative_Fusion_Validation_Summary.json`\n")

    report.append("## Important Note\n")
    report.append(summary["note"])

    with open(REPORTS_OUT / "Stage10G_Adaptive_Cooperative_Fusion_Validation_Report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("=" * 80)
    print("STAGE 10G ADAPTIVE COOPERATIVE FUSION VALIDATION COMPLETED")
    print("=" * 80)
    print(f"Participants fused: {summary['participants_fused']}")
    print(f"Complete fusion pairs: {summary['complete_fusion_pairs']}")
    print(f"Incomplete fusion pairs: {summary['incomplete_fusion_pairs']}")
    print(f"Mean fused shape: {fused_mean_embeddings.shape}")
    print(f"Norm-weighted fused shape: {fused_weighted_embeddings.shape}")
    print(f"RGB-centered fused shape: {fused_rgb_dominant_embeddings.shape}")
    print(f"NaN detected: {summary['has_nan_any']}")
    print(f"Inf detected: {summary['has_inf_any']}")
    print(f"Outputs saved to: {STAGE_OUT}")
    print("=" * 80)
    print("NOTE: This is validation fusion only, not final trained adaptive CPMR-Net fusion.")


if __name__ == "__main__":
    main()