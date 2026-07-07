# -*- coding: utf-8 -*-
"""
Stage 10E - Anatomical Specialist Embedding Aggregation

Purpose:
Aggregate Stage 10D representation-level embeddings into anatomical specialist embeddings.

Input:
- all_representation_embedding_manifest.csv
- all_representation_embeddings.npy

Output:
- One specialist embedding per participant × modality × anatomical view
- Representation-level contribution weights within each specialist
- Validation summaries

Important:
Stage 10D embeddings are initialized encoder embeddings for pipeline validation only.
Therefore, Stage 10E specialist embeddings are also validation embeddings, not final trained scientific embeddings.
"""

from pathlib import Path
import json
from datetime import datetime

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

BASE_DIR = Path(r"D:\47\472\New-Papers\Anemia_Paper")
OUTPUTS_DIR = BASE_DIR / "Outputs"

STAGE10D_DIR = OUTPUTS_DIR / "Stage10D_Representation_Encoder_Implementation"

INPUT_MANIFEST = STAGE10D_DIR / "tables" / "all_representation_embedding_manifest.csv"
INPUT_EMBEDDINGS = STAGE10D_DIR / "embeddings" / "all_representation_embeddings.npy"

STAGE_OUT = OUTPUTS_DIR / "Stage10E_Anatomical_Specialist_Embedding_Aggregation"
TABLES_OUT = STAGE_OUT / "tables"
REPORTS_OUT = STAGE_OUT / "reports"
EMBEDDINGS_OUT = STAGE_OUT / "embeddings"

for p in [TABLES_OUT, REPORTS_OUT, EMBEDDINGS_OUT]:
    p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

EMBEDDING_DIM = 128

EXPECTED_MODALITIES = ["rgb", "thermal"]
EXPECTED_VIEWS = ["l_dorsal", "l_palmar", "r_dorsal", "r_palmar"]

EXPECTED_REPRESENTATIONS = {
    "rgb": 9,       # rgb_original, rgb_hsv, rgb_lab, rgb_texture, 5 patches
    "thermal": 7,   # thermal_normalized, thermal_texture, 5 patches
}


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def softmax_weights(values):
    """
    Deterministic attention-like weights based on embedding norms.

    This is not a trained attention mechanism.
    It is used only to validate the specialist aggregation pipeline.
    """
    values = np.asarray(values, dtype=np.float32)

    if len(values) == 0:
        return values

    values = values - np.max(values)
    exp_values = np.exp(values)
    denom = np.sum(exp_values)

    if denom <= 0:
        return np.ones_like(values) / len(values)

    return exp_values / denom


def aggregate_specialist(group_df, embeddings):
    """
    Aggregate representation embeddings for one:
    participant_id × modality × view.

    Returns:
    - mean embedding
    - norm-weighted embedding
    - representation weights
    """

    indices = group_df["global_embedding_index"].astype(int).values
    group_embeddings = embeddings[indices]

    mean_embedding = group_embeddings.mean(axis=0)

    norms = np.linalg.norm(group_embeddings, axis=1)
    weights = softmax_weights(norms)

    weighted_embedding = np.sum(group_embeddings * weights[:, None], axis=0)

    return mean_embedding, weighted_embedding, weights, norms


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    if not INPUT_MANIFEST.exists():
        raise FileNotFoundError(f"Missing Stage 10D embedding manifest: {INPUT_MANIFEST}")

    if not INPUT_EMBEDDINGS.exists():
        raise FileNotFoundError(f"Missing Stage 10D embeddings: {INPUT_EMBEDDINGS}")

    manifest = pd.read_csv(INPUT_MANIFEST)
    embeddings = np.load(INPUT_EMBEDDINGS)

    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embedding array, got shape: {embeddings.shape}")

    if embeddings.shape[1] != EMBEDDING_DIM:
        raise ValueError(f"Expected embedding dim {EMBEDDING_DIM}, got {embeddings.shape[1]}")

    if len(manifest) != embeddings.shape[0]:
        raise ValueError(
            f"Manifest rows ({len(manifest)}) do not match embeddings rows ({embeddings.shape[0]})"
        )

    required_columns = [
        "participant_id",
        "label",
        "modality",
        "view",
        "representation",
        "global_embedding_index",
    ]

    missing_columns = [c for c in required_columns if c not in manifest.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns in Stage 10D manifest: {missing_columns}")

    specialist_records = []
    mean_specialist_embeddings = []
    weighted_specialist_embeddings = []
    representation_weight_records = []

    grouped = manifest.groupby(["participant_id", "label", "modality", "view"], sort=True)

    for specialist_index, ((participant_id, label, modality, view), group_df) in enumerate(grouped):
        group_df = group_df.sort_values(["representation", "global_embedding_index"]).reset_index(drop=True)

        mean_embedding, weighted_embedding, weights, norms = aggregate_specialist(group_df, embeddings)

        mean_specialist_embeddings.append(mean_embedding.astype(np.float32))
        weighted_specialist_embeddings.append(weighted_embedding.astype(np.float32))

        expected_count = EXPECTED_REPRESENTATIONS.get(modality, None)
        observed_count = len(group_df)

        specialist_records.append({
            "specialist_index": specialist_index,
            "participant_id": participant_id,
            "label": int(label),
            "modality": modality,
            "view": view,
            "observed_representation_count": int(observed_count),
            "expected_representation_count": int(expected_count) if expected_count is not None else None,
            "complete_specialist": bool(observed_count == expected_count),
            "mean_embedding_norm": float(np.linalg.norm(mean_embedding)),
            "weighted_embedding_norm": float(np.linalg.norm(weighted_embedding)),
        })

        for i, row in group_df.iterrows():
            representation_weight_records.append({
                "specialist_index": specialist_index,
                "participant_id": participant_id,
                "label": int(label),
                "modality": modality,
                "view": view,
                "representation": row["representation"],
                "global_embedding_index": int(row["global_embedding_index"]),
                "embedding_group": row.get("embedding_group", ""),
                "representation_norm": float(norms[i]),
                "representation_weight": float(weights[i]),
            })

    specialist_manifest = pd.DataFrame(specialist_records)

    mean_specialist_embeddings = np.vstack(mean_specialist_embeddings).astype(np.float32)
    weighted_specialist_embeddings = np.vstack(weighted_specialist_embeddings).astype(np.float32)

    weights_df = pd.DataFrame(representation_weight_records)

    specialist_manifest.to_csv(TABLES_OUT / "anatomical_specialist_embedding_manifest.csv", index=False)
    weights_df.to_csv(TABLES_OUT / "specialist_representation_weights.csv", index=False)

    np.save(EMBEDDINGS_OUT / "mean_anatomical_specialist_embeddings.npy", mean_specialist_embeddings)
    np.save(EMBEDDINGS_OUT / "weighted_anatomical_specialist_embeddings.npy", weighted_specialist_embeddings)

    # Completeness validation
    expected_specialists_per_participant = len(EXPECTED_MODALITIES) * len(EXPECTED_VIEWS)

    participant_specialist_counts = (
        specialist_manifest.groupby("participant_id")
        .agg(
            specialist_count=("specialist_index", "count"),
            complete_specialists=("complete_specialist", "sum")
        )
        .reset_index()
    )

    participant_specialist_counts["expected_specialists"] = expected_specialists_per_participant
    participant_specialist_counts["complete_participant_specialists"] = (
        participant_specialist_counts["specialist_count"] == participant_specialist_counts["expected_specialists"]
    ) & (
        participant_specialist_counts["complete_specialists"] == participant_specialist_counts["expected_specialists"]
    )

    participant_specialist_counts.to_csv(TABLES_OUT / "participant_specialist_completeness.csv", index=False)

    specialist_count_summary = (
        specialist_manifest.groupby(["modality", "view"])
        .agg(
            specialists=("specialist_index", "count"),
            complete_specialists=("complete_specialist", "sum"),
            mean_observed_representations=("observed_representation_count", "mean"),
            mean_weighted_embedding_norm=("weighted_embedding_norm", "mean")
        )
        .reset_index()
    )

    specialist_count_summary.to_csv(TABLES_OUT / "specialist_count_summary_by_modality_view.csv", index=False)

    representation_weight_summary = (
        weights_df.groupby(["modality", "view", "representation"])
        .agg(
            records=("representation_weight", "count"),
            mean_weight=("representation_weight", "mean"),
            std_weight=("representation_weight", "std"),
            mean_norm=("representation_norm", "mean")
        )
        .reset_index()
    )

    representation_weight_summary.to_csv(TABLES_OUT / "representation_weight_summary.csv", index=False)

    # Wide participant specialist manifest for easier downstream use
    wide_rows = []

    for participant_id, p_df in specialist_manifest.groupby("participant_id"):
        row = {
            "participant_id": participant_id,
            "label": int(p_df["label"].iloc[0]),
        }

        for _, r in p_df.iterrows():
            key = f"{r['modality']}_{r['view']}_specialist_index"
            row[key] = int(r["specialist_index"])

        wide_rows.append(row)

    wide_df = pd.DataFrame(wide_rows)
    wide_df.to_csv(TABLES_OUT / "participant_to_specialist_index_map.csv", index=False)

    summary = {
        "stage": "Stage10E",
        "title": "Anatomical Specialist Embedding Aggregation",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_manifest": str(INPUT_MANIFEST),
        "input_embeddings": str(INPUT_EMBEDDINGS),
        "input_representation_embeddings": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "specialist_embeddings": int(len(specialist_manifest)),
        "expected_specialists_per_participant": expected_specialists_per_participant,
        "participants": int(specialist_manifest["participant_id"].nunique()),
        "participants_complete_specialists": int(participant_specialist_counts["complete_participant_specialists"].sum()),
        "participants_incomplete_specialists": int((~participant_specialist_counts["complete_participant_specialists"]).sum()),
        "complete_specialists": int(specialist_manifest["complete_specialist"].sum()),
        "incomplete_specialists": int((~specialist_manifest["complete_specialist"]).sum()),
        "mean_specialist_embeddings_shape": list(mean_specialist_embeddings.shape),
        "weighted_specialist_embeddings_shape": list(weighted_specialist_embeddings.shape),
        "has_nan_mean_embeddings": bool(np.isnan(mean_specialist_embeddings).any()),
        "has_inf_mean_embeddings": bool(np.isinf(mean_specialist_embeddings).any()),
        "has_nan_weighted_embeddings": bool(np.isnan(weighted_specialist_embeddings).any()),
        "has_inf_weighted_embeddings": bool(np.isinf(weighted_specialist_embeddings).any()),
        "aggregation_strategy": "mean embedding and norm-weighted deterministic attention-like aggregation",
        "note": (
            "Specialist embeddings are generated from initialized Stage 10D encoder embeddings for pipeline validation. "
            "They are not final trained scientific specialist embeddings."
        ),
        "outputs_saved_to": str(STAGE_OUT),
    }

    with open(STAGE_OUT / "Stage10E_Anatomical_Specialist_Embedding_Aggregation_Summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    report = []
    report.append("# Stage 10E Anatomical Specialist Embedding Aggregation\n")
    report.append(f"Generated at: {summary['created_at']}\n")

    report.append("## Purpose\n")
    report.append(
        "This stage aggregates representation-level embeddings into anatomical specialist embeddings. "
        "Each specialist corresponds to one participant, one modality, and one anatomical view. "
        "This converts the representation evidence produced in Stage 10D into the anatomical evidence units required by CPMR-Net.\n"
    )

    report.append("## Aggregation Strategy\n")
    report.append(
        "Two forms of specialist embeddings are produced: simple mean embeddings and deterministic norm-weighted embeddings. "
        "The norm-weighted version is not a trained attention mechanism; it is used only to validate the anatomical specialist aggregation pipeline. "
        "Trainable specialist attention will be implemented later during full CPMR-Net training.\n"
    )

    report.append("## Summary\n")
    report.append(f"- Input representation embeddings: {summary['input_representation_embeddings']}")
    report.append(f"- Embedding dimension: {summary['embedding_dim']}")
    report.append(f"- Specialist embeddings: {summary['specialist_embeddings']}")
    report.append(f"- Participants: {summary['participants']}")
    report.append(f"- Expected specialists per participant: {summary['expected_specialists_per_participant']}")
    report.append(f"- Participants complete specialists: {summary['participants_complete_specialists']}")
    report.append(f"- Participants incomplete specialists: {summary['participants_incomplete_specialists']}")
    report.append(f"- Complete specialists: {summary['complete_specialists']}")
    report.append(f"- Incomplete specialists: {summary['incomplete_specialists']}")
    report.append(f"- Mean specialist embeddings shape: {summary['mean_specialist_embeddings_shape']}")
    report.append(f"- Weighted specialist embeddings shape: {summary['weighted_specialist_embeddings_shape']}")
    report.append(f"- NaN in mean embeddings: {summary['has_nan_mean_embeddings']}")
    report.append(f"- Inf in mean embeddings: {summary['has_inf_mean_embeddings']}")
    report.append(f"- NaN in weighted embeddings: {summary['has_nan_weighted_embeddings']}")
    report.append(f"- Inf in weighted embeddings: {summary['has_inf_weighted_embeddings']}\n")

    report.append("## Output Files\n")
    report.append("- `tables/anatomical_specialist_embedding_manifest.csv`")
    report.append("- `tables/specialist_representation_weights.csv`")
    report.append("- `tables/participant_specialist_completeness.csv`")
    report.append("- `tables/specialist_count_summary_by_modality_view.csv`")
    report.append("- `tables/representation_weight_summary.csv`")
    report.append("- `tables/participant_to_specialist_index_map.csv`")
    report.append("- `embeddings/mean_anatomical_specialist_embeddings.npy`")
    report.append("- `embeddings/weighted_anatomical_specialist_embeddings.npy`")
    report.append("- `Stage10E_Anatomical_Specialist_Embedding_Aggregation_Summary.json`\n")

    report.append("## Important Note\n")
    report.append(summary["note"])

    with open(REPORTS_OUT / "Stage10E_Anatomical_Specialist_Embedding_Aggregation_Report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("=" * 80)
    print("STAGE 10E ANATOMICAL SPECIALIST EMBEDDING AGGREGATION COMPLETED")
    print("=" * 80)
    print(f"Input representation embeddings: {summary['input_representation_embeddings']}")
    print(f"Specialist embeddings: {summary['specialist_embeddings']}")
    print(f"Participants: {summary['participants']}")
    print(f"Expected specialists per participant: {summary['expected_specialists_per_participant']}")
    print(f"Participants complete specialists: {summary['participants_complete_specialists']}")
    print(f"Participants incomplete specialists: {summary['participants_incomplete_specialists']}")
    print(f"Mean specialist embeddings shape: {mean_specialist_embeddings.shape}")
    print(f"Weighted specialist embeddings shape: {weighted_specialist_embeddings.shape}")
    print(f"NaN weighted embeddings: {summary['has_nan_weighted_embeddings']}")
    print(f"Inf weighted embeddings: {summary['has_inf_weighted_embeddings']}")
    print(f"Outputs saved to: {STAGE_OUT}")
    print("=" * 80)
    print("NOTE: These specialist embeddings are for pipeline validation only, not final trained scientific embeddings.")


if __name__ == "__main__":
    main()