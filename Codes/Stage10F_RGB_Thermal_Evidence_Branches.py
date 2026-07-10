# -*- coding: utf-8 -*-
"""
Stage 10F - RGB-Centered and Thermal-Auxiliary Evidence Branches

Purpose:
Aggregate anatomical specialist embeddings from Stage 10E into participant-level
modality evidence vectors:

1. RGB-centered participant evidence vector
2. Thermal auxiliary participant evidence vector

Input:
- anatomical_specialist_embedding_manifest.csv
- weighted_anatomical_specialist_embeddings.npy

Output:
- RGB evidence embeddings
- Thermal evidence embeddings
- Modality branch manifest
- View contribution weights
- Participant-level modality completeness validation

Important:
These modality evidence vectors are generated from initialized validation embeddings.
They are not final trained scientific evidence vectors.
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

STAGE10E_DIR = OUTPUTS_DIR / "Stage10E_Anatomical_Specialist_Embedding_Aggregation"

INPUT_SPECIALIST_MANIFEST = STAGE10E_DIR / "tables" / "anatomical_specialist_embedding_manifest.csv"
INPUT_SPECIALIST_EMBEDDINGS = STAGE10E_DIR / "embeddings" / "weighted_anatomical_specialist_embeddings.npy"

STAGE_OUT = OUTPUTS_DIR / "Stage10F_RGB_Thermal_Evidence_Branches"
TABLES_OUT = STAGE_OUT / "tables"
REPORTS_OUT = STAGE_OUT / "reports"
EMBEDDINGS_OUT = STAGE_OUT / "embeddings"

for p in [TABLES_OUT, REPORTS_OUT, EMBEDDINGS_OUT]:
    p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

EMBEDDING_DIM = 128
EXPECTED_VIEWS = ["l_dorsal", "l_palmar", "r_dorsal", "r_palmar"]
EXPECTED_SPECIALISTS_PER_MODALITY = 4
EXPECTED_MODALITIES = ["rgb", "thermal"]


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def softmax_weights(values):
    values = np.asarray(values, dtype=np.float32)

    if len(values) == 0:
        return values

    values = values - np.max(values)
    exp_values = np.exp(values)
    denom = np.sum(exp_values)

    if denom <= 0:
        return np.ones_like(values) / len(values)

    return exp_values / denom


def aggregate_modality_branch(group_df, specialist_embeddings):
    """
    Aggregate four anatomical specialists into one modality-level evidence vector.
    """

    indices = group_df["specialist_index"].astype(int).values
    emb = specialist_embeddings[indices]

    mean_embedding = emb.mean(axis=0)

    norms = np.linalg.norm(emb, axis=1)
    weights = softmax_weights(norms)

    weighted_embedding = np.sum(emb * weights[:, None], axis=0)

    return mean_embedding, weighted_embedding, weights, norms


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    if not INPUT_SPECIALIST_MANIFEST.exists():
        raise FileNotFoundError(f"Missing Stage 10E specialist manifest: {INPUT_SPECIALIST_MANIFEST}")

    if not INPUT_SPECIALIST_EMBEDDINGS.exists():
        raise FileNotFoundError(f"Missing Stage 10E specialist embeddings: {INPUT_SPECIALIST_EMBEDDINGS}")

    specialist_manifest = pd.read_csv(INPUT_SPECIALIST_MANIFEST)
    specialist_embeddings = np.load(INPUT_SPECIALIST_EMBEDDINGS)

    if specialist_embeddings.ndim != 2:
        raise ValueError(f"Expected 2D specialist embeddings, got shape: {specialist_embeddings.shape}")

    if specialist_embeddings.shape[1] != EMBEDDING_DIM:
        raise ValueError(f"Expected embedding dim {EMBEDDING_DIM}, got {specialist_embeddings.shape[1]}")

    if len(specialist_manifest) != specialist_embeddings.shape[0]:
        raise ValueError(
            f"Manifest rows ({len(specialist_manifest)}) do not match embeddings rows ({specialist_embeddings.shape[0]})"
        )

    required_cols = [
        "specialist_index",
        "participant_id",
        "label",
        "modality",
        "view",
        "complete_specialist",
    ]

    missing_cols = [c for c in required_cols if c not in specialist_manifest.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns in specialist manifest: {missing_cols}")

    branch_records = []
    mean_branch_embeddings = []
    weighted_branch_embeddings = []
    view_weight_records = []

    grouped = specialist_manifest.groupby(["participant_id", "label", "modality"], sort=True)

    for branch_index, ((participant_id, label, modality), group_df) in enumerate(grouped):
        group_df = group_df.sort_values("view").reset_index(drop=True)

        mean_embedding, weighted_embedding, weights, norms = aggregate_modality_branch(
            group_df,
            specialist_embeddings
        )

        mean_branch_embeddings.append(mean_embedding.astype(np.float32))
        weighted_branch_embeddings.append(weighted_embedding.astype(np.float32))

        observed_views = sorted(group_df["view"].tolist())
        missing_views = sorted(list(set(EXPECTED_VIEWS) - set(observed_views)))

        complete_branch = (
            len(group_df) == EXPECTED_SPECIALISTS_PER_MODALITY
            and len(missing_views) == 0
            and bool(group_df["complete_specialist"].all())
        )

        branch_records.append({
            "branch_index": branch_index,
            "participant_id": participant_id,
            "label": int(label),
            "modality": modality,
            "observed_specialists": int(len(group_df)),
            "expected_specialists": EXPECTED_SPECIALISTS_PER_MODALITY,
            "observed_views": ";".join(observed_views),
            "missing_views": ";".join(missing_views),
            "complete_branch": bool(complete_branch),
            "mean_branch_embedding_norm": float(np.linalg.norm(mean_embedding)),
            "weighted_branch_embedding_norm": float(np.linalg.norm(weighted_embedding)),
            "branch_role": "primary_rgb_evidence" if modality == "rgb" else "auxiliary_thermal_evidence",
        })

        for i, row in group_df.iterrows():
            view_weight_records.append({
                "branch_index": branch_index,
                "participant_id": participant_id,
                "label": int(label),
                "modality": modality,
                "view": row["view"],
                "specialist_index": int(row["specialist_index"]),
                "specialist_norm": float(norms[i]),
                "view_weight": float(weights[i]),
            })

    branch_manifest = pd.DataFrame(branch_records)
    view_weights_df = pd.DataFrame(view_weight_records)

    mean_branch_embeddings = np.vstack(mean_branch_embeddings).astype(np.float32)
    weighted_branch_embeddings = np.vstack(weighted_branch_embeddings).astype(np.float32)

    branch_manifest.to_csv(TABLES_OUT / "modality_evidence_branch_manifest.csv", index=False)
    view_weights_df.to_csv(TABLES_OUT / "modality_view_contribution_weights.csv", index=False)

    np.save(EMBEDDINGS_OUT / "mean_modality_evidence_embeddings.npy", mean_branch_embeddings)
    np.save(EMBEDDINGS_OUT / "weighted_modality_evidence_embeddings.npy", weighted_branch_embeddings)

    rgb_branch_manifest = branch_manifest[branch_manifest["modality"] == "rgb"].copy()
    thermal_branch_manifest = branch_manifest[branch_manifest["modality"] == "thermal"].copy()

    rgb_indices = rgb_branch_manifest["branch_index"].astype(int).values
    thermal_indices = thermal_branch_manifest["branch_index"].astype(int).values

    rgb_weighted_embeddings = weighted_branch_embeddings[rgb_indices]
    thermal_weighted_embeddings = weighted_branch_embeddings[thermal_indices]

    rgb_mean_embeddings = mean_branch_embeddings[rgb_indices]
    thermal_mean_embeddings = mean_branch_embeddings[thermal_indices]

    rgb_branch_manifest.to_csv(TABLES_OUT / "rgb_centered_evidence_branch_manifest.csv", index=False)
    thermal_branch_manifest.to_csv(TABLES_OUT / "thermal_auxiliary_evidence_branch_manifest.csv", index=False)

    np.save(EMBEDDINGS_OUT / "rgb_centered_weighted_evidence_embeddings.npy", rgb_weighted_embeddings)
    np.save(EMBEDDINGS_OUT / "thermal_auxiliary_weighted_evidence_embeddings.npy", thermal_weighted_embeddings)
    np.save(EMBEDDINGS_OUT / "rgb_centered_mean_evidence_embeddings.npy", rgb_mean_embeddings)
    np.save(EMBEDDINGS_OUT / "thermal_auxiliary_mean_evidence_embeddings.npy", thermal_mean_embeddings)

    # Participant-level paired RGB and thermal map
    participant_rows = []

    for participant_id, p_df in branch_manifest.groupby("participant_id"):
        p_df = p_df.sort_values("modality")

        row = {
            "participant_id": participant_id,
            "label": int(p_df["label"].iloc[0]),
        }

        for _, r in p_df.iterrows():
            row[f"{r['modality']}_branch_index"] = int(r["branch_index"])
            row[f"{r['modality']}_complete_branch"] = bool(r["complete_branch"])

        row["has_rgb_branch"] = "rgb_branch_index" in row
        row["has_thermal_branch"] = "thermal_branch_index" in row
        row["complete_modality_pair"] = (
            row.get("has_rgb_branch", False)
            and row.get("has_thermal_branch", False)
            and row.get("rgb_complete_branch", False)
            and row.get("thermal_complete_branch", False)
        )

        participant_rows.append(row)

    participant_branch_map = pd.DataFrame(participant_rows)
    participant_branch_map.to_csv(TABLES_OUT / "participant_modality_branch_map.csv", index=False)

    branch_summary = (
        branch_manifest.groupby("modality")
        .agg(
            branches=("branch_index", "count"),
            complete_branches=("complete_branch", "sum"),
            mean_observed_specialists=("observed_specialists", "mean"),
            mean_weighted_embedding_norm=("weighted_branch_embedding_norm", "mean"),
        )
        .reset_index()
    )

    branch_summary.to_csv(TABLES_OUT / "modality_branch_summary.csv", index=False)

    view_weight_summary = (
        view_weights_df.groupby(["modality", "view"])
        .agg(
            records=("view_weight", "count"),
            mean_view_weight=("view_weight", "mean"),
            std_view_weight=("view_weight", "std"),
            mean_specialist_norm=("specialist_norm", "mean"),
        )
        .reset_index()
    )

    view_weight_summary.to_csv(TABLES_OUT / "view_weight_summary_by_modality.csv", index=False)

    summary = {
        "stage": "Stage10F",
        "title": "RGB-Centered and Thermal-Auxiliary Evidence Branches",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_specialist_manifest": str(INPUT_SPECIALIST_MANIFEST),
        "input_specialist_embeddings": str(INPUT_SPECIALIST_EMBEDDINGS),
        "input_specialist_embeddings_count": int(specialist_embeddings.shape[0]),
        "embedding_dim": int(specialist_embeddings.shape[1]),
        "modality_branches": int(len(branch_manifest)),
        "rgb_branches": int(len(rgb_branch_manifest)),
        "thermal_branches": int(len(thermal_branch_manifest)),
        "expected_branches_per_participant": len(EXPECTED_MODALITIES),
        "participants": int(branch_manifest["participant_id"].nunique()),
        "participants_complete_modality_pair": int(participant_branch_map["complete_modality_pair"].sum()),
        "participants_incomplete_modality_pair": int((~participant_branch_map["complete_modality_pair"]).sum()),
        "complete_branches": int(branch_manifest["complete_branch"].sum()),
        "incomplete_branches": int((~branch_manifest["complete_branch"]).sum()),
        "rgb_weighted_embeddings_shape": list(rgb_weighted_embeddings.shape),
        "thermal_weighted_embeddings_shape": list(thermal_weighted_embeddings.shape),
        "mean_branch_embeddings_shape": list(mean_branch_embeddings.shape),
        "weighted_branch_embeddings_shape": list(weighted_branch_embeddings.shape),
        "has_nan_weighted_branch_embeddings": bool(np.isnan(weighted_branch_embeddings).any()),
        "has_inf_weighted_branch_embeddings": bool(np.isinf(weighted_branch_embeddings).any()),
        "aggregation_strategy": "mean branch embedding and norm-weighted deterministic view aggregation",
        "note": (
            "Branch evidence embeddings are generated from initialized validation specialist embeddings. "
            "They are not final trained scientific modality evidence vectors."
        ),
        "outputs_saved_to": str(STAGE_OUT),
    }

    with open(STAGE_OUT / "Stage10F_RGB_Thermal_Evidence_Branches_Summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    report = []
    report.append("# Stage 10F RGB-Centered and Thermal-Auxiliary Evidence Branches\n")
    report.append(f"Generated at: {summary['created_at']}\n")

    report.append("## Purpose\n")
    report.append(
        "This stage aggregates anatomical specialist embeddings into participant-level modality evidence vectors. "
        "For each participant, four RGB specialists are aggregated into one RGB-centered evidence vector, "
        "and four thermal specialists are aggregated into one thermal auxiliary evidence vector.\n"
    )

    report.append("## Aggregation Strategy\n")
    report.append(
        "Two evidence vectors are generated for each modality branch: a mean branch embedding and a deterministic "
        "norm-weighted branch embedding. The norm-weighted version is used only for pipeline validation. "
        "Trainable modality and view attention will be implemented later during full CPMR-Net training.\n"
    )

    report.append("## Summary\n")
    report.append(f"- Input specialist embeddings: {summary['input_specialist_embeddings_count']}")
    report.append(f"- Embedding dimension: {summary['embedding_dim']}")
    report.append(f"- Modality branches: {summary['modality_branches']}")
    report.append(f"- RGB branches: {summary['rgb_branches']}")
    report.append(f"- Thermal branches: {summary['thermal_branches']}")
    report.append(f"- Participants: {summary['participants']}")
    report.append(f"- Participants complete modality pair: {summary['participants_complete_modality_pair']}")
    report.append(f"- Participants incomplete modality pair: {summary['participants_incomplete_modality_pair']}")
    report.append(f"- Complete branches: {summary['complete_branches']}")
    report.append(f"- Incomplete branches: {summary['incomplete_branches']}")
    report.append(f"- RGB weighted embeddings shape: {summary['rgb_weighted_embeddings_shape']}")
    report.append(f"- Thermal weighted embeddings shape: {summary['thermal_weighted_embeddings_shape']}")
    report.append(f"- NaN in weighted branch embeddings: {summary['has_nan_weighted_branch_embeddings']}")
    report.append(f"- Inf in weighted branch embeddings: {summary['has_inf_weighted_branch_embeddings']}\n")

    report.append("## Output Files\n")
    report.append("- `tables/modality_evidence_branch_manifest.csv`")
    report.append("- `tables/rgb_centered_evidence_branch_manifest.csv`")
    report.append("- `tables/thermal_auxiliary_evidence_branch_manifest.csv`")
    report.append("- `tables/modality_view_contribution_weights.csv`")
    report.append("- `tables/participant_modality_branch_map.csv`")
    report.append("- `tables/modality_branch_summary.csv`")
    report.append("- `tables/view_weight_summary_by_modality.csv`")
    report.append("- `embeddings/rgb_centered_weighted_evidence_embeddings.npy`")
    report.append("- `embeddings/thermal_auxiliary_weighted_evidence_embeddings.npy`")
    report.append("- `embeddings/rgb_centered_mean_evidence_embeddings.npy`")
    report.append("- `embeddings/thermal_auxiliary_mean_evidence_embeddings.npy`")
    report.append("- `embeddings/mean_modality_evidence_embeddings.npy`")
    report.append("- `embeddings/weighted_modality_evidence_embeddings.npy`")
    report.append("- `Stage10F_RGB_Thermal_Evidence_Branches_Summary.json`\n")

    report.append("## Important Note\n")
    report.append(summary["note"])

    with open(REPORTS_OUT / "Stage10F_RGB_Thermal_Evidence_Branches_Report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("=" * 80)
    print("STAGE 10F RGB-CENTERED AND THERMAL-AUXILIARY EVIDENCE BRANCHES COMPLETED")
    print("=" * 80)
    print(f"Input specialist embeddings: {summary['input_specialist_embeddings_count']}")
    print(f"Modality branches: {summary['modality_branches']}")
    print(f"RGB branches: {summary['rgb_branches']}")
    print(f"Thermal branches: {summary['thermal_branches']}")
    print(f"Participants: {summary['participants']}")
    print(f"Participants complete modality pair: {summary['participants_complete_modality_pair']}")
    print(f"Participants incomplete modality pair: {summary['participants_incomplete_modality_pair']}")
    print(f"RGB weighted embeddings shape: {rgb_weighted_embeddings.shape}")
    print(f"Thermal weighted embeddings shape: {thermal_weighted_embeddings.shape}")
    print(f"NaN weighted branch embeddings: {summary['has_nan_weighted_branch_embeddings']}")
    print(f"Inf weighted branch embeddings: {summary['has_inf_weighted_branch_embeddings']}")
    print(f"Outputs saved to: {STAGE_OUT}")
    print("=" * 80)
    print("NOTE: These branch evidence vectors are for pipeline validation only, not final trained scientific vectors.")


if __name__ == "__main__":
    main()