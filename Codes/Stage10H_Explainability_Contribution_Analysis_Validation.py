# -*- coding: utf-8 -*-
"""
Stage 10H - Explainability and Contribution Analysis Validation

Purpose:
Validate CPMR-Net explainability outputs before supervised training.

This stage consolidates contribution information from:
- Stage 10E representation-to-specialist weights
- Stage 10F view-to-modality weights
- Stage 10G modality-to-fusion weights

It produces participant-level, modality-level, view-level, and representation-level
explainability reports.

Important:
These are validation-stage deterministic weights, not final trained attention weights.
"""

from pathlib import Path
import json
from datetime import datetime

import pandas as pd
import numpy as np


BASE_DIR = Path(r"D:\47\472\New-Papers\Anemia_Paper")
OUTPUTS_DIR = BASE_DIR / "Outputs"

STAGE10E_DIR = OUTPUTS_DIR / "Stage10E_Anatomical_Specialist_Embedding_Aggregation"
STAGE10F_DIR = OUTPUTS_DIR / "Stage10F_RGB_Thermal_Evidence_Branches"
STAGE10G_DIR = OUTPUTS_DIR / "Stage10G_Adaptive_Cooperative_Fusion_Validation"

SPECIALIST_REP_WEIGHTS = STAGE10E_DIR / "tables" / "specialist_representation_weights.csv"
VIEW_WEIGHTS = STAGE10F_DIR / "tables" / "modality_view_contribution_weights.csv"
MODALITY_WEIGHTS = STAGE10G_DIR / "tables" / "modality_contribution_weights.csv"
FUSION_MANIFEST = STAGE10G_DIR / "tables" / "adaptive_cooperative_fusion_manifest.csv"

STAGE_OUT = OUTPUTS_DIR / "Stage10H_Explainability_Contribution_Analysis_Validation"
TABLES_OUT = STAGE_OUT / "tables"
REPORTS_OUT = STAGE_OUT / "reports"

TABLES_OUT.mkdir(parents=True, exist_ok=True)
REPORTS_OUT.mkdir(parents=True, exist_ok=True)


def require_file(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing required input: {path}")


def normalize_group_weights(df, group_cols, weight_col, new_col):
    out = df.copy()
    sums = out.groupby(group_cols)[weight_col].transform("sum")
    out[new_col] = np.where(sums > 0, out[weight_col] / sums, 0)
    return out


def main():
    for p in [SPECIALIST_REP_WEIGHTS, VIEW_WEIGHTS, MODALITY_WEIGHTS, FUSION_MANIFEST]:
        require_file(p)

    rep_weights = pd.read_csv(SPECIALIST_REP_WEIGHTS)
    view_weights = pd.read_csv(VIEW_WEIGHTS)
    modality_weights = pd.read_csv(MODALITY_WEIGHTS)
    fusion_manifest = pd.read_csv(FUSION_MANIFEST)

    # -----------------------------------------------------------------
    # Normalize all validation weights to ensure interpretability sanity
    # -----------------------------------------------------------------

    rep_weights = normalize_group_weights(
        rep_weights,
        ["participant_id", "modality", "view"],
        "representation_weight",
        "representation_weight_normalized"
    )

    view_weights = normalize_group_weights(
        view_weights,
        ["participant_id", "modality"],
        "view_weight",
        "view_weight_normalized"
    )

    modality_weights = normalize_group_weights(
        modality_weights,
        ["participant_id"],
        "rgb_centered_weight",
        "modality_weight_rgb_centered_normalized"
    )

    modality_weights = normalize_group_weights(
        modality_weights,
        ["participant_id"],
        "norm_based_weight",
        "modality_weight_norm_based_normalized"
    )

    # -----------------------------------------------------------------
    # Representation-level explainability
    # -----------------------------------------------------------------

    representation_summary = (
        rep_weights.groupby(["modality", "view", "representation"])
        .agg(
            records=("representation_weight_normalized", "count"),
            mean_weight=("representation_weight_normalized", "mean"),
            std_weight=("representation_weight_normalized", "std"),
            mean_representation_norm=("representation_norm", "mean")
        )
        .reset_index()
        .sort_values(["modality", "view", "mean_weight"], ascending=[True, True, False])
    )

    representation_summary.to_csv(TABLES_OUT / "representation_level_contribution_summary.csv", index=False)

    participant_representation_summary = (
        rep_weights.groupby(["participant_id", "label", "modality", "representation"])
        .agg(
            total_representation_weight=("representation_weight_normalized", "sum"),
            mean_representation_weight=("representation_weight_normalized", "mean"),
            records=("representation_weight_normalized", "count")
        )
        .reset_index()
    )

    participant_representation_summary.to_csv(TABLES_OUT / "participant_representation_contributions.csv", index=False)

    # -----------------------------------------------------------------
    # View-level explainability
    # -----------------------------------------------------------------

    view_summary = (
        view_weights.groupby(["modality", "view"])
        .agg(
            records=("view_weight_normalized", "count"),
            mean_view_weight=("view_weight_normalized", "mean"),
            std_view_weight=("view_weight_normalized", "std"),
            mean_specialist_norm=("specialist_norm", "mean")
        )
        .reset_index()
        .sort_values(["modality", "mean_view_weight"], ascending=[True, False])
    )

    view_summary.to_csv(TABLES_OUT / "view_level_contribution_summary.csv", index=False)

    participant_view_summary = (
        view_weights.groupby(["participant_id", "label", "modality", "view"])
        .agg(
            view_weight=("view_weight_normalized", "mean"),
            specialist_norm=("specialist_norm", "mean")
        )
        .reset_index()
    )

    participant_view_summary.to_csv(TABLES_OUT / "participant_view_contributions.csv", index=False)

    # -----------------------------------------------------------------
    # Modality-level explainability
    # -----------------------------------------------------------------

    modality_summary = (
        modality_weights.groupby("modality")
        .agg(
            records=("modality", "count"),
            mean_norm=("norm", "mean"),
            std_norm=("norm", "std"),
            mean_norm_based_weight=("modality_weight_norm_based_normalized", "mean"),
            std_norm_based_weight=("modality_weight_norm_based_normalized", "std"),
            mean_rgb_centered_weight=("modality_weight_rgb_centered_normalized", "mean"),
            std_rgb_centered_weight=("modality_weight_rgb_centered_normalized", "std")
        )
        .reset_index()
    )

    modality_summary.to_csv(TABLES_OUT / "modality_level_contribution_summary.csv", index=False)

    participant_modality_summary = modality_weights[
        [
            "participant_id",
            "label",
            "modality",
            "norm",
            "modality_weight_norm_based_normalized",
            "modality_weight_rgb_centered_normalized",
        ]
    ].copy()

    participant_modality_summary.to_csv(TABLES_OUT / "participant_modality_contributions.csv", index=False)

    # -----------------------------------------------------------------
    # Full hierarchical contribution table
    # modality weight × view weight × representation weight
    # -----------------------------------------------------------------

    rep_view = rep_weights.merge(
        view_weights[
            [
                "participant_id",
                "label",
                "modality",
                "view",
                "view_weight_normalized",
                "specialist_norm",
            ]
        ],
        on=["participant_id", "label", "modality", "view"],
        how="left"
    )

    full_hierarchy_rgb_centered = rep_view.merge(
        modality_weights[
            [
                "participant_id",
                "label",
                "modality",
                "modality_weight_rgb_centered_normalized",
                "modality_weight_norm_based_normalized",
            ]
        ],
        on=["participant_id", "label", "modality"],
        how="left"
    )

    full_hierarchy_rgb_centered["hierarchical_weight_rgb_centered"] = (
        full_hierarchy_rgb_centered["modality_weight_rgb_centered_normalized"]
        * full_hierarchy_rgb_centered["view_weight_normalized"]
        * full_hierarchy_rgb_centered["representation_weight_normalized"]
    )

    full_hierarchy_rgb_centered["hierarchical_weight_norm_based"] = (
        full_hierarchy_rgb_centered["modality_weight_norm_based_normalized"]
        * full_hierarchy_rgb_centered["view_weight_normalized"]
        * full_hierarchy_rgb_centered["representation_weight_normalized"]
    )

    full_hierarchy_rgb_centered.to_csv(
        TABLES_OUT / "full_hierarchical_contribution_table.csv",
        index=False
    )

    # -----------------------------------------------------------------
    # Participant-level explainability profile
    # -----------------------------------------------------------------

    participant_profiles = []

    for participant_id, p_df in full_hierarchy_rgb_centered.groupby("participant_id"):
        label = int(p_df["label"].iloc[0])

        top_rgb_centered = p_df.sort_values(
            "hierarchical_weight_rgb_centered",
            ascending=False
        ).iloc[0]

        top_norm_based = p_df.sort_values(
            "hierarchical_weight_norm_based",
            ascending=False
        ).iloc[0]

        rgb_modality_weight = modality_weights[
            (modality_weights["participant_id"] == participant_id)
            & (modality_weights["modality"] == "rgb")
        ]["modality_weight_rgb_centered_normalized"].iloc[0]

        thermal_modality_weight = modality_weights[
            (modality_weights["participant_id"] == participant_id)
            & (modality_weights["modality"] == "thermal")
        ]["modality_weight_rgb_centered_normalized"].iloc[0]

        participant_profiles.append({
            "participant_id": participant_id,
            "label": label,
            "rgb_centered_rgb_weight": float(rgb_modality_weight),
            "rgb_centered_thermal_weight": float(thermal_modality_weight),
            "top_rgb_centered_modality": top_rgb_centered["modality"],
            "top_rgb_centered_view": top_rgb_centered["view"],
            "top_rgb_centered_representation": top_rgb_centered["representation"],
            "top_rgb_centered_hierarchical_weight": float(top_rgb_centered["hierarchical_weight_rgb_centered"]),
            "top_norm_based_modality": top_norm_based["modality"],
            "top_norm_based_view": top_norm_based["view"],
            "top_norm_based_representation": top_norm_based["representation"],
            "top_norm_based_hierarchical_weight": float(top_norm_based["hierarchical_weight_norm_based"]),
        })

    participant_profiles_df = pd.DataFrame(participant_profiles)
    participant_profiles_df.to_csv(TABLES_OUT / "participant_explainability_profiles.csv", index=False)

    # -----------------------------------------------------------------
    # Completeness and weight sanity checks
    # -----------------------------------------------------------------

    rep_sum_check = (
        rep_weights.groupby(["participant_id", "modality", "view"])
        ["representation_weight_normalized"]
        .sum()
        .reset_index(name="representation_weight_sum")
    )

    view_sum_check = (
        view_weights.groupby(["participant_id", "modality"])
        ["view_weight_normalized"]
        .sum()
        .reset_index(name="view_weight_sum")
    )

    modality_sum_check_rgb_centered = (
        modality_weights.groupby("participant_id")
        ["modality_weight_rgb_centered_normalized"]
        .sum()
        .reset_index(name="rgb_centered_modality_weight_sum")
    )

    modality_sum_check_norm_based = (
        modality_weights.groupby("participant_id")
        ["modality_weight_norm_based_normalized"]
        .sum()
        .reset_index(name="norm_based_modality_weight_sum")
    )

    sanity_checks = {
        "representation_weight_sums_min": float(rep_sum_check["representation_weight_sum"].min()),
        "representation_weight_sums_max": float(rep_sum_check["representation_weight_sum"].max()),
        "view_weight_sums_min": float(view_sum_check["view_weight_sum"].min()),
        "view_weight_sums_max": float(view_sum_check["view_weight_sum"].max()),
        "rgb_centered_modality_weight_sums_min": float(modality_sum_check_rgb_centered["rgb_centered_modality_weight_sum"].min()),
        "rgb_centered_modality_weight_sums_max": float(modality_sum_check_rgb_centered["rgb_centered_modality_weight_sum"].max()),
        "norm_based_modality_weight_sums_min": float(modality_sum_check_norm_based["norm_based_modality_weight_sum"].min()),
        "norm_based_modality_weight_sums_max": float(modality_sum_check_norm_based["norm_based_modality_weight_sum"].max()),
    }

    pd.DataFrame([sanity_checks]).to_csv(TABLES_OUT / "explainability_weight_sanity_checks.csv", index=False)

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    summary = {
        "stage": "Stage10H",
        "title": "Explainability and Contribution Analysis Validation",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "participants": int(participant_profiles_df["participant_id"].nunique()),
        "representation_weight_records": int(len(rep_weights)),
        "view_weight_records": int(len(view_weights)),
        "modality_weight_records": int(len(modality_weights)),
        "hierarchical_contribution_records": int(len(full_hierarchy_rgb_centered)),
        "participant_profiles": int(len(participant_profiles_df)),
        "sanity_checks": sanity_checks,
        "has_missing_hierarchical_weights": bool(
            full_hierarchy_rgb_centered[
                [
                    "hierarchical_weight_rgb_centered",
                    "hierarchical_weight_norm_based"
                ]
            ].isna().any().any()
        ),
        "note": (
            "This stage validates contribution-report generation using deterministic validation weights. "
            "Final explainability outputs will be generated from trained CPMR-Net attention/fusion weights."
        ),
        "outputs_saved_to": str(STAGE_OUT)
    }

    with open(STAGE_OUT / "Stage10H_Explainability_Contribution_Analysis_Validation_Summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    report = []
    report.append("# Stage 10H Explainability and Contribution Analysis Validation\n")
    report.append(f"Generated at: {summary['created_at']}\n")

    report.append("## Purpose\n")
    report.append(
        "This stage validates the explainability pathway of CPMR-Net by consolidating representation-level, "
        "view-level, and modality-level contribution weights into hierarchical participant-level contribution reports. "
        "The current weights are deterministic validation weights, not final trained attention weights.\n"
    )

    report.append("## Explainability Levels\n")
    report.append("- Representation level: contribution of RGB, HSV, LAB, texture, thermal maps, and patches.")
    report.append("- View level: contribution of left/right dorsal and palmar anatomical specialists.")
    report.append("- Modality level: contribution of RGB-centered and thermal-auxiliary branches.")
    report.append("- Participant level: top hierarchical evidence pathway per participant.\n")

    report.append("## Summary\n")
    report.append(f"- Participants: {summary['participants']}")
    report.append(f"- Representation weight records: {summary['representation_weight_records']}")
    report.append(f"- View weight records: {summary['view_weight_records']}")
    report.append(f"- Modality weight records: {summary['modality_weight_records']}")
    report.append(f"- Hierarchical contribution records: {summary['hierarchical_contribution_records']}")
    report.append(f"- Participant explainability profiles: {summary['participant_profiles']}")
    report.append(f"- Missing hierarchical weights: {summary['has_missing_hierarchical_weights']}\n")

    report.append("## Weight Sanity Checks\n")
    for k, v in sanity_checks.items():
        report.append(f"- {k}: {v}")

    report.append("\n## Output Files\n")
    report.append("- `representation_level_contribution_summary.csv`")
    report.append("- `participant_representation_contributions.csv`")
    report.append("- `view_level_contribution_summary.csv`")
    report.append("- `participant_view_contributions.csv`")
    report.append("- `modality_level_contribution_summary.csv`")
    report.append("- `participant_modality_contributions.csv`")
    report.append("- `full_hierarchical_contribution_table.csv`")
    report.append("- `participant_explainability_profiles.csv`")
    report.append("- `explainability_weight_sanity_checks.csv`")
    report.append("- `Stage10H_Explainability_Contribution_Analysis_Validation_Summary.json`\n")

    report.append("## Important Note\n")
    report.append(summary["note"])

    with open(REPORTS_OUT / "Stage10H_Explainability_Contribution_Analysis_Validation_Report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("=" * 80)
    print("STAGE 10H EXPLAINABILITY AND CONTRIBUTION ANALYSIS VALIDATION COMPLETED")
    print("=" * 80)
    print(f"Participants: {summary['participants']}")
    print(f"Representation weight records: {summary['representation_weight_records']}")
    print(f"View weight records: {summary['view_weight_records']}")
    print(f"Modality weight records: {summary['modality_weight_records']}")
    print(f"Hierarchical contribution records: {summary['hierarchical_contribution_records']}")
    print(f"Participant profiles: {summary['participant_profiles']}")
    print(f"Missing hierarchical weights: {summary['has_missing_hierarchical_weights']}")
    print(f"Outputs saved to: {STAGE_OUT}")
    print("=" * 80)
    print("NOTE: These are validation explainability outputs, not final trained CPMR-Net explanations.")


if __name__ == "__main__":
    main()