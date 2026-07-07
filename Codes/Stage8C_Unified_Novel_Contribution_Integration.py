# -*- coding: utf-8 -*-
"""
Stage 8C - Unified Novel Contribution Integration

Purpose:
Merge three evidence streams into one ranked novelty blueprint:
1. S1-S6 handcrafted empirical findings
2. Stage 7A-7B literature and novelty evidence
3. Conceptual ideas from the current manuscript and The idea document

This stage does not train models.
"""

from pathlib import Path
import json
import pandas as pd
from datetime import datetime

BASE_DIR = Path(r"D:\47\472\New-Papers\Anemia_Paper")
OUTPUTS_DIR = BASE_DIR / "Outputs"

STAGE8A_DIR = OUTPUTS_DIR / "Stage8A_HandcraftedPipeline_KnowledgeBase"
STAGE8B_DIR = OUTPUTS_DIR / "Stage8B_Literature_Novelty_KnowledgeBase"

STAGE_OUT = OUTPUTS_DIR / "Stage8C_Unified_Novel_Contribution_Integration"
TABLES_OUT = STAGE_OUT / "tables"
REPORTS_OUT = STAGE_OUT / "reports"

TABLES_OUT.mkdir(parents=True, exist_ok=True)
REPORTS_OUT.mkdir(parents=True, exist_ok=True)


def safe_read_csv(path):
    path = Path(path)
    if not path.exists():
        return None
    for enc in ["utf-8", "latin1", "cp1252"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return None


def safe_read_json(path):
    path = Path(path)
    if not path.exists():
        return None
    for enc in ["utf-8", "latin1", "cp1252"]:
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------

inputs = {
    "stage8a_json": STAGE8A_DIR / "Stage8A_HandcraftedPipeline_KnowledgeBase.json",
    "stage8a_findings": STAGE8A_DIR / "tables" / "handcrafted_pipeline_interpretive_findings.csv",
    "stage8a_best_models": STAGE8A_DIR / "tables" / "stage6_best_model_records.csv",

    "stage8b_json": STAGE8B_DIR / "Stage8B_Literature_Novelty_KnowledgeBase.json",
    "stage8b_candidate_contributions": STAGE8B_DIR / "tables" / "candidate_novel_contributions.csv",
    "stage8b_novelty_questions": STAGE8B_DIR / "tables" / "novelty_assessment_questions.csv",
    "stage8b_gap_screening": STAGE8B_DIR / "tables" / "literature_gap_keyword_screening.csv",
    "stage8b_high_priority_literature": STAGE8B_DIR / "tables" / "high_priority_literature_evidence.csv",
}


missing_inputs = [str(p) for p in inputs.values() if not p.exists()]

stage8a_findings = safe_read_csv(inputs["stage8a_findings"])
stage8a_best_models = safe_read_csv(inputs["stage8a_best_models"])

stage8b_candidates = safe_read_csv(inputs["stage8b_candidate_contributions"])
stage8b_questions = safe_read_csv(inputs["stage8b_novelty_questions"])
stage8b_gaps = safe_read_csv(inputs["stage8b_gap_screening"])
stage8b_literature = safe_read_csv(inputs["stage8b_high_priority_literature"])

stage8a_json = safe_read_json(inputs["stage8a_json"])
stage8b_json = safe_read_json(inputs["stage8b_json"])


# ---------------------------------------------------------------------
# Unified candidate contribution pool
# ---------------------------------------------------------------------

candidate_pool = [
    {
        "idea_id": "U1",
        "candidate_contribution": "Participant-level diagnostic modeling",
        "core_meaning": "Use the participant, not the individual image, as the diagnostic unit.",
        "source_handcrafted": "S1-S4 verified complete participant-level multimodal structure.",
        "source_literature": "Stage 8B identified participant-level modeling as a key novelty question.",
        "source_conceptual": "The manuscript frames HandDx-200 as a participant-level dataset with multiple views per participant.",
        "empirical_support": 5,
        "literature_gap_support": 4,
        "physiological_support": 5,
        "implementation_feasibility": 5,
        "interpretability_value": 4,
        "risk_level": "Low",
        "recommended_status": "Keep",
    },
    {
        "idea_id": "U2",
        "candidate_contribution": "Multi-view anatomical specialist agents",
        "core_meaning": "Represent left/right palmar and dorsal observations as specialist visual agents.",
        "source_handcrafted": "S4 confirmed four RGB and four thermal views per participant.",
        "source_literature": "Stage 8B identified joint four-view anatomical modeling as a gap.",
        "source_conceptual": "The idea document explicitly proposes anatomical specialist observers.",
        "empirical_support": 4,
        "literature_gap_support": 5,
        "physiological_support": 5,
        "implementation_feasibility": 4,
        "interpretability_value": 5,
        "risk_level": "Low-Medium",
        "recommended_status": "Keep",
    },
    {
        "idea_id": "U3",
        "candidate_contribution": "RGB-centered multimodal architecture",
        "core_meaning": "Treat RGB as the primary anemia signal while retaining thermal evidence as auxiliary information.",
        "source_handcrafted": "S6 showed stronger RGB signal than thermal-only modeling.",
        "source_literature": "Prior anemia imaging studies are frequently RGB/body-site focused; RGB-thermal hand fusion remains less common.",
        "source_conceptual": "The merged framework should reflect empirical signal strength rather than forcing equal modality weighting.",
        "empirical_support": 5,
        "literature_gap_support": 4,
        "physiological_support": 5,
        "implementation_feasibility": 5,
        "interpretability_value": 4,
        "risk_level": "Low",
        "recommended_status": "Keep",
    },
    {
        "idea_id": "U4",
        "candidate_contribution": "Thermal auxiliary physiological branch",
        "core_meaning": "Model thermal images as supportive perfusion/thermoregulatory evidence rather than a standalone dominant classifier.",
        "source_handcrafted": "Thermal images were successfully decoded, but thermal-only performance was weak.",
        "source_literature": "Thermal hand evidence is relatively less represented than RGB, palm, nail, or conjunctiva evidence.",
        "source_conceptual": "The idea document frames thermal information as a physiological manifestation of latent state.",
        "empirical_support": 5,
        "literature_gap_support": 4,
        "physiological_support": 4,
        "implementation_feasibility": 5,
        "interpretability_value": 4,
        "risk_level": "Low",
        "recommended_status": "Keep",
    },
    {
        "idea_id": "U5",
        "candidate_contribution": "Multi-representation image evidence",
        "core_meaning": "Generate multiple complementary representations from each image, including color spaces, texture, patches, enhancement, and saliency.",
        "source_handcrafted": "S5-S6 showed value in color-space, texture, global, and local patch descriptors.",
        "source_literature": "Stage 8B listed multi-representation evidence as a high-priority candidate contribution.",
        "source_conceptual": "The project philosophy states that an image is a source of multiple representations, not the data itself.",
        "empirical_support": 5,
        "literature_gap_support": 4,
        "physiological_support": 4,
        "implementation_feasibility": 4,
        "interpretability_value": 5,
        "risk_level": "Medium",
        "recommended_status": "Keep",
    },
    {
        "idea_id": "U6",
        "candidate_contribution": "Local anatomical evidence prioritization",
        "core_meaning": "Give explicit modeling priority to local anatomical regions because patches outperformed global descriptors.",
        "source_handcrafted": "S6 identified local patches as more informative than global image descriptors.",
        "source_literature": "Anemia literature often focuses on localized pallor regions such as palm, nailbed, conjunctiva, and mucosa.",
        "source_conceptual": "Anatomical specialist reasoning naturally requires local evidence extraction.",
        "empirical_support": 5,
        "literature_gap_support": 4,
        "physiological_support": 5,
        "implementation_feasibility": 5,
        "interpretability_value": 5,
        "risk_level": "Low",
        "recommended_status": "Keep",
    },
    {
        "idea_id": "U7",
        "candidate_contribution": "Adaptive cooperative fusion",
        "core_meaning": "Use learned trust or attention weights across views, modalities, and representations instead of simple concatenation.",
        "source_handcrafted": "S6D6A showed redundancy and performance gains after nonredundant feature selection.",
        "source_literature": "Stage 8B identified adaptive specialist weighting as a novelty question.",
        "source_conceptual": "The idea document proposes adaptive trust and cooperative weighting.",
        "empirical_support": 5,
        "literature_gap_support": 5,
        "physiological_support": 4,
        "implementation_feasibility": 4,
        "interpretability_value": 5,
        "risk_level": "Medium",
        "recommended_status": "Keep",
    },
    {
        "idea_id": "U8",
        "candidate_contribution": "Bilateral symmetry-aware reasoning",
        "core_meaning": "Encourage consistency between left and right corresponding anatomical observations.",
        "source_handcrafted": "S4 confirmed bilateral views for all participants.",
        "source_literature": "Stage 8B identified symmetry reasoning as a potential gap.",
        "source_conceptual": "The idea document proposes symmetry-enforced reasoning from CATS.",
        "empirical_support": 3,
        "literature_gap_support": 5,
        "physiological_support": 5,
        "implementation_feasibility": 3,
        "interpretability_value": 4,
        "risk_level": "Medium",
        "recommended_status": "Keep as secondary contribution",
    },
    {
        "idea_id": "U9",
        "candidate_contribution": "Physiological consistency constraints",
        "core_meaning": "Encourage RGB, thermal, palmar, and dorsal evidence to provide compatible physiological interpretations.",
        "source_handcrafted": "S1-S6 established multimodal evidence streams with unequal predictive value.",
        "source_literature": "Structured physiological fusion is less common than generic CNN fusion.",
        "source_conceptual": "The idea document proposes latent physiological-state reasoning and consistency preservation.",
        "empirical_support": 3,
        "literature_gap_support": 4,
        "physiological_support": 5,
        "implementation_feasibility": 3,
        "interpretability_value": 5,
        "risk_level": "Medium-High",
        "recommended_status": "Keep but implement carefully",
    },
    {
        "idea_id": "U10",
        "candidate_contribution": "Latent physiological-state reasoning",
        "core_meaning": "Interpret visible images as manifestations of hidden anemia-related physiology rather than treating classification as pure pattern recognition.",
        "source_handcrafted": "S6 identified observable color and thermal features associated with anemia status.",
        "source_literature": "Most prior studies optimize diagnosis or Hb prediction without explicit latent-state framing.",
        "source_conceptual": "The idea document proposes inverse latent physiological reconstruction.",
        "empirical_support": 3,
        "literature_gap_support": 4,
        "physiological_support": 5,
        "implementation_feasibility": 2,
        "interpretability_value": 5,
        "risk_level": "High",
        "recommended_status": "Use as conceptual framing, not primary implementation claim",
    },
    {
        "idea_id": "U11",
        "candidate_contribution": "Explainable evidence pathway",
        "core_meaning": "Report contribution weights at representation, view, modality, and participant levels.",
        "source_handcrafted": "Feature ranking and statistical tests already provide interpretable handcrafted evidence.",
        "source_literature": "Stage 8B identified explainability as a novelty question.",
        "source_conceptual": "The idea document emphasizes transparent specialist contributions.",
        "empirical_support": 5,
        "literature_gap_support": 4,
        "physiological_support": 4,
        "implementation_feasibility": 5,
        "interpretability_value": 5,
        "risk_level": "Low",
        "recommended_status": "Keep",
    },
    {
        "idea_id": "U12",
        "candidate_contribution": "Handcrafted-to-learned continuity",
        "core_meaning": "Use S1-S6 handcrafted findings to guide deep-learning design rather than discarding them.",
        "source_handcrafted": "The handcrafted pipeline produced a validated baseline, feature families, and performance ceiling.",
        "source_literature": "Many studies jump directly to CNNs without systematic handcrafted evidence analysis.",
        "source_conceptual": "The merged project philosophy supports evidence-guided architecture design.",
        "empirical_support": 5,
        "literature_gap_support": 4,
        "physiological_support": 4,
        "implementation_feasibility": 5,
        "interpretability_value": 5,
        "risk_level": "Low",
        "recommended_status": "Keep",
    },
    {
        "idea_id": "U13",
        "candidate_contribution": "Benchmark-anchored advancement beyond handcrafted ceiling",
        "core_meaning": "Require every learned architecture to be compared against the best handcrafted baseline.",
        "source_handcrafted": "Stage 8A established S6D6B as the current handcrafted ceiling.",
        "source_literature": "Many prior studies report high performance without strong internal ablation or baseline anchoring.",
        "source_conceptual": "The new framework should demonstrate measurable improvement, not only conceptual novelty.",
        "empirical_support": 5,
        "literature_gap_support": 3,
        "physiological_support": 3,
        "implementation_feasibility": 5,
        "interpretability_value": 4,
        "risk_level": "Low",
        "recommended_status": "Keep",
    },
    {
        "idea_id": "U14",
        "candidate_contribution": "Semantic memory or evidence memory",
        "core_meaning": "Maintain a memory-like representation of distributed diagnostic evidence.",
        "source_handcrafted": "No direct empirical requirement from S1-S6.",
        "source_literature": "Not clearly supported by HandDx-200 literature.",
        "source_conceptual": "The idea document mentions semantic memory from CATS.",
        "empirical_support": 1,
        "literature_gap_support": 2,
        "physiological_support": 2,
        "implementation_feasibility": 2,
        "interpretability_value": 3,
        "risk_level": "High",
        "recommended_status": "Defer",
    },
]


df = pd.DataFrame(candidate_pool)

score_cols = [
    "empirical_support",
    "literature_gap_support",
    "physiological_support",
    "implementation_feasibility",
    "interpretability_value",
]

df["total_support_score"] = df[score_cols].sum(axis=1)
df["mean_support_score"] = df[score_cols].mean(axis=1).round(2)

risk_penalty = {
    "Low": 0,
    "Low-Medium": 0.5,
    "Medium": 1,
    "Medium-High": 1.5,
    "High": 2,
}

df["risk_penalty"] = df["risk_level"].map(risk_penalty).fillna(1)
df["priority_score"] = (df["total_support_score"] - df["risk_penalty"]).round(2)

df = df.sort_values(["priority_score", "total_support_score"], ascending=False)
df.to_csv(TABLES_OUT / "unified_candidate_contributions_ranked.csv", index=False)


# ---------------------------------------------------------------------
# Final contribution grouping
# ---------------------------------------------------------------------

final_core = df[df["recommended_status"].isin(["Keep", "Keep as secondary contribution", "Keep but implement carefully"])].copy()
deferred = df[df["recommended_status"].isin(["Defer"])].copy()

final_core.to_csv(TABLES_OUT / "final_retained_contribution_blueprint.csv", index=False)
deferred.to_csv(TABLES_OUT / "deferred_or_low_priority_ideas.csv", index=False)


# ---------------------------------------------------------------------
# Architecture implication table
# ---------------------------------------------------------------------

architecture_rows = [
    {
        "architecture_component": "Participant-level input container",
        "linked_contributions": "U1, U13",
        "required_inputs": "All RGB and thermal views for each participant",
        "purpose": "Prevent image-level leakage and model the true diagnostic unit."
    },
    {
        "architecture_component": "Multi-representation generator",
        "linked_contributions": "U5, U6, U12",
        "required_inputs": "Original images, color-space conversions, local patches, texture/enhanced images",
        "purpose": "Transform each image into multiple complementary evidence streams."
    },
    {
        "architecture_component": "Anatomical specialist encoders",
        "linked_contributions": "U2, U6, U11",
        "required_inputs": "Left/right palmar and dorsal representations",
        "purpose": "Allow each anatomical view to contribute as an interpretable specialist."
    },
    {
        "architecture_component": "RGB-centered branch",
        "linked_contributions": "U3, U6",
        "required_inputs": "RGB-derived representations",
        "purpose": "Prioritize the empirically stronger anemia-related signal."
    },
    {
        "architecture_component": "Thermal auxiliary branch",
        "linked_contributions": "U4, U9",
        "required_inputs": "Thermal representations",
        "purpose": "Capture complementary perfusion and thermoregulatory information."
    },
    {
        "architecture_component": "Adaptive cooperative fusion",
        "linked_contributions": "U7, U11",
        "required_inputs": "Specialist embeddings and confidence/trust scores",
        "purpose": "Fuse evidence adaptively rather than by naive concatenation."
    },
    {
        "architecture_component": "Symmetry consistency module",
        "linked_contributions": "U8, U9",
        "required_inputs": "Left-right paired anatomical embeddings",
        "purpose": "Encourage physiologically coherent bilateral reasoning."
    },
    {
        "architecture_component": "Explainability output layer",
        "linked_contributions": "U11",
        "required_inputs": "Representation, view, modality, and fusion weights",
        "purpose": "Report how the final decision was formed."
    },
]

architecture_df = pd.DataFrame(architecture_rows)
architecture_df.to_csv(TABLES_OUT / "architecture_implications_from_contributions.csv", index=False)


# ---------------------------------------------------------------------
# Knowledge base JSON
# ---------------------------------------------------------------------

knowledge_base = {
    "stage": "Stage8C",
    "title": "Unified Novel Contribution Integration",
    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "base_dir": str(BASE_DIR),
    "scope": "Integrates Stage8A handcrafted evidence, Stage8B literature novelty evidence, and conceptual architecture ideas.",
    "missing_inputs": missing_inputs,
    "num_candidate_contributions": int(len(df)),
    "num_retained_contributions": int(len(final_core)),
    "num_deferred_ideas": int(len(deferred)),
    "top_contributions": final_core.head(8).to_dict(orient="records"),
    "deferred_ideas": deferred.to_dict(orient="records"),
    "architecture_implications": architecture_rows,
}

with open(STAGE_OUT / "Stage8C_Unified_Novel_Contribution_Integration.json", "w", encoding="utf-8") as f:
    json.dump(knowledge_base, f, indent=4, ensure_ascii=False)


# ---------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------

report = []
report.append("# Stage 8C Unified Novel Contribution Integration\n")
report.append(f"Generated at: {knowledge_base['created_at']}\n")
report.append(f"Base directory: `{BASE_DIR}`\n")

report.append("## Purpose\n")
report.append(
    "This stage merges three evidence streams into one ranked novelty blueprint: "
    "the S1-S6 handcrafted empirical findings, the Stage 7A-7B literature/novelty evidence, "
    "and the conceptual framework proposed in the manuscript and idea documents.\n"
)

report.append("## Input Status\n")
if missing_inputs:
    report.append("The following expected inputs were missing:")
    for p in missing_inputs:
        report.append(f"- `{p}`")
else:
    report.append("All expected Stage 8A and Stage 8B inputs were found.")

report.append("\n## Retained Contributions\n")
for _, row in final_core.iterrows():
    report.append(f"### {row['idea_id']}. {row['candidate_contribution']}")
    report.append(f"**Meaning:** {row['core_meaning']}")
    report.append(f"**Handcrafted support:** {row['source_handcrafted']}")
    report.append(f"**Literature support:** {row['source_literature']}")
    report.append(f"**Conceptual support:** {row['source_conceptual']}")
    report.append(f"**Priority score:** {row['priority_score']}")
    report.append(f"**Recommended status:** {row['recommended_status']}\n")

report.append("## Deferred Ideas\n")
if deferred.empty:
    report.append("No ideas were deferred.")
else:
    for _, row in deferred.iterrows():
        report.append(f"- **{row['idea_id']} {row['candidate_contribution']}**: {row['recommended_status']} because support is currently weaker or implementation risk is high.")

report.append("\n## Architecture Implications\n")
for _, row in architecture_df.iterrows():
    report.append(f"### {row['architecture_component']}")
    report.append(f"- Linked contributions: {row['linked_contributions']}")
    report.append(f"- Required inputs: {row['required_inputs']}")
    report.append(f"- Purpose: {row['purpose']}\n")

report.append("## Generated Output Files\n")
report.append("- `Stage8C_Unified_Novel_Contribution_Integration.json`")
report.append("- `tables/unified_candidate_contributions_ranked.csv`")
report.append("- `tables/final_retained_contribution_blueprint.csv`")
report.append("- `tables/deferred_or_low_priority_ideas.csv`")
report.append("- `tables/architecture_implications_from_contributions.csv`")

with open(REPORTS_OUT / "Stage8C_Unified_Novel_Contribution_Integration_Report.md", "w", encoding="utf-8") as f:
    f.write("\n".join(report))


print("=" * 80)
print("STAGE 8C UNIFIED NOVEL CONTRIBUTION INTEGRATION COMPLETED")
print("=" * 80)
print(f"Candidate contributions: {len(df)}")
print(f"Retained contributions: {len(final_core)}")
print(f"Deferred ideas: {len(deferred)}")
print(f"Missing inputs: {len(missing_inputs)}")
print(f"Outputs saved to: {STAGE_OUT}")
print("=" * 80)

if missing_inputs:
    print("Missing inputs:")
    for p in missing_inputs:
        print(p)