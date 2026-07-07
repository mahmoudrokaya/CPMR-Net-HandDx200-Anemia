# -*- coding: utf-8 -*-
"""
Stage 9A - Architecture Decision Matrix

Answers Phase B questions:
1. What are the actual novelties?
2. Which ideas complement each other?
3. Which ideas are redundant?
4. Which ideas are implementable?
5. Which ideas are publishable?
"""

from pathlib import Path
import json
import pandas as pd
from datetime import datetime

BASE_DIR = Path(r"D:\47\472\New-Papers\Anemia_Paper")
OUTPUTS_DIR = BASE_DIR / "Outputs"

STAGE8C_DIR = OUTPUTS_DIR / "Stage8C_Unified_Novel_Contribution_Integration"

STAGE_OUT = OUTPUTS_DIR / "Stage9A_Architecture_Decision_Matrix"
TABLES_OUT = STAGE_OUT / "tables"
REPORTS_OUT = STAGE_OUT / "reports"

TABLES_OUT.mkdir(parents=True, exist_ok=True)
REPORTS_OUT.mkdir(parents=True, exist_ok=True)


def safe_read_csv(path):
    for enc in ["utf-8", "latin1", "cp1252"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return None


input_file = STAGE8C_DIR / "tables" / "unified_candidate_contributions_ranked.csv"

df = safe_read_csv(input_file)

if df is None:
    raise FileNotFoundError(f"Could not read input file: {input_file}")


# ---------------------------------------------------------------------
# 1. Actual novelties
# ---------------------------------------------------------------------

actual_novelties = [
    {
        "novelty_id": "N1",
        "actual_novelty": "Participant-level multi-view anemia reasoning",
        "merged_ideas": "U1, U2, U13",
        "why_it_is_novel": "The model treats the participant as the diagnostic unit and jointly uses all anatomical views rather than classifying isolated images.",
        "publishability": "High",
        "implementation_status": "Core"
    },
    {
        "novelty_id": "N2",
        "actual_novelty": "RGB-centered multimodal physiological fusion",
        "merged_ideas": "U3, U4, U9",
        "why_it_is_novel": "RGB is modeled as the primary anemia signal while thermal imaging is retained as auxiliary perfusion evidence rather than being fused equally.",
        "publishability": "High",
        "implementation_status": "Core"
    },
    {
        "novelty_id": "N3",
        "actual_novelty": "Multi-representation anatomical evidence learning",
        "merged_ideas": "U5, U6, U12",
        "why_it_is_novel": "Each image is treated as a source of multiple representations, including color, texture, enhancement, and local anatomical evidence.",
        "publishability": "High",
        "implementation_status": "Core"
    },
    {
        "novelty_id": "N4",
        "actual_novelty": "Adaptive cooperative specialist fusion",
        "merged_ideas": "U2, U7, U11",
        "why_it_is_novel": "Anatomical specialists are fused through learned trust or attention weights rather than simple concatenation.",
        "publishability": "High",
        "implementation_status": "Core"
    },
    {
        "novelty_id": "N5",
        "actual_novelty": "Explainable evidence pathway",
        "merged_ideas": "U7, U11",
        "why_it_is_novel": "The system reports contribution weights at representation, view, modality, and participant levels.",
        "publishability": "High",
        "implementation_status": "Core"
    },
    {
        "novelty_id": "N6",
        "actual_novelty": "Bilateral physiological consistency",
        "merged_ideas": "U8, U9",
        "why_it_is_novel": "The model encourages left-right coherence because anemia is systemic rather than localized.",
        "publishability": "Moderate-High",
        "implementation_status": "Secondary"
    }
]

actual_novelties_df = pd.DataFrame(actual_novelties)
actual_novelties_df.to_csv(TABLES_OUT / "actual_novelties.csv", index=False)


# ---------------------------------------------------------------------
# 2. Complementarity map
# ---------------------------------------------------------------------

complementarity = [
    {
        "group_id": "G1",
        "group_name": "Participant-level reasoning group",
        "ideas": "U1, U2, U13",
        "complementarity": "U1 defines the diagnostic unit, U2 defines view-level specialists, and U13 defines the benchmark requirement.",
        "architecture_role": "Input organization and evaluation protocol"
    },
    {
        "group_id": "G2",
        "group_name": "Representation learning group",
        "ideas": "U5, U6, U12",
        "complementarity": "U5 creates multiple representations, U6 prioritizes local anatomical evidence, and U12 links learned design to handcrafted findings.",
        "architecture_role": "Representation generator and anatomical feature learning"
    },
    {
        "group_id": "G3",
        "group_name": "Multimodal physiological group",
        "ideas": "U3, U4, U9",
        "complementarity": "U3 gives RGB dominance, U4 gives thermal auxiliary role, and U9 adds physiological consistency.",
        "architecture_role": "RGB-thermal fusion design"
    },
    {
        "group_id": "G4",
        "group_name": "Cooperative fusion group",
        "ideas": "U2, U7, U11",
        "complementarity": "U2 creates specialists, U7 fuses them adaptively, and U11 makes the cooperation explainable.",
        "architecture_role": "Specialist cooperation and explainability"
    },
    {
        "group_id": "G5",
        "group_name": "Physiological regularization group",
        "ideas": "U8, U9",
        "complementarity": "U8 enforces bilateral consistency and U9 enforces cross-modality physiological compatibility.",
        "architecture_role": "Secondary regularization and robustness"
    }
]

complementarity_df = pd.DataFrame(complementarity)
complementarity_df.to_csv(TABLES_OUT / "complementarity_map.csv", index=False)


# ---------------------------------------------------------------------
# 3. Redundancy analysis
# ---------------------------------------------------------------------

redundancy = [
    {
        "redundancy_id": "R1",
        "overlapping_ideas": "U1 and U2",
        "overlap": "Both relate to multi-view participant structure.",
        "decision": "Not redundant",
        "reason": "U1 defines the diagnostic unit; U2 defines the internal anatomical-specialist organization."
    },
    {
        "redundancy_id": "R2",
        "overlapping_ideas": "U3 and U4",
        "overlap": "Both relate to modality roles.",
        "decision": "Not redundant",
        "reason": "U3 defines RGB dominance; U4 defines the thermal branch as auxiliary physiological evidence."
    },
    {
        "redundancy_id": "R3",
        "overlapping_ideas": "U5 and U6",
        "overlap": "Both relate to image evidence extraction.",
        "decision": "Complementary, merge in narrative",
        "reason": "U5 is multi-representation learning; U6 is local anatomical prioritization."
    },
    {
        "redundancy_id": "R4",
        "overlapping_ideas": "U7 and U11",
        "overlap": "Both relate to fusion weights.",
        "decision": "Complementary",
        "reason": "U7 uses weights for adaptive fusion; U11 exposes them for explainability."
    },
    {
        "redundancy_id": "R5",
        "overlapping_ideas": "U8 and U9",
        "overlap": "Both impose physiological structure.",
        "decision": "Partially redundant",
        "reason": "U8 should be implemented as a specific case of U9, focused on bilateral consistency."
    },
    {
        "redundancy_id": "R6",
        "overlapping_ideas": "U10 and U9",
        "overlap": "Both relate to latent physiological interpretation.",
        "decision": "Reduce U10 to conceptual framing",
        "reason": "U10 is difficult to implement directly without hemoglobin or physiological labels."
    },
    {
        "redundancy_id": "R7",
        "overlapping_ideas": "U14 and U7/U11",
        "overlap": "Memory may duplicate adaptive evidence accumulation.",
        "decision": "Defer",
        "reason": "Semantic memory has weak empirical support and high implementation risk."
    }
]

redundancy_df = pd.DataFrame(redundancy)
redundancy_df.to_csv(TABLES_OUT / "redundancy_analysis.csv", index=False)


# ---------------------------------------------------------------------
# 4. Implementability and publishability
# ---------------------------------------------------------------------

implementation = [
    {
        "component": "Participant-level input container",
        "linked_ideas": "U1, U13",
        "implementability": "High",
        "publishability": "High",
        "decision": "Implement first"
    },
    {
        "component": "Multi-representation generator",
        "linked_ideas": "U5, U6, U12",
        "implementability": "High",
        "publishability": "High",
        "decision": "Implement first"
    },
    {
        "component": "Anatomical specialist encoders",
        "linked_ideas": "U2, U6",
        "implementability": "High",
        "publishability": "High",
        "decision": "Implement first"
    },
    {
        "component": "RGB-centered branch",
        "linked_ideas": "U3",
        "implementability": "High",
        "publishability": "High",
        "decision": "Implement first"
    },
    {
        "component": "Thermal auxiliary branch",
        "linked_ideas": "U4",
        "implementability": "High",
        "publishability": "Moderate-High",
        "decision": "Implement, but keep auxiliary"
    },
    {
        "component": "Adaptive cooperative fusion",
        "linked_ideas": "U7, U11",
        "implementability": "Moderate-High",
        "publishability": "High",
        "decision": "Implement after baseline branches"
    },
    {
        "component": "Bilateral symmetry consistency",
        "linked_ideas": "U8, U9",
        "implementability": "Moderate",
        "publishability": "Moderate-High",
        "decision": "Implement as ablation/regularization"
    },
    {
        "component": "Latent physiological-state framing",
        "linked_ideas": "U10",
        "implementability": "Low",
        "publishability": "Moderate",
        "decision": "Use in discussion, not as primary model claim"
    },
    {
        "component": "Semantic memory",
        "linked_ideas": "U14",
        "implementability": "Low",
        "publishability": "Low-Medium",
        "decision": "Do not implement in this paper"
    }
]

implementation_df = pd.DataFrame(implementation)
implementation_df.to_csv(TABLES_OUT / "implementability_publishability_matrix.csv", index=False)


# ---------------------------------------------------------------------
# Final architecture decision
# ---------------------------------------------------------------------

final_architecture_decisions = [
    {
        "decision_id": "D1",
        "decision": "The final framework should be participant-level, not image-level.",
        "justification": "This is supported by the dataset structure and avoids leakage."
    },
    {
        "decision_id": "D2",
        "decision": "The model should be RGB-centered with thermal as an auxiliary physiological branch.",
        "justification": "S1-S6 showed stronger RGB evidence and weak standalone thermal performance."
    },
    {
        "decision_id": "D3",
        "decision": "Each image should generate multiple representations.",
        "justification": "Handcrafted evidence showed value in color, texture, and local descriptors."
    },
    {
        "decision_id": "D4",
        "decision": "The architecture should use anatomical specialist encoders.",
        "justification": "This converts four-view hand acquisition into interpretable diagnostic observers."
    },
    {
        "decision_id": "D5",
        "decision": "Fusion should be adaptive and explainable.",
        "justification": "Naive concatenation conflicts with redundancy findings and weakens interpretability."
    },
    {
        "decision_id": "D6",
        "decision": "Bilateral symmetry should be tested as a secondary regularization module.",
        "justification": "It is physiologically meaningful but should not be the primary novelty claim."
    },
    {
        "decision_id": "D7",
        "decision": "Semantic memory should be excluded from the first architecture.",
        "justification": "It is not sufficiently supported by the dataset or literature evidence."
    }
]

final_decisions_df = pd.DataFrame(final_architecture_decisions)
final_decisions_df.to_csv(TABLES_OUT / "final_architecture_decisions.csv", index=False)


# ---------------------------------------------------------------------
# JSON + report
# ---------------------------------------------------------------------

knowledge_base = {
    "stage": "Stage9A",
    "title": "Architecture Decision Matrix",
    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "input_file": str(input_file),
    "actual_novelties": actual_novelties,
    "complementarity_map": complementarity,
    "redundancy_analysis": redundancy,
    "implementation_publishability": implementation,
    "final_architecture_decisions": final_architecture_decisions
}

with open(STAGE_OUT / "Stage9A_Architecture_Decision_Matrix.json", "w", encoding="utf-8") as f:
    json.dump(knowledge_base, f, indent=4, ensure_ascii=False)


report = []
report.append("# Stage 9A Architecture Decision Matrix\n")
report.append(f"Generated at: {knowledge_base['created_at']}\n")

report.append("## Question 1: What are the actual novelties?\n")
for item in actual_novelties:
    report.append(f"### {item['novelty_id']}. {item['actual_novelty']}")
    report.append(f"- Merged ideas: {item['merged_ideas']}")
    report.append(f"- Why novel: {item['why_it_is_novel']}")
    report.append(f"- Publishability: {item['publishability']}")
    report.append(f"- Implementation status: {item['implementation_status']}\n")

report.append("## Question 2: Which ideas complement each other?\n")
for item in complementarity:
    report.append(f"### {item['group_id']}. {item['group_name']}")
    report.append(f"- Ideas: {item['ideas']}")
    report.append(f"- Complementarity: {item['complementarity']}")
    report.append(f"- Architecture role: {item['architecture_role']}\n")

report.append("## Question 3: Which ideas are redundant?\n")
for item in redundancy:
    report.append(f"### {item['redundancy_id']}. {item['overlapping_ideas']}")
    report.append(f"- Overlap: {item['overlap']}")
    report.append(f"- Decision: {item['decision']}")
    report.append(f"- Reason: {item['reason']}\n")

report.append("## Question 4 and 5: Which ideas are implementable and publishable?\n")
for item in implementation:
    report.append(f"### {item['component']}")
    report.append(f"- Linked ideas: {item['linked_ideas']}")
    report.append(f"- Implementability: {item['implementability']}")
    report.append(f"- Publishability: {item['publishability']}")
    report.append(f"- Decision: {item['decision']}\n")

report.append("## Final Architecture Decisions\n")
for item in final_architecture_decisions:
    report.append(f"### {item['decision_id']}")
    report.append(f"**Decision:** {item['decision']}")
    report.append(f"**Justification:** {item['justification']}\n")

report.append("## Generated Output Files\n")
report.append("- `Stage9A_Architecture_Decision_Matrix.json`")
report.append("- `tables/actual_novelties.csv`")
report.append("- `tables/complementarity_map.csv`")
report.append("- `tables/redundancy_analysis.csv`")
report.append("- `tables/implementability_publishability_matrix.csv`")
report.append("- `tables/final_architecture_decisions.csv`")

with open(REPORTS_OUT / "Stage9A_Architecture_Decision_Matrix_Report.md", "w", encoding="utf-8") as f:
    f.write("\n".join(report))


print("=" * 80)
print("STAGE 9A ARCHITECTURE DECISION MATRIX COMPLETED")
print("=" * 80)
print(f"Actual novelties: {len(actual_novelties)}")
print(f"Complementarity groups: {len(complementarity)}")
print(f"Redundancy decisions: {len(redundancy)}")
print(f"Implementation decisions: {len(implementation)}")
print(f"Final architecture decisions: {len(final_architecture_decisions)}")
print(f"Outputs saved to: {STAGE_OUT}")
print("=" * 80)