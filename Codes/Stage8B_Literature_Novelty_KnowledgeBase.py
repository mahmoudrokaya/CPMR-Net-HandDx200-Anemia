# -*- coding: utf-8 -*-
r"""
Stage 8B - Literature and Novelty Knowledge Base

Consolidates Stage 7A and Stage 7B literature outputs into a structured
knowledge base for novelty extraction and architecture design.
"""

from pathlib import Path
import json
import pandas as pd
from datetime import datetime

BASE_DIR = Path(r"D:\47\472\New-Papers\Anemia_Paper")
OUTPUTS_DIR = BASE_DIR / "Outputs"

PAPERS_DIR = OUTPUTS_DIR / "Papers"
STAGE7B_DIR = PAPERS_DIR / "Stage7B_Filtered_Novelty_Map"

STAGE_OUT = OUTPUTS_DIR / "Stage8B_Literature_Novelty_KnowledgeBase"
TABLES_OUT = STAGE_OUT / "tables"
REPORTS_OUT = STAGE_OUT / "reports"

TABLES_OUT.mkdir(parents=True, exist_ok=True)
REPORTS_OUT.mkdir(parents=True, exist_ok=True)


def safe_read_csv(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_csv(path, encoding="latin1")
        except Exception:
            return None


def safe_read_text(path):
    path = Path(path)
    if not path.exists():
        return ""
    for enc in ["utf-8", "latin1", "cp1252"]:
        try:
            return path.read_text(encoding=enc)
        except Exception:
            continue
    return ""


def table_summary(path, name):
    df = safe_read_csv(path)
    if df is None:
        return {
            "name": name,
            "path": str(path),
            "exists": False,
            "rows": None,
            "columns": None,
            "column_names": []
        }
    return {
        "name": name,
        "path": str(path),
        "exists": True,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns)
    }


# ---------------------------------------------------------------------
# Stage 7A / 7B expected inputs
# ---------------------------------------------------------------------

input_files = {
    "stage7a_records_csv": PAPERS_DIR / "tables" / "collected_literature_records.csv",
    "stage7a_records_xlsx": PAPERS_DIR / "tables" / "collected_literature_records.xlsx",
    "stage7a_report": PAPERS_DIR / "Stage7A_LiteratureCrawler_Report.md",

    "stage7b_all_scored": STAGE7B_DIR / "tables" / "Stage7B_all_records_scored.csv",
    "stage7b_priority_records": STAGE7B_DIR / "tables" / "Stage7B_priority_records.csv",
    "stage7b_direct_hand_records": STAGE7B_DIR / "tables" / "Stage7B_direct_hand_related_records.csv",
    "stage7b_reviews_datasets": STAGE7B_DIR / "tables" / "Stage7B_reviews_and_datasets.csv",
    "stage7b_excluded": STAGE7B_DIR / "tables" / "Stage7B_excluded_low_relevance_records.csv",
    "stage7b_category_summary": STAGE7B_DIR / "tables" / "Stage7B_category_summary.csv",
    "stage7b_report_md": STAGE7B_DIR / "reports" / "Stage7B_Filtered_Novelty_Map_Report.md",
    "stage7b_report_docx": STAGE7B_DIR / "reports" / "Stage7B_Filtered_Novelty_Map.docx",
}


knowledge_base = {
    "stage": "Stage8B",
    "title": "Literature and Novelty Knowledge Base",
    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "base_dir": str(BASE_DIR),
    "scope": "Stage 7A literature crawler and Stage 7B filtered novelty map",
    "input_file_inventory": [],
    "missing_files": [],
    "literature_statistics": {},
    "novelty_questions": [],
    "candidate_novel_contributions": []
}


# ---------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------

for name, path in input_files.items():
    item = table_summary(path, name) if path.suffix.lower() == ".csv" else {
        "name": name,
        "path": str(path),
        "exists": path.exists(),
        "rows": None,
        "columns": None,
        "column_names": []
    }

    knowledge_base["input_file_inventory"].append(item)

    if not path.exists():
        knowledge_base["missing_files"].append(str(path))


inventory_df = pd.DataFrame(knowledge_base["input_file_inventory"])
inventory_df.to_csv(TABLES_OUT / "stage7_literature_file_inventory.csv", index=False)


# ---------------------------------------------------------------------
# Load core tables
# ---------------------------------------------------------------------

all_records = safe_read_csv(input_files["stage7a_records_csv"])
all_scored = safe_read_csv(input_files["stage7b_all_scored"])
priority_records = safe_read_csv(input_files["stage7b_priority_records"])
direct_hand = safe_read_csv(input_files["stage7b_direct_hand_records"])
reviews_datasets = safe_read_csv(input_files["stage7b_reviews_datasets"])
category_summary = safe_read_csv(input_files["stage7b_category_summary"])


# ---------------------------------------------------------------------
# Literature statistics
# ---------------------------------------------------------------------

def count_df(df):
    return 0 if df is None else int(df.shape[0])


knowledge_base["literature_statistics"] = {
    "stage7a_total_records": count_df(all_records),
    "stage7b_all_scored_records": count_df(all_scored),
    "priority_records": count_df(priority_records),
    "direct_hand_related_records": count_df(direct_hand),
    "reviews_and_datasets": count_df(reviews_datasets),
    "category_summary_rows": count_df(category_summary)
}


if category_summary is not None:
    category_summary.to_csv(TABLES_OUT / "stage7b_category_summary_copy.csv", index=False)


# ---------------------------------------------------------------------
# Extract high-priority evidence
# ---------------------------------------------------------------------

high_priority_tables = []

for name, df in [
    ("priority", priority_records),
    ("direct_hand", direct_hand),
    ("reviews_datasets", reviews_datasets)
]:
    if df is None:
        continue

    temp = df.copy()
    temp.insert(0, "evidence_group", name)

    useful_cols = []
    for col in [
        "title", "authors", "year", "journal", "doi", "url",
        "category", "decision", "priority_score", "relevance_score",
        "citation_count", "abstract", "citation"
    ]:
        if col in temp.columns:
            useful_cols.append(col)

    if useful_cols:
        temp = temp[["evidence_group"] + useful_cols]

    high_priority_tables.append(temp)


if high_priority_tables:
    high_priority_evidence = pd.concat(high_priority_tables, ignore_index=True)
    high_priority_evidence = high_priority_evidence.drop_duplicates(
        subset=[c for c in ["title", "doi"] if c in high_priority_evidence.columns],
        keep="first"
    )
else:
    high_priority_evidence = pd.DataFrame()

high_priority_evidence.to_csv(TABLES_OUT / "high_priority_literature_evidence.csv", index=False)


# ---------------------------------------------------------------------
# Automated gap screening
# ---------------------------------------------------------------------

gap_keywords = {
    "participant_level_multiview": [
        "participant-level", "multi-view", "multiview", "four views",
        "left dorsal", "right dorsal", "left palmar", "right palmar"
    ],
    "rgb_thermal_hand_fusion": [
        "rgb", "thermal", "infrared", "hand", "palm"
    ],
    "adaptive_attention_or_fusion": [
        "attention", "adaptive", "fusion", "weighting", "gating"
    ],
    "explainability": [
        "explainable", "interpretability", "xai", "saliency", "grad-cam"
    ],
    "symmetry_reasoning": [
        "symmetry", "bilateral", "left-right", "left right"
    ],
    "latent_physiological_state": [
        "latent", "physiological", "hemoglobin", "perfusion", "oxygen"
    ]
}


gap_rows = []

if high_priority_evidence is not None and not high_priority_evidence.empty:
    for _, row in high_priority_evidence.iterrows():
        title = str(row.get("title", ""))
        abstract = str(row.get("abstract", ""))
        combined = (title + " " + abstract).lower()

        rec = {
            "title": title,
            "year": row.get("year", ""),
            "doi": row.get("doi", ""),
            "evidence_group": row.get("evidence_group", "")
        }

        for gap_name, keywords in gap_keywords.items():
            rec[gap_name] = int(any(k.lower() in combined for k in keywords))

        gap_rows.append(rec)

gap_screen_df = pd.DataFrame(gap_rows)
gap_screen_df.to_csv(TABLES_OUT / "literature_gap_keyword_screening.csv", index=False)


# ---------------------------------------------------------------------
# Novelty questions and candidate contributions
# ---------------------------------------------------------------------

novelty_questions = [
    {
        "question_id": "NQ1",
        "question": "Do prior studies model HandDx-200-like data at participant level rather than independent image level?",
        "importance": "Participant-level modeling prevents leakage and reflects real diagnostic decision-making."
    },
    {
        "question_id": "NQ2",
        "question": "Do prior studies exploit four anatomical hand views jointly?",
        "importance": "The four-view structure is one of the strongest dataset-specific opportunities."
    },
    {
        "question_id": "NQ3",
        "question": "Do prior studies combine RGB and thermal hand evidence in a structured physiological way?",
        "importance": "RGB and thermal data represent different manifestations: pallor/color versus perfusion/temperature."
    },
    {
        "question_id": "NQ4",
        "question": "Do prior studies use adaptive specialist weighting instead of simple feature concatenation?",
        "importance": "Adaptive weighting supports participant-specific diagnostic evidence."
    },
    {
        "question_id": "NQ5",
        "question": "Do prior studies enforce bilateral or anatomical physiological consistency?",
        "importance": "Anemia is systemic; left-right and palmar-dorsal evidence should be physiologically coherent."
    },
    {
        "question_id": "NQ6",
        "question": "Do prior studies explain which view, modality, or representation drives the final anemia decision?",
        "importance": "Clinical screening tools require interpretable evidence pathways."
    }
]

candidate_contributions = [
    {
        "contribution_id": "C1",
        "title": "Participant-level multimodal anemia classification",
        "description": "The model should operate on a participant as the diagnostic unit, using all available RGB and thermal views jointly.",
        "source_rationale": "HandDx-200 provides eight images per participant, making participant-level fusion more appropriate than isolated image classification.",
        "implementation_priority": "High"
    },
    {
        "contribution_id": "C2",
        "title": "Multi-view anatomical specialist agents",
        "description": "Each anatomical view can be represented as a specialist observer: left/right palmar and left/right dorsal, for RGB and thermal modalities.",
        "source_rationale": "This directly uses the dataset acquisition protocol and supports interpretable anatomical reasoning.",
        "implementation_priority": "High"
    },
    {
        "contribution_id": "C3",
        "title": "RGB-centered multimodal learning with thermal auxiliary evidence",
        "description": "RGB should be treated as the dominant branch, while thermal evidence should be modeled as complementary physiological information.",
        "source_rationale": "The handcrafted pipeline showed stronger RGB signal and weaker standalone thermal performance.",
        "implementation_priority": "High"
    },
    {
        "contribution_id": "C4",
        "title": "Adaptive cooperative fusion",
        "description": "Fusion should use learned trust or attention weights across anatomical specialists instead of fixed concatenation.",
        "source_rationale": "Different participants may express anemia-related changes differently across views and modalities.",
        "implementation_priority": "High"
    },
    {
        "contribution_id": "C5",
        "title": "Symmetry-aware physiological consistency",
        "description": "The model should encourage coherent left-right evidence across corresponding anatomical regions.",
        "source_rationale": "Anemia is systemic, so bilateral hand evidence should not be treated as unrelated.",
        "implementation_priority": "Medium-High"
    },
    {
        "contribution_id": "C6",
        "title": "Multi-representation image evidence",
        "description": "Each image should generate multiple complementary representations such as original RGB, HSV, LAB, texture, patches, and enhanced regions.",
        "source_rationale": "The handcrafted pipeline showed value in color-space, texture, global, and local descriptors.",
        "implementation_priority": "High"
    },
    {
        "contribution_id": "C7",
        "title": "Explainable evidence pathway",
        "description": "The final system should report view-level, modality-level, and representation-level contribution weights.",
        "source_rationale": "Explainability differentiates the framework from generic CNN-based anemia classifiers.",
        "implementation_priority": "High"
    }
]

knowledge_base["novelty_questions"] = novelty_questions
knowledge_base["candidate_novel_contributions"] = candidate_contributions

pd.DataFrame(novelty_questions).to_csv(TABLES_OUT / "novelty_assessment_questions.csv", index=False)
pd.DataFrame(candidate_contributions).to_csv(TABLES_OUT / "candidate_novel_contributions.csv", index=False)


# ---------------------------------------------------------------------
# Save JSON
# ---------------------------------------------------------------------

with open(STAGE_OUT / "Stage8B_Literature_Novelty_KnowledgeBase.json", "w", encoding="utf-8") as f:
    json.dump(knowledge_base, f, indent=4, ensure_ascii=False)


# ---------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------

report = []
report.append("# Stage 8B Literature and Novelty Knowledge Base\n")
report.append(f"Generated at: {knowledge_base['created_at']}\n")
report.append(f"Base directory: `{BASE_DIR}`\n")

report.append("## Scope\n")
report.append("This stage consolidates Stage 7A literature-crawler outputs and Stage 7B filtered novelty-map outputs.\n")

report.append("## Evidence Coverage\n")
for k, v in knowledge_base["literature_statistics"].items():
    report.append(f"- {k}: {v}")

report.append("\n## Missing Files\n")
if knowledge_base["missing_files"]:
    for p in knowledge_base["missing_files"]:
        report.append(f"- `{p}`")
else:
    report.append("No required Stage 7A–7B files were missing.")

report.append("\n## Novelty Assessment Questions\n")
for q in novelty_questions:
    report.append(f"### {q['question_id']}. {q['question']}")
    report.append(f"**Importance:** {q['importance']}\n")

report.append("## Candidate Novel Contributions\n")
for c in candidate_contributions:
    report.append(f"### {c['contribution_id']}. {c['title']}")
    report.append(f"**Description:** {c['description']}")
    report.append(f"**Rationale:** {c['source_rationale']}")
    report.append(f"**Implementation priority:** {c['implementation_priority']}\n")

report.append("## Generated Output Files\n")
report.append("- `Stage8B_Literature_Novelty_KnowledgeBase.json`")
report.append("- `tables/stage7_literature_file_inventory.csv`")
report.append("- `tables/high_priority_literature_evidence.csv`")
report.append("- `tables/literature_gap_keyword_screening.csv`")
report.append("- `tables/novelty_assessment_questions.csv`")
report.append("- `tables/candidate_novel_contributions.csv`")

with open(REPORTS_OUT / "Stage8B_Literature_Novelty_KnowledgeBase_Report.md", "w", encoding="utf-8") as f:
    f.write("\n".join(report))


print("=" * 80)
print("STAGE 8B LITERATURE AND NOVELTY KNOWLEDGE BASE COMPLETED")
print("=" * 80)
print(f"Stage 7A total records: {knowledge_base['literature_statistics']['stage7a_total_records']}")
print(f"Priority records: {knowledge_base['literature_statistics']['priority_records']}")
print(f"Direct hand-related records: {knowledge_base['literature_statistics']['direct_hand_related_records']}")
print(f"Reviews and datasets: {knowledge_base['literature_statistics']['reviews_and_datasets']}")
print(f"Missing files: {len(knowledge_base['missing_files'])}")
print(f"Outputs saved to: {STAGE_OUT}")
print("=" * 80)

if knowledge_base["missing_files"]:
    print("Missing files:")
    for p in knowledge_base["missing_files"]:
        print(p)