# -*- coding: utf-8 -*-
r"""
Stage 8A - Handcrafted Pipeline Knowledge Base

Purpose:
Consolidate all existing S1-S6 outputs into one structured knowledge base
before moving to deep-learning architecture design.

This script does NOT train models and does NOT modify previous outputs.
"""

from pathlib import Path
import json
import pandas as pd
from datetime import datetime

BASE_DIR = Path(r"D:\47\472\New-Papers\Anemia_Paper")
OUTPUTS_DIR = BASE_DIR / "Outputs"
STAGE_OUT = OUTPUTS_DIR / "Stage8A_HandcraftedPipeline_KnowledgeBase"

TABLES_OUT = STAGE_OUT / "tables"
REPORTS_OUT = STAGE_OUT / "reports"

TABLES_OUT.mkdir(parents=True, exist_ok=True)
REPORTS_OUT.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------

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
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        try:
            return path.read_text(encoding="latin1")
        except Exception:
            return ""


def summarize_table(path, name):
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


def best_model_from_summary(path, stage_name):
    df = safe_read_csv(path)
    if df is None or df.empty:
        return None

    # Try to find standard metric columns
    possible_auc_cols = ["roc_auc", "ROC-AUC", "roc_auc_mean", "mean_roc_auc"]
    possible_acc_cols = ["accuracy", "Accuracy", "accuracy_mean", "mean_accuracy"]

    sort_col = None
    for col in possible_auc_cols:
        if col in df.columns:
            sort_col = col
            break

    if sort_col is None:
        for col in possible_acc_cols:
            if col in df.columns:
                sort_col = col
                break

    if sort_col is None:
        return {
            "stage": stage_name,
            "path": str(path),
            "note": "Performance summary found, but no standard metric column detected.",
            "columns": list(df.columns)
        }

    best = df.sort_values(sort_col, ascending=False).iloc[0].to_dict()

    return {
        "stage": stage_name,
        "path": str(path),
        "ranking_metric": sort_col,
        "best_model_record": best
    }


# -------------------------------------------------------------------------
# Expected S1-S6 evidence files
# -------------------------------------------------------------------------

evidence_files = {
    "S1_dataset_audit": {
        "report": OUTPUTS_DIR / "Stage1_Metadata_and_Image_Audit" / "reports" / "Stage1_Metadata_and_Image_Audit_Report.md",
        "tables": {
            "metadata_summary": OUTPUTS_DIR / "Stage1_Metadata_and_Image_Audit" / "tables" / "metadata_file_summary.csv",
            "image_modality_summary": OUTPUTS_DIR / "Stage1_Metadata_and_Image_Audit" / "tables" / "image_modality_summary.csv",
            "image_inventory_quality_audit": OUTPUTS_DIR / "Stage1_Metadata_and_Image_Audit" / "tables" / "image_inventory_quality_audit.csv",
            "failed_images": OUTPUTS_DIR / "Stage1_Metadata_and_Image_Audit" / "tables" / "failed_images.csv",
            "duplicate_images_by_hash": OUTPUTS_DIR / "Stage1_Metadata_and_Image_Audit" / "tables" / "duplicate_images_by_hash.csv",
        }
    },

    "S2_thermal_decoder": {
        "report": OUTPUTS_DIR / "Stage2_Thermal_BMP_Decoder" / "Stage2_Thermal_BMP_Decoder_Report.md",
        "tables": {
            "thermal_decoding_summary": OUTPUTS_DIR / "Stage2_Thermal_BMP_Decoder" / "tables" / "thermal_decoding_summary.csv",
            "thermal_bmp_decoding_audit": OUTPUTS_DIR / "Stage2_Thermal_BMP_Decoder" / "tables" / "thermal_bmp_decoding_audit.csv",
        }
    },

    "S3_thermal_png_organization": {
        "report": OUTPUTS_DIR / "Stage3_Organize_Thermal_PNG_By_Class" / "reports" / "Stage3_Thermal_PNG_Organization_Report.md",
        "tables": {
            "thermal_png_validation_counts": OUTPUTS_DIR / "Stage3_Organize_Thermal_PNG_By_Class" / "tables" / "thermal_png_validation_counts.csv",
            "thermal_png_organization_log": OUTPUTS_DIR / "Stage3_Organize_Thermal_PNG_By_Class" / "tables" / "thermal_png_organization_log.csv",
        }
    },

    "S4_dataset_characterization": {
        "report": OUTPUTS_DIR / "Stage4_Dataset_Characterization_and_Inventory" / "reports" / "Stage4_Dataset_Characterization_and_Inventory_Report.md",
        "tables": {
            "participant_dataset": OUTPUTS_DIR / "Stage4_Dataset_Characterization_and_Inventory" / "tables" / "participant_dataset.csv",
            "participant_inventory_summary": OUTPUTS_DIR / "Stage4_Dataset_Characterization_and_Inventory" / "tables" / "participant_inventory_summary.csv",
            "participant_multimodal_inventory": OUTPUTS_DIR / "Stage4_Dataset_Characterization_and_Inventory" / "tables" / "participant_multimodal_inventory.csv",
            "table1_participant_demographics": OUTPUTS_DIR / "Stage4_Dataset_Characterization_and_Inventory" / "tables" / "table1_participant_demographics.csv",
            "table2_image_statistics": OUTPUTS_DIR / "Stage4_Dataset_Characterization_and_Inventory" / "tables" / "table2_image_statistics.csv",
            "missing_participants_or_views": OUTPUTS_DIR / "Stage4_Dataset_Characterization_and_Inventory" / "tables" / "missing_participants_or_views.csv",
        }
    },

    "S5_feature_extraction": {
        "reports": [
            OUTPUTS_DIR / "Stage5A_Global_Feature_Extraction" / "reports" / "Stage5A_Global_Feature_Extraction_Report.md",
            OUTPUTS_DIR / "Stage5B_Local_Patch_Feature_Extraction" / "reports" / "Stage5B_Local_Patch_Feature_Extraction_Report.md",
        ],
        "tables": {
            "participant_global_features": OUTPUTS_DIR / "Stage5A_Global_Feature_Extraction" / "tables" / "participant_global_features.csv",
            "global_feature_summary_statistics": OUTPUTS_DIR / "Stage5A_Global_Feature_Extraction" / "tables" / "global_feature_summary_statistics.csv",
            "participant_patch_features": OUTPUTS_DIR / "Stage5B_Local_Patch_Feature_Extraction" / "tables" / "participant_patch_features.csv",
            "patch_feature_summary_statistics": OUTPUTS_DIR / "Stage5B_Local_Patch_Feature_Extraction" / "tables" / "patch_feature_summary_statistics.csv",
            "rgb_patch_features_long": OUTPUTS_DIR / "Stage5B_Local_Patch_Feature_Extraction" / "tables" / "rgb_patch_features_long.csv",
            "thermal_patch_features_long": OUTPUTS_DIR / "Stage5B_Local_Patch_Feature_Extraction" / "tables" / "thermal_patch_features_long.csv",
        }
    },

    "S6_statistical_and_ml_analysis": {
        "reports": [
            OUTPUTS_DIR / "Stage6A_Global_Feature_Statistical_Testing" / "reports" / "Stage6A_Global_Feature_Statistical_Testing_Report.md",
            OUTPUTS_DIR / "Stage6B_Local_Patch_Statistical_Testing" / "reports" / "Stage6B_Local_Patch_Feature_Statistical_Testing_Report.md",
            OUTPUTS_DIR / "Stage6C_Global_vs_Local_Feature_Integration_and_Filtering" / "reports" / "Stage6C_Global_vs_Local_Feature_Integration_and_Filtering_Report.md",
            OUTPUTS_DIR / "Stage6D6A_NonredundantFeatureSelection" / "reports" / "Stage6D6A_NonredundantFeatureSelection_Report.md",
            OUTPUTS_DIR / "Stage6D6B_NonredundantFeature_ML_Benchmark" / "reports" / "Stage6D6B_NonredundantFeature_ML_Benchmark_Report.md",
        ],
        "tables": {
            "global_feature_statistical_tests": OUTPUTS_DIR / "Stage6A_Global_Feature_Statistical_Testing" / "tables" / "global_feature_statistical_tests.csv",
            "local_patch_feature_statistical_tests": OUTPUTS_DIR / "Stage6B_Local_Patch_Statistical_Testing" / "tables" / "local_patch_feature_statistical_tests.csv",
            "top50_features_overall": OUTPUTS_DIR / "Stage6C_Global_vs_Local_Feature_Integration_and_Filtering" / "tables" / "top50_features_overall.csv",
            "top50_rgb_features": OUTPUTS_DIR / "Stage6C_Global_vs_Local_Feature_Integration_and_Filtering" / "tables" / "top50_rgb_features.csv",
            "top50_thermal_features": OUTPUTS_DIR / "Stage6C_Global_vs_Local_Feature_Integration_and_Filtering" / "tables" / "top50_thermal_features.csv",
            "redundant_features_removed": OUTPUTS_DIR / "Stage6D6A_NonredundantFeatureSelection" / "tables" / "redundant_features_removed.csv",
            "selected_nonredundant_feature_names": OUTPUTS_DIR / "Stage6D6A_NonredundantFeatureSelection" / "tables" / "selected_nonredundant_feature_names.csv",
            "top50_nonredundant_features": OUTPUTS_DIR / "Stage6D6A_NonredundantFeatureSelection" / "tables" / "top50_nonredundant_features.csv",
        }
    }
}


ml_benchmark_summaries = {
    "D1_Global": OUTPUTS_DIR / "Stage6D1_GlobalFeature_ML_Benchmark" / "tables" / "model_performance_summary.csv",
    "D2_LocalPatch": OUTPUTS_DIR / "Stage6D2_LocalPatchFeature_ML_Benchmark" / "tables" / "model_performance_summary.csv",
    "D3_RGB": OUTPUTS_DIR / "Stage6D3_RGBFeature_ML_Benchmark" / "tables" / "model_performance_summary.csv",
    "D4_Thermal": OUTPUTS_DIR / "Stage6D4_ThermalFeature_ML_Benchmark" / "tables" / "model_performance_summary.csv",
    "D5_Overall": OUTPUTS_DIR / "Stage6D5_OverallFeature_ML_Benchmark" / "tables" / "model_performance_summary.csv",
    "D6B_Nonredundant": OUTPUTS_DIR / "Stage6D6B_NonredundantFeature_ML_Benchmark" / "tables" / "model_performance_summary.csv",
}


# -------------------------------------------------------------------------
# Build knowledge base
# -------------------------------------------------------------------------

knowledge_base = {
    "stage": "Stage8A",
    "title": "Handcrafted Pipeline Knowledge Base",
    "base_dir": str(BASE_DIR),
    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "scope": "Existing handcrafted pipeline only: S1-S6",
    "evidence_tables": [],
    "reports_found": [],
    "reports_missing": [],
    "ml_best_models": [],
    "missing_required_files": []
}


# Reports and tables
for section_name, section in evidence_files.items():

    if "report" in section:
        report_path = section["report"]
        if report_path.exists():
            knowledge_base["reports_found"].append(str(report_path))
        else:
            knowledge_base["reports_missing"].append(str(report_path))

    if "reports" in section:
        for report_path in section["reports"]:
            if report_path.exists():
                knowledge_base["reports_found"].append(str(report_path))
            else:
                knowledge_base["reports_missing"].append(str(report_path))

    for table_name, table_path in section.get("tables", {}).items():
        summary = summarize_table(table_path, f"{section_name}::{table_name}")
        knowledge_base["evidence_tables"].append(summary)
        if not summary["exists"]:
            knowledge_base["missing_required_files"].append(str(table_path))


# ML summaries
ml_summary_rows = []

for stage_name, path in ml_benchmark_summaries.items():
    result = best_model_from_summary(path, stage_name)
    if result is not None:
        knowledge_base["ml_best_models"].append(result)

    df = safe_read_csv(path)
    if df is not None:
        df = df.copy()
        df.insert(0, "benchmark_stage", stage_name)
        df.insert(1, "source_file", str(path))
        ml_summary_rows.append(df)
    else:
        knowledge_base["missing_required_files"].append(str(path))

if ml_summary_rows:
    all_ml = pd.concat(ml_summary_rows, ignore_index=True)
    all_ml.to_csv(TABLES_OUT / "all_stage6_ml_performance_summaries.csv", index=False)
else:
    all_ml = pd.DataFrame()


# Table inventory
table_inventory = pd.DataFrame(knowledge_base["evidence_tables"])
table_inventory.to_csv(TABLES_OUT / "handcrafted_pipeline_evidence_inventory.csv", index=False)

# Best models table
best_model_records = []
for item in knowledge_base["ml_best_models"]:
    rec = {
        "stage": item.get("stage"),
        "ranking_metric": item.get("ranking_metric"),
        "path": item.get("path")
    }
    best = item.get("best_model_record", {})
    for k, v in best.items():
        rec[k] = v
    best_model_records.append(rec)

best_models_df = pd.DataFrame(best_model_records)
best_models_df.to_csv(TABLES_OUT / "stage6_best_model_records.csv", index=False)


# -------------------------------------------------------------------------
# Interpretive summary
# -------------------------------------------------------------------------

interpretive_findings = [
    {
        "finding_id": "F1",
        "category": "Dataset integrity",
        "finding": "The S1-S4 pipeline verifies the dataset structure, participant inventory, image completeness, and multimodal organization before modeling.",
        "implication_for_next_phase": "The future architecture can safely operate at participant level rather than treating images as unrelated samples."
    },
    {
        "finding_id": "F2",
        "category": "Thermal preprocessing",
        "finding": "The thermal BMP decoding and PNG organization stages make the thermal modality usable for downstream analysis.",
        "implication_for_next_phase": "Thermal images should be retained as an auxiliary physiological branch, but not assumed to be the dominant signal."
    },
    {
        "finding_id": "F3",
        "category": "Feature extraction",
        "finding": "S5A and S5B generated global and local handcrafted descriptors from RGB and thermal images.",
        "implication_for_next_phase": "Future representation learning should preserve the distinction between global context and localized anatomical evidence."
    },
    {
        "finding_id": "F4",
        "category": "Statistical testing",
        "finding": "S6A-S6C identified stronger anemia-related signals in RGB features than in thermal features, with local anatomical patches outperforming broad global descriptors.",
        "implication_for_next_phase": "The new AI framework should be RGB-centered and anatomically localized, with thermal information used as complementary evidence."
    },
    {
        "finding_id": "F5",
        "category": "Redundancy",
        "finding": "S6D6A showed that many top-ranked handcrafted features were redundant, and redundancy reduction improved model performance.",
        "implication_for_next_phase": "The new architecture should include structured attention or adaptive trust rather than naive feature concatenation."
    },
    {
        "finding_id": "F6",
        "category": "Handcrafted ceiling",
        "finding": "S6D benchmarks establish the current handcrafted baseline and performance ceiling.",
        "implication_for_next_phase": "Any new deep-learning or cooperative physiological-intelligence framework must be compared directly against the best handcrafted baseline."
    }
]

findings_df = pd.DataFrame(interpretive_findings)
findings_df.to_csv(TABLES_OUT / "handcrafted_pipeline_interpretive_findings.csv", index=False)


# -------------------------------------------------------------------------
# Save JSON knowledge base
# -------------------------------------------------------------------------

with open(STAGE_OUT / "Stage8A_HandcraftedPipeline_KnowledgeBase.json", "w", encoding="utf-8") as f:
    json.dump(knowledge_base, f, indent=4, ensure_ascii=False)


# -------------------------------------------------------------------------
# Markdown report
# -------------------------------------------------------------------------

report = []
report.append("# Stage 8A Handcrafted Pipeline Knowledge Base\n")
report.append(f"Generated at: {knowledge_base['created_at']}\n")
report.append(f"Base directory: `{BASE_DIR}`\n")
report.append("## Scope\n")
report.append("This stage consolidates the existing handcrafted pipeline from S1 to S6 only.\n")

report.append("## Evidence Coverage\n")
report.append(f"- Reports found: {len(knowledge_base['reports_found'])}")
report.append(f"- Reports missing: {len(knowledge_base['reports_missing'])}")
report.append(f"- Evidence tables checked: {len(knowledge_base['evidence_tables'])}")
report.append(f"- Missing required files: {len(knowledge_base['missing_required_files'])}\n")

report.append("## Core Interpretive Findings\n")
for item in interpretive_findings:
    report.append(f"### {item['finding_id']}. {item['category']}")
    report.append(f"**Finding:** {item['finding']}")
    report.append(f"**Implication:** {item['implication_for_next_phase']}\n")

report.append("## Generated Output Files\n")
report.append("- `Stage8A_HandcraftedPipeline_KnowledgeBase.json`")
report.append("- `tables/handcrafted_pipeline_evidence_inventory.csv`")
report.append("- `tables/all_stage6_ml_performance_summaries.csv`")
report.append("- `tables/stage6_best_model_records.csv`")
report.append("- `tables/handcrafted_pipeline_interpretive_findings.csv`\n")

if knowledge_base["missing_required_files"]:
    report.append("## Missing Files Needing User Attention\n")
    for missing in knowledge_base["missing_required_files"]:
        report.append(f"- `{missing}`")
else:
    report.append("## Missing Files Needing User Attention\n")
    report.append("No required S1-S6 files were missing.\n")

with open(REPORTS_OUT / "Stage8A_HandcraftedPipeline_KnowledgeBase_Report.md", "w", encoding="utf-8") as f:
    f.write("\n".join(report))


print("=" * 80)
print("STAGE 8A HANDCRAFTED PIPELINE KNOWLEDGE BASE COMPLETED")
print("=" * 80)
print(f"Reports found: {len(knowledge_base['reports_found'])}")
print(f"Reports missing: {len(knowledge_base['reports_missing'])}")
print(f"Evidence tables checked: {len(knowledge_base['evidence_tables'])}")
print(f"Missing required files: {len(knowledge_base['missing_required_files'])}")
print(f"Outputs saved to: {STAGE_OUT}")
print("=" * 80)

if knowledge_base["missing_required_files"]:
    print("Missing files:")
    for missing in knowledge_base["missing_required_files"]:
        print(missing)