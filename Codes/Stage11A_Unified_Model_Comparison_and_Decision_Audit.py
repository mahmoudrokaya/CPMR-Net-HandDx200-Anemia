# ============================================================
# Stage 11A — Unified Model Comparison and Decision Audit
# CPMR-Net HandDx-200 Project
#
# Save as:
#   D:\47\472\New-Papers\Anemia_Paper\Codes\Stage11A_Unified_Model_Comparison_and_Decision_Audit.py
#
# This stage does NOT train a model. It audits existing evidence.
# ============================================================

from pathlib import Path
from datetime import datetime
import json
import re
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(r"D:\47\472\New-Papers\Anemia_Paper")
OUTPUTS_ROOT = BASE_DIR / "Outputs"

OUTPUT_DIR = OUTPUTS_ROOT / "Stage11A_Unified_Model_Comparison_and_Decision_Audit"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"

for d in [OUTPUT_DIR, TABLES_DIR, FIGURES_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# MANUAL EVIDENCE FROM COMPLETED STAGES
# ============================================================
# These values are used as trusted stage-level summaries. The script also
# searches each stage output folder and fills missing values when possible.
MANUAL_EVIDENCE = [
    {
        "model_id": "Stage6D6B_RBFSVM_Nonredundant_Handcrafted",
        "stage": "Stage6D6B",
        "model_family": "Classical ML",
        "model_name": "RBFSVM on nonredundant handcrafted features",
        "primary_claim": "Strongest validated handcrafted benchmark",
        "validation_roc_auc": np.nan,
        "holdout_roc_auc": np.nan,
        "repeated_cv_roc_auc_mean": 0.7447,
        "repeated_cv_roc_auc_std": np.nan,
        "accuracy": 0.7374,
        "precision": 0.6585,
        "recall": 0.4154,
        "f1_score": 0.5094,
        "representation_learning": "Handcrafted physiological descriptors",
        "comment": "Mandatory benchmark for all future CPMR-Net claims.",
    },
    {
        "model_id": "Stage10L_Initial_CPMRNet",
        "stage": "Stage10L",
        "model_family": "CPMR-Net",
        "model_name": "Initial full CPMR-Net",
        "primary_claim": "High validation ROC-AUC but weak holdout transfer",
        "validation_roc_auc": 0.8052,
        "holdout_roc_auc": 0.5556,
        "repeated_cv_roc_auc_mean": np.nan,
        "repeated_cv_roc_auc_std": np.nan,
        "accuracy": np.nan,
        "precision": np.nan,
        "recall": np.nan,
        "f1_score": np.nan,
        "representation_learning": "Supervised training from scratch",
        "comment": "Do not select on validation AUC alone.",
    },
    {
        "model_id": "Stage10P_Repeated_CV_CPMRNet",
        "stage": "Stage10P",
        "model_family": "CPMR-Net",
        "model_name": "Repeated-CV CPMR-Net",
        "primary_claim": "Confirmed systematic generalization instability",
        "validation_roc_auc": np.nan,
        "holdout_roc_auc": np.nan,
        "repeated_cv_roc_auc_mean": 0.5923,
        "repeated_cv_roc_auc_std": 0.1022,
        "accuracy": np.nan,
        "precision": np.nan,
        "recall": 0.4585,
        "f1_score": 0.3967,
        "representation_learning": "Supervised training from scratch",
        "comment": "Most important evidence for generalization diagnosis.",
    },
    {
        "model_id": "Stage10X_Progressive_Contrastive_FineTuning",
        "stage": "Stage10X",
        "model_family": "CPMR-Net",
        "model_name": "Progressive contrastive fine-tuning",
        "primary_claim": "Best stabilized deep-learning validation model",
        "validation_roc_auc": 0.7879,
        "holdout_roc_auc": np.nan,
        "repeated_cv_roc_auc_mean": np.nan,
        "repeated_cv_roc_auc_std": np.nan,
        "accuracy": np.nan,
        "precision": np.nan,
        "recall": np.nan,
        "f1_score": np.nan,
        "representation_learning": "Contrastive pretraining plus progressive fine-tuning",
        "comment": "Current strongest CPMR-Net candidate; needs conservative final verification.",
    },
    {
        "model_id": "Stage10Y_Participant_Consistency",
        "stage": "Stage10Y",
        "model_family": "CPMR-Net",
        "model_name": "Participant consistency regularization",
        "primary_claim": "Did not improve over Stage10X",
        "validation_roc_auc": 0.7749,
        "holdout_roc_auc": np.nan,
        "repeated_cv_roc_auc_mean": np.nan,
        "repeated_cv_roc_auc_std": np.nan,
        "accuracy": np.nan,
        "precision": np.nan,
        "recall": np.nan,
        "f1_score": np.nan,
        "representation_learning": "Fine-tuning plus embedding consistency regularization",
        "comment": "Useful negative/diagnostic experiment.",
    },
    {
        "model_id": "Stage10Z_Knowledge_Distillation",
        "stage": "Stage10Z",
        "model_family": "CPMR-Net",
        "model_name": "Knowledge distillation from handcrafted RBFSVM teacher",
        "primary_claim": "Beneficial but below Stage10X",
        "validation_roc_auc": 0.7835,
        "holdout_roc_auc": np.nan,
        "repeated_cv_roc_auc_mean": np.nan,
        "repeated_cv_roc_auc_std": np.nan,
        "accuracy": np.nan,
        "precision": np.nan,
        "recall": np.nan,
        "f1_score": np.nan,
        "representation_learning": "Teacher-guided representation learning",
        "comment": "Supportive evidence but not current best deep-learning candidate.",
    },
]

EXPECTED_STAGE_DIRS = {
    "Stage6D6B": OUTPUTS_ROOT / "Stage6D6B_NonredundantFeature_ML_Benchmark",
    "Stage10L": OUTPUTS_ROOT / "Stage10L_CPMRNet_Training_Engine",
    "Stage10M": OUTPUTS_ROOT / "Stage10M_Validation_Threshold_Analysis",
    "Stage10N": OUTPUTS_ROOT / "Stage10N_Independent_Holdout_Evaluation",
    "Stage10O": OUTPUTS_ROOT / "Stage10O_Generalization_Gap_Diagnosis",
    "Stage10P": OUTPUTS_ROOT / "Stage10P_Repeated_CrossValidation",
    "Stage10Q": OUTPUTS_ROOT / "Stage10Q_Frozen_Encoder_Experiments",
    "Stage10R": OUTPUTS_ROOT / "Stage10R_RGB_Only_CPMRNet",
    "Stage10S": OUTPUTS_ROOT / "Stage10S_ImageNet_Pretrained_Baselines",
    "Stage10T": OUTPUTS_ROOT / "Stage10T_Classical_ML_on_Learned_Embeddings",
    "Stage10U": OUTPUTS_ROOT / "Stage10U_SelfSupervised_Contrastive_Pretraining",
    "Stage10V": OUTPUTS_ROOT / "Stage10V_Contrastive_Embedding_Evaluation",
    "Stage10W": OUTPUTS_ROOT / "Stage10W_MultiTask_Learning",
    "Stage10X": OUTPUTS_ROOT / "Stage10X_Progressive_Contrastive_FineTuning",
    "Stage10Y": OUTPUTS_ROOT / "Stage10Y_Participant_Consistency_Regularization",
    "Stage10Z": OUTPUTS_ROOT / "Stage10Z_Handcrafted_Teacher_Knowledge_Distillation",
}

METRIC_ALIASES = {
    "validation_roc_auc": ["validation_roc_auc", "val_roc_auc", "val_auc", "best_val_auc", "validation_auc"],
    "holdout_roc_auc": ["holdout_roc_auc", "test_roc_auc", "test_auc", "independent_test_roc_auc"],
    "repeated_cv_roc_auc_mean": ["mean_test_roc_auc", "test_roc_auc_mean", "mean_roc_auc", "roc_auc_mean"],
    "repeated_cv_roc_auc_std": ["std_test_roc_auc", "test_roc_auc_std", "std_roc_auc", "roc_auc_std"],
    "accuracy": ["accuracy", "acc", "test_accuracy", "mean_accuracy"],
    "precision": ["precision", "test_precision", "mean_precision"],
    "recall": ["recall", "sensitivity", "test_recall", "mean_recall"],
    "f1_score": ["f1", "f1_score", "test_f1", "mean_f1_score"],
    "pr_auc": ["pr_auc", "average_precision", "val_pr_auc", "test_pr_auc"],
    "balanced_accuracy": ["balanced_accuracy", "bal_acc", "test_balanced_accuracy"],
}

def to_float(x):
    try:
        if x is None:
            return np.nan
        if isinstance(x, str):
            x = x.strip().replace("%", "")
            if x.lower() in ["", "na", "n/a", "nan", "none"]:
                return np.nan
        return float(x)
    except Exception:
        return np.nan

def first_available(*values):
    for v in values:
        fv = to_float(v)
        if not np.isnan(fv):
            return fv
    return np.nan

def normalize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip().lower().replace("-", "_").replace(" ", "_") for c in df.columns]
    return df

def flatten_json(obj, prefix=""):
    flat = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, (dict, list)):
                flat.update(flatten_json(v, key))
            else:
                flat[key] = v
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            key = f"{prefix}.{i}" if prefix else str(i)
            if isinstance(v, (dict, list)):
                flat.update(flatten_json(v, key))
            else:
                flat[key] = v
    return flat

def discover_files(stage_dir):
    if not stage_dir.exists():
        return []
    files = []
    for pat in ["*.csv", "*.json", "*.md", "*.txt"]:
        files.extend(stage_dir.rglob(pat))
    return sorted(files)

def metric_from_flat(flat, aliases):
    norm = {str(k).lower().replace("-", "_").replace(" ", "_"): v for k, v in flat.items()}
    for alias in aliases:
        a = alias.lower()
        for k, v in norm.items():
            if k == a or k.endswith("." + a) or k.endswith("_" + a):
                fv = to_float(v)
                if not np.isnan(fv):
                    return fv
    return np.nan

def extract_from_json(path):
    try:
        obj = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        flat = flatten_json(obj)
        return {m: metric_from_flat(flat, aliases) for m, aliases in METRIC_ALIASES.items()}
    except Exception:
        return {m: np.nan for m in METRIC_ALIASES}

def extract_from_csv(path):
    try:
        df = pd.read_csv(path)
    except Exception:
        try:
            df = pd.read_csv(path, encoding="latin1")
        except Exception:
            return {m: np.nan for m in METRIC_ALIASES}
    if df.empty:
        return {m: np.nan for m in METRIC_ALIASES}
    df = normalize_columns(df)
    out = {m: np.nan for m in METRIC_ALIASES}
    for metric, aliases in METRIC_ALIASES.items():
        for alias in aliases:
            if alias in df.columns:
                vals = pd.to_numeric(df[alias], errors="coerce").dropna()
                if len(vals):
                    out[metric] = float(vals.max() if "validation" in metric or "holdout" in metric else vals.iloc[-1])
                    break
    if "metric" in df.columns:
        value_cols = [c for c in ["value", "score", "mean", "result"] if c in df.columns]
        if value_cols:
            flat = {str(r["metric"]).lower().replace(" ", "_"): r[value_cols[0]] for _, r in df.iterrows()}
            for metric, aliases in METRIC_ALIASES.items():
                out[metric] = first_available(out[metric], metric_from_flat(flat, aliases))
    return out

def extract_from_text(path):
    try:
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
    except Exception:
        return {m: np.nan for m in METRIC_ALIASES}
    patterns = {
        "validation_roc_auc": [r"validation\s+roc[- ]?auc\s*[:=]?\s*([0-9]*\.?[0-9]+)", r"val[_\s-]?auc\s*[:=]?\s*([0-9]*\.?[0-9]+)"],
        "holdout_roc_auc": [r"test\s+roc[- ]?auc\s*[:=]?\s*([0-9]*\.?[0-9]+)", r"holdout\s+roc[- ]?auc\s*[:=]?\s*([0-9]*\.?[0-9]+)"],
        "repeated_cv_roc_auc_mean": [r"mean\s+test\s+roc[- ]?auc\s*[:=]?\s*([0-9]*\.?[0-9]+)", r"roc[- ]?auc\s*[:=]?\s*([0-9]*\.?[0-9]+)\s*[±]"],
        "repeated_cv_roc_auc_std": [r"roc[- ]?auc\s*[:=]?\s*[0-9]*\.?[0-9]+\s*[±]\s*([0-9]*\.?[0-9]+)"],
        "accuracy": [r"accuracy\s*[:=]?\s*([0-9]*\.?[0-9]+)"],
        "precision": [r"precision\s*[:=]?\s*([0-9]*\.?[0-9]+)"],
        "recall": [r"recall\s*[:=]?\s*([0-9]*\.?[0-9]+)"],
        "f1_score": [r"f1[-_ ]?score\s*[:=]?\s*([0-9]*\.?[0-9]+)", r"\bf1\s*[:=]?\s*([0-9]*\.?[0-9]+)"],
        "pr_auc": [r"pr[- ]?auc\s*[:=]?\s*([0-9]*\.?[0-9]+)"],
        "balanced_accuracy": [r"balanced\s+accuracy\s*[:=]?\s*([0-9]*\.?[0-9]+)"],
    }
    out = {m: np.nan for m in METRIC_ALIASES}
    for m, pats in patterns.items():
        for pat in pats:
            hit = re.search(pat, text)
            if hit:
                out[m] = to_float(hit.group(1))
                break
    return out

def discover_metrics(stage_dir):
    files = discover_files(stage_dir)
    merged = {m: np.nan for m in METRIC_ALIASES}
    for f in files:
        if f.suffix.lower() == ".json":
            found = extract_from_json(f)
        elif f.suffix.lower() == ".csv":
            found = extract_from_csv(f)
        elif f.suffix.lower() in [".md", ".txt"]:
            found = extract_from_text(f)
        else:
            found = {}
        for k, v in found.items():
            merged[k] = first_available(merged.get(k, np.nan), v)
    return merged, files

def evidence_tier(row):
    if not np.isnan(to_float(row.get("repeated_cv_roc_auc_mean"))):
        return "Tier 1: repeated-CV evidence"
    if not np.isnan(to_float(row.get("holdout_roc_auc"))):
        return "Tier 2: independent holdout evidence"
    if not np.isnan(to_float(row.get("validation_roc_auc"))):
        return "Tier 3: validation-only evidence"
    return "Tier 4: incomplete metric evidence"

def generalization_gap(row):
    val = to_float(row.get("validation_roc_auc"))
    test = to_float(row.get("holdout_roc_auc"))
    if not np.isnan(val) and not np.isnan(test):
        return val - test
    return np.nan

def best_available_auc(row):
    return first_available(row.get("repeated_cv_roc_auc_mean"), row.get("holdout_roc_auc"), row.get("validation_roc_auc"))

def stability_label(row):
    std = to_float(row.get("repeated_cv_roc_auc_std"))
    gap = to_float(row.get("generalization_gap_val_minus_holdout"))
    if not np.isnan(std):
        if std <= 0.05:
            return "Stable repeated-CV"
        if std <= 0.10:
            return "Moderately variable repeated-CV"
        return "Highly variable repeated-CV"
    if not np.isnan(gap):
        if abs(gap) <= 0.05:
            return "Small validation-holdout gap"
        if abs(gap) <= 0.15:
            return "Moderate validation-holdout gap"
        return "Large validation-holdout gap"
    return "Stability not fully established"

def decision_score(row):
    rep = to_float(row.get("repeated_cv_roc_auc_mean"))
    hold = to_float(row.get("holdout_roc_auc"))
    val = to_float(row.get("validation_roc_auc"))
    std = to_float(row.get("repeated_cv_roc_auc_std"))
    gap = to_float(row.get("generalization_gap_val_minus_holdout"))
    if not np.isnan(rep):
        score = rep - (0.5 * std if not np.isnan(std) else 0)
    elif not np.isnan(hold):
        score = hold - 0.03
    elif not np.isnan(val):
        score = val - 0.10
    else:
        score = 0.0
    if not np.isnan(gap) and gap > 0:
        score -= 0.5 * gap
    return score

def recommendation(row):
    stage = row["stage"]
    if stage == "Stage6D6B":
        return "Retain as strongest validated handcrafted benchmark and mandatory comparator."
    if stage == "Stage10X":
        return "Retain as current best stabilized CPMR-Net candidate; verify with holdout/repeated-CV before final manuscript selection."
    if stage == "Stage10Z":
        return "Retain as supportive teacher-guidance evidence but not above Stage10X."
    if stage == "Stage10Y":
        return "Treat as useful negative experiment; consistency regularization did not improve the best model."
    if stage == "Stage10L":
        return "Do not select based on validation AUC because holdout transfer was poor."
    if stage == "Stage10P":
        return "Use as the main evidence that deep-learning generalization is unstable with 198 participants."
    if "validation-only" in row["evidence_tier"]:
        return "Promising but insufficient for final selection without stronger generalization evidence."
    return "Retain for comparison and review source evidence."

# ============================================================
# BUILD AUDIT TABLE
# ============================================================
records = []
for rec in MANUAL_EVIDENCE:
    rec = dict(rec)
    stage_dir = EXPECTED_STAGE_DIRS.get(rec["stage"], OUTPUTS_ROOT / rec["stage"])
    discovered, files = discover_metrics(stage_dir)
    for metric in METRIC_ALIASES:
        rec[metric] = first_available(rec.get(metric, np.nan), discovered.get(metric, np.nan))
    rec["stage_output_dir"] = str(stage_dir)
    rec["stage_dir_found"] = stage_dir.exists()
    rec["evidence_files_found"] = len(files)
    rec["evidence_file_examples"] = " | ".join([f.name for f in files[:5]])
    records.append(rec)

# Add discovered folders not already represented.
represented = {r["stage"] for r in records}
for stage, stage_dir in EXPECTED_STAGE_DIRS.items():
    if stage in represented:
        continue
    discovered, files = discover_metrics(stage_dir)
    has_metric = any(not np.isnan(to_float(v)) for v in discovered.values())
    if stage_dir.exists() or has_metric:
        records.append({
            "model_id": f"{stage}_Discovered",
            "stage": stage,
            "model_family": "Discovered evidence",
            "model_name": f"{stage} discovered experiment",
            "primary_claim": "Automatically discovered from outputs; review required",
            "representation_learning": "Discovered",
            "comment": "Review before manuscript use.",
            **discovered,
            "stage_output_dir": str(stage_dir),
            "stage_dir_found": stage_dir.exists(),
            "evidence_files_found": len(files),
            "evidence_file_examples": " | ".join([f.name for f in files[:5]]),
        })

comparison = pd.DataFrame(records)
for c in METRIC_ALIASES:
    if c not in comparison.columns:
        comparison[c] = np.nan
    comparison[c] = pd.to_numeric(comparison[c], errors="coerce")

comparison["best_available_auc_conservative"] = comparison.apply(best_available_auc, axis=1)
comparison["generalization_gap_val_minus_holdout"] = comparison.apply(generalization_gap, axis=1)
comparison["evidence_tier"] = comparison.apply(evidence_tier, axis=1)
comparison["stability_label"] = comparison.apply(stability_label, axis=1)
comparison["decision_score"] = comparison.apply(decision_score, axis=1)
comparison["rank_by_decision_score"] = comparison["decision_score"].rank(ascending=False, method="dense").astype(int)
comparison["stage11a_recommendation"] = comparison.apply(recommendation, axis=1)
comparison = comparison.sort_values(["rank_by_decision_score", "stage"]).reset_index(drop=True)

core_cols = [
    "rank_by_decision_score", "model_id", "stage", "model_family", "model_name", "primary_claim",
    "validation_roc_auc", "holdout_roc_auc", "repeated_cv_roc_auc_mean", "repeated_cv_roc_auc_std",
    "best_available_auc_conservative", "generalization_gap_val_minus_holdout", "accuracy", "precision",
    "recall", "f1_score", "pr_auc", "balanced_accuracy", "evidence_tier", "stability_label",
    "decision_score", "representation_learning", "comment", "stage11a_recommendation",
    "stage_dir_found", "evidence_files_found", "evidence_file_examples", "stage_output_dir"
]
for c in core_cols:
    if c not in comparison.columns:
        comparison[c] = np.nan
comparison_core = comparison[core_cols].copy()

missing = pd.DataFrame([
    {"stage": s, "expected_output_dir": str(p), "found": p.exists(), "files_found": len(discover_files(p))}
    for s, p in EXPECTED_STAGE_DIRS.items()
])

comparison_core.to_csv(TABLES_DIR / "stage11a_unified_model_comparison.csv", index=False)
comparison.to_csv(TABLES_DIR / "stage11a_unified_model_comparison_full.csv", index=False)
comparison_core.to_csv(TABLES_DIR / "stage11a_final_decision_ranking.csv", index=False)
comparison_core[[
    "stage", "model_name", "validation_roc_auc", "holdout_roc_auc", "repeated_cv_roc_auc_mean",
    "repeated_cv_roc_auc_std", "evidence_tier", "stability_label", "stage11a_recommendation"
]].to_csv(TABLES_DIR / "stage11a_decision_matrix.csv", index=False)
missing.to_csv(TABLES_DIR / "stage11a_missing_evidence_audit.csv", index=False)

# ============================================================
# FIGURES
# ============================================================
def plot_bar(df, metric, filename, title):
    plot_df = df.dropna(subset=[metric]).sort_values(metric, ascending=True)
    if plot_df.empty:
        return
    labels = plot_df["stage"].astype(str) + "\n" + plot_df["model_name"].astype(str).str.slice(0, 36)
    plt.figure(figsize=(12, max(5, 0.55 * len(plot_df))))
    plt.barh(range(len(plot_df)), plot_df[metric])
    plt.yticks(range(len(plot_df)), labels)
    plt.xlabel(metric.replace("_", " ").title())
    plt.title(title)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close()

plot_bar(comparison_core, "validation_roc_auc", "stage11a_validation_roc_auc_comparison.png", "Validation ROC-AUC comparison")
plot_bar(comparison_core, "best_available_auc_conservative", "stage11a_conservative_auc_comparison.png", "Conservative best-available ROC-AUC comparison")
plot_bar(comparison_core, "decision_score", "stage11a_decision_score_ranking.png", "Conservative decision-score ranking")

# ============================================================
# REPORT
# ============================================================
def fmt(x):
    x = to_float(x)
    return "NA" if np.isnan(x) else f"{x:.4f}"

ranking = comparison_core.sort_values("decision_score", ascending=False).reset_index(drop=True)
rbf = comparison_core[comparison_core["stage"] == "Stage6D6B"]
stage10x = comparison_core[comparison_core["stage"] == "Stage10X"]

lines = []
lines.append("# Stage 11A — Unified Model Comparison and Decision Audit")
lines.append("")
lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
lines.append("")
lines.append("## Purpose")
lines.append("")
lines.append("This stage consolidates handcrafted and CPMR-Net evidence into one conservative decision audit. It does not train, tune, or redesign any model.")
lines.append("")
lines.append("## Fixed Scientific Rules")
lines.append("")
lines.append("- Diagnosis is participant-level only.")
lines.append("- The CPMR-Net hierarchy remains fixed.")
lines.append("- Validation ROC-AUC alone is not sufficient for final model selection.")
lines.append("- Independent holdout and repeated-CV evidence have higher priority than a single validation split.")
lines.append("")
lines.append("## Main Findings")
lines.append("")
if len(rbf):
    r = rbf.iloc[0]
    lines.append(f"- Stage 6D6B remains the strongest validated handcrafted benchmark: ROC-AUC {fmt(r['best_available_auc_conservative'])}, accuracy {fmt(r['accuracy'])}, recall {fmt(r['recall'])}, F1-score {fmt(r['f1_score'])}.")
if len(stage10x):
    r = stage10x.iloc[0]
    lines.append(f"- Stage 10X remains the strongest stabilized CPMR-Net candidate by validation evidence: validation ROC-AUC {fmt(r['validation_roc_auc'])}.")
lines.append("- Stage 10L should not be selected despite high validation ROC-AUC because its independent holdout ROC-AUC dropped to 0.5556.")
lines.append("- Stage 10P confirms that deep-learning generalization is unstable under repeated CV, with ROC-AUC 0.5923 ± 0.1022.")
lines.append("- Stage 10Z supports the value of handcrafted-teacher knowledge distillation but does not exceed Stage 10X.")
lines.append("")
lines.append("## Conservative Ranking")
lines.append("")
lines.append("| Rank | Stage | Model | Val AUC | Holdout AUC | Repeated-CV AUC | Decision score | Evidence tier |")
lines.append("|---:|---|---|---:|---:|---:|---:|---|")
for _, r in ranking.iterrows():
    rep = fmt(r["repeated_cv_roc_auc_mean"])
    if not np.isnan(to_float(r["repeated_cv_roc_auc_std"])):
        rep += f" ± {fmt(r['repeated_cv_roc_auc_std'])}"
    lines.append(f"| {int(r['rank_by_decision_score'])} | {r['stage']} | {r['model_name']} | {fmt(r['validation_roc_auc'])} | {fmt(r['holdout_roc_auc'])} | {rep} | {fmt(r['decision_score'])} | {r['evidence_tier']} |")
lines.append("")
lines.append("## Decision Interpretation")
lines.append("")
lines.append("The audit separates model promise from model reliability. The handcrafted RBFSVM remains the safest validated benchmark. Stage 10X is the leading CPMR-Net candidate, but it should be finalized only after conservative generalization confirmation if holdout or repeated-CV evidence is not already available.")
lines.append("")
lines.append("## Recommended Next Stage")
lines.append("")
lines.append("Proceed to Stage 11B: final-candidate verification for Stage 10X using holdout evaluation, repeated-CV evaluation, or both, without changing the CPMR-Net architecture.")
lines.append("")
lines.append("## Generated Outputs")
lines.append("")
lines.append("- tables/stage11a_unified_model_comparison.csv")
lines.append("- tables/stage11a_unified_model_comparison_full.csv")
lines.append("- tables/stage11a_final_decision_ranking.csv")
lines.append("- tables/stage11a_decision_matrix.csv")
lines.append("- tables/stage11a_missing_evidence_audit.csv")
lines.append("- figures/stage11a_validation_roc_auc_comparison.png")
lines.append("- figures/stage11a_conservative_auc_comparison.png")
lines.append("- figures/stage11a_decision_score_ranking.png")

(REPORTS_DIR / "Stage11A_Unified_Model_Comparison_and_Decision_Audit_Report.md").write_text("\n".join(lines), encoding="utf-8")

summary = {
    "stage": "Stage11A",
    "models_audited": int(len(comparison_core)),
    "expected_stage_dirs_checked": int(len(missing)),
    "existing_expected_dirs": int(missing["found"].sum()),
    "top_ranked_model": ranking.iloc[0].to_dict() if len(ranking) else {},
    "interpretation": "Stage6D6B remains the strongest validated benchmark; Stage10X remains the strongest stabilized CPMR-Net candidate pending conservative generalization confirmation.",
    "output_dir": str(OUTPUT_DIR),
}
with open(OUTPUT_DIR / "Stage11A_Unified_Model_Comparison_and_Decision_Audit_Summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, default=str)

print("=" * 80)
print("STAGE 11A UNIFIED MODEL COMPARISON AND DECISION AUDIT COMPLETED")
print("=" * 80)
print(f"Models/stages audited: {len(comparison_core)}")
print(f"Expected output directories checked: {len(missing)}")
print(f"Existing expected directories: {int(missing['found'].sum())}")
print("Top conservative ranking:")
for _, r in ranking.head(5).iterrows():
    print(f"  Rank {int(r['rank_by_decision_score'])}: {r['stage']} | {r['model_name']} | score={fmt(r['decision_score'])} | {r['evidence_tier']}")
print(f"Results saved to: {OUTPUT_DIR}")
print("=" * 80)
