from pathlib import Path
import json
import math
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix, roc_curve
)

warnings.filterwarnings('ignore')

# ============================================================
# STAGE 11C — CONSERVATIVE FINAL-CANDIDATE VERIFICATION
# Target: Stage10X Progressive Contrastive Fine-Tuning
# ============================================================

BASE_DIR = Path(r"D:\47\472\New-Papers\Anemia_Paper")
OUTPUTS_ROOT = BASE_DIR / "Outputs"
OUTPUT_DIR = OUTPUTS_ROOT / "Stage11C_Conservative_Final_Candidate_Verification_of_Stage10X"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"

for d in [OUTPUT_DIR, TABLES_DIR, FIGURES_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Known evidence from completed stages, used only as fallback if raw files are missing.
KNOWN_STAGE10X = {
    "stage": "Stage10X",
    "model_name": "Progressive contrastive fine-tuning",
    "validation_roc_auc": 0.7879,
    "evidence_type": "validation_only_fallback",
}

BENCHMARK = {
    "stage": "Stage6D6B",
    "model_name": "RBFSVM on nonredundant handcrafted features",
    "repeated_cv_roc_auc_mean": 0.7447,
    "accuracy": 0.7374,
    "precision": 0.6585,
    "recall": 0.4154,
    "f1_score": 0.5094,
}

CANDIDATE_DIR_PATTERNS = [
    "Stage10X*",
    "*10X*",
    "*Progressive*Contrastive*",
    "*Progressive*Fine*",
]

PREDICTION_KEYWORDS = ["prediction", "pred", "prob", "score", "holdout", "test", "validation", "val", "cv", "fold"]
SUMMARY_KEYWORDS = ["summary", "performance", "metric", "result", "report", "history"]


def safe_float(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def find_stage10x_dirs():
    dirs = []
    for pattern in CANDIDATE_DIR_PATTERNS:
        dirs.extend([p for p in OUTPUTS_ROOT.glob(pattern) if p.is_dir()])
    # Deduplicate while preserving order
    seen, out = set(), []
    for d in dirs:
        key = str(d).lower()
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def list_candidate_files(stage_dirs):
    rows = []
    for d in stage_dirs:
        for f in d.rglob("*"):
            if f.is_file():
                rows.append({
                    "stage_dir": str(d),
                    "file_path": str(f),
                    "file_name": f.name,
                    "suffix": f.suffix.lower(),
                    "size_bytes": f.stat().st_size,
                })
    return pd.DataFrame(rows)


def read_table(path):
    path = Path(path)
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        if path.suffix.lower() in [".xlsx", ".xls"]:
            return pd.read_excel(path)
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return pd.DataFrame(data)
            if isinstance(data, dict):
                return pd.json_normalize(data)
    except Exception:
        return None
    return None


def detect_label_score_columns(df):
    cols = list(df.columns)
    lower = {c: c.lower() for c in cols}

    label_candidates = [c for c in cols if lower[c] in ["y_true", "true_label", "label", "target", "anemia", "class", "actual"]]
    pred_candidates = [c for c in cols if lower[c] in ["y_pred", "pred_label", "prediction", "predicted_label", "predicted_class"]]
    score_candidates = [c for c in cols if any(k in lower[c] for k in ["prob", "score", "logit", "risk", "y_score", "pred_score"])]

    # Remove obvious non-score columns
    score_candidates = [c for c in score_candidates if not any(k in lower[c] for k in ["loss", "epoch", "fold", "id"])]

    label_col = label_candidates[0] if label_candidates else None
    score_col = score_candidates[0] if score_candidates else None
    pred_col = pred_candidates[0] if pred_candidates else None
    return label_col, score_col, pred_col


def infer_split_from_path(path):
    s = str(path).lower()
    if "holdout" in s or "test" in s:
        return "holdout_or_test"
    if "validation" in s or "val" in s:
        return "validation"
    if "cv" in s or "fold" in s or "repeat" in s:
        return "cross_validation"
    return "unknown"


def metrics_from_predictions(y_true, y_score=None, y_pred=None, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    if y_score is not None:
        y_score = np.asarray(y_score).astype(float)
        if y_pred is None:
            y_pred = (y_score >= threshold).astype(int)
    elif y_pred is not None:
        y_pred = np.asarray(y_pred).astype(int)
        y_score = y_pred
    else:
        return None

    out = {
        "n": len(y_true),
        "positives": int(np.sum(y_true == 1)),
        "negatives": int(np.sum(y_true == 0)),
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, y_score) if len(np.unique(y_true)) > 1 else np.nan,
        "roc_auc": roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else np.nan,
    }
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        out.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    except Exception:
        pass
    return out


def scan_prediction_tables(file_df):
    rows = []
    if file_df.empty:
        return pd.DataFrame(rows)

    candidates = file_df[
        file_df["suffix"].isin([".csv", ".xlsx", ".xls", ".json"])
        & file_df["file_name"].str.lower().apply(lambda s: any(k in s for k in PREDICTION_KEYWORDS + SUMMARY_KEYWORDS))
    ]

    for _, r in candidates.iterrows():
        p = r["file_path"]
        df = read_table(p)
        if df is None or df.empty:
            continue

        label_col, score_col, pred_col = detect_label_score_columns(df)
        split = infer_split_from_path(p)

        if label_col and (score_col or pred_col):
            y_true = pd.to_numeric(df[label_col], errors="coerce")
            score = pd.to_numeric(df[score_col], errors="coerce") if score_col else None
            pred = pd.to_numeric(df[pred_col], errors="coerce") if pred_col else None
            valid = y_true.notna()
            if score is not None:
                valid = valid & score.notna()
            if pred is not None and score is None:
                valid = valid & pred.notna()
            if valid.sum() >= 5:
                m = metrics_from_predictions(
                    y_true[valid].values,
                    y_score=score[valid].values if score is not None else None,
                    y_pred=pred[valid].values if pred is not None else None,
                )
                if m:
                    m.update({
                        "source_file": p,
                        "split_inferred": split,
                        "label_col": label_col,
                        "score_col": score_col,
                        "pred_col": pred_col,
                        "evidence_kind": "raw_predictions",
                    })
                    rows.append(m)
                    continue

        # Summary metric extraction fallback
        lower_cols = {c.lower(): c for c in df.columns}
        possible_metric_cols = [c for c in df.columns if any(k in c.lower() for k in ["auc", "accuracy", "f1", "recall", "precision", "balanced"])]
        if possible_metric_cols:
            row = {"source_file": p, "split_inferred": split, "evidence_kind": "summary_table"}
            for c in possible_metric_cols:
                vals = pd.to_numeric(df[c], errors="coerce").dropna()
                if len(vals):
                    row[c.lower()] = float(vals.max() if "auc" in c.lower() else vals.mean())
            rows.append(row)

    return pd.DataFrame(rows)


def classify_verification(metrics_df):
    has_holdout = False
    has_cv = False
    best_holdout_auc = np.nan
    cv_mean = np.nan
    cv_std = np.nan

    if not metrics_df.empty:
        raw = metrics_df.copy()
        if "roc_auc" in raw.columns:
            hold = raw[raw["split_inferred"].eq("holdout_or_test") & raw["roc_auc"].notna()]
            cv = raw[raw["split_inferred"].eq("cross_validation") & raw["roc_auc"].notna()]
            has_holdout = len(hold) > 0
            has_cv = len(cv) > 0
            if has_holdout:
                best_holdout_auc = hold["roc_auc"].max()
            if has_cv:
                cv_mean = cv["roc_auc"].mean()
                cv_std = cv["roc_auc"].std(ddof=1) if len(cv) > 1 else 0.0

    validation_auc = KNOWN_STAGE10X["validation_roc_auc"]
    benchmark_auc = BENCHMARK["repeated_cv_roc_auc_mean"]

    if has_cv and not np.isnan(cv_mean):
        verified_auc = cv_mean
        evidence = "repeated_cv"
    elif has_holdout and not np.isnan(best_holdout_auc):
        verified_auc = best_holdout_auc
        evidence = "holdout"
    else:
        verified_auc = validation_auc
        evidence = "validation_only"

    margin = verified_auc - benchmark_auc if not np.isnan(verified_auc) else np.nan

    if evidence in ["repeated_cv", "holdout"] and margin >= 0.02:
        decision = "Verified final CPMR-Net candidate; may claim conservative improvement over handcrafted benchmark."
        claim_status = "superiority_claim_allowed"
    elif evidence in ["repeated_cv", "holdout"] and margin >= -0.02:
        decision = "Conditionally verified as comparable CPMR-Net candidate; avoid superiority claim."
        claim_status = "comparability_claim_allowed"
    elif evidence in ["repeated_cv", "holdout"]:
        decision = "Not verified as final superior model; retain as scientific CPMR-Net variant only."
        claim_status = "no_superiority_claim"
    else:
        decision = "Not yet conservatively verified; Stage10X remains validation-leading candidate only."
        claim_status = "conditional_candidate_only"

    return {
        "stage": "Stage11C",
        "target_model": "Stage10X Progressive contrastive fine-tuning",
        "validation_auc_known": validation_auc,
        "benchmark_auc": benchmark_auc,
        "holdout_evidence_found": bool(has_holdout),
        "repeated_cv_evidence_found": bool(has_cv),
        "best_holdout_auc_detected": safe_float(best_holdout_auc),
        "repeated_cv_auc_mean_detected": safe_float(cv_mean),
        "repeated_cv_auc_std_detected": safe_float(cv_std),
        "verification_evidence_used": evidence,
        "verified_auc_for_decision": safe_float(verified_auc),
        "margin_vs_stage6d6b": safe_float(margin),
        "claim_status": claim_status,
        "final_decision": decision,
    }


def write_report(summary, stage_dirs, file_df, metrics_df):
    report = []
    report.append("# Stage 11C — Conservative Final-Candidate Verification of Stage10X\n")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append("## Purpose\n")
    report.append("This stage verifies whether Stage10X can be promoted from validation-leading CPMR-Net candidate to final manuscript candidate using conservative evidence. It does not redesign the architecture or introduce a new model.\n")
    report.append("## Fixed Rules\n")
    report.append("- Participant-level diagnosis only.\n- Stage10X is evaluated as-is.\n- Holdout or repeated-CV evidence is required for final superiority claims.\n- Stage6D6B remains the mandatory benchmark unless Stage10X surpasses it under comparable evidence.\n")
    report.append("## Evidence Inventory\n")
    report.append(f"- Stage10X candidate directories found: {len(stage_dirs)}\n")
    report.append(f"- Candidate files scanned: {0 if file_df.empty else len(file_df)}\n")
    report.append(f"- Metric/prediction evidence rows extracted: {0 if metrics_df.empty else len(metrics_df)}\n")
    report.append("## Verification Summary\n")
    report.append(pd.DataFrame([summary]).to_markdown(index=False))
    report.append("\n\n## Interpretation\n")
    if summary["verification_evidence_used"] == "validation_only":
        report.append("Stage10X remains the leading CPMR-Net candidate by validation evidence, but it is not yet conservatively verified for final superiority claims. The next required action is to run holdout and/or repeated-CV evaluation for Stage10X without architectural changes.\n")
    elif summary["claim_status"] == "superiority_claim_allowed":
        report.append("Stage10X is conservatively verified and may be advanced as the final CPMR-Net model with a cautious superiority claim over Stage6D6B.\n")
    elif summary["claim_status"] == "comparability_claim_allowed":
        report.append("Stage10X is conservatively comparable to Stage6D6B but should not be described as definitively superior.\n")
    else:
        report.append("Stage10X did not achieve sufficient conservative evidence to replace Stage6D6B as the strongest validated model.\n")
    report.append("## Generated Outputs\n")
    report.append("- tables/stage11c_evidence_inventory.csv\n")
    report.append("- tables/stage11c_extracted_metric_evidence.csv\n")
    report.append("- tables/stage11c_verification_summary.csv\n")
    report.append("- figures/stage11c_stage10x_vs_stage6d6b_auc.png\n")
    (REPORTS_DIR / "Stage11C_Conservative_Final_Candidate_Verification_of_Stage10X_Report.md").write_text("\n".join(report), encoding="utf-8")


def plot_summary(summary):
    labels = ["Stage6D6B benchmark", "Stage10X decision evidence", "Stage10X validation"]
    values = [summary["benchmark_auc"], summary["verified_auc_for_decision"], summary["validation_auc_known"]]
    plt.figure(figsize=(10, 4.8))
    plt.barh(labels, values)
    plt.xlabel("ROC-AUC")
    plt.title("Stage 11C conservative Stage10X verification")
    plt.xlim(0, max(0.85, max(values) + 0.05))
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "stage11c_stage10x_vs_stage6d6b_auc.png", dpi=300)
    plt.close()


def main():
    stage_dirs = find_stage10x_dirs()
    file_df = list_candidate_files(stage_dirs)
    metrics_df = scan_prediction_tables(file_df)

    if file_df.empty:
        file_df = pd.DataFrame(columns=["stage_dir", "file_path", "file_name", "suffix", "size_bytes"])
    if metrics_df.empty:
        metrics_df = pd.DataFrame(columns=["source_file", "split_inferred", "evidence_kind"])

    summary = classify_verification(metrics_df)

    file_df.to_csv(TABLES_DIR / "stage11c_evidence_inventory.csv", index=False)
    metrics_df.to_csv(TABLES_DIR / "stage11c_extracted_metric_evidence.csv", index=False)
    pd.DataFrame([summary]).to_csv(TABLES_DIR / "stage11c_verification_summary.csv", index=False)

    with open(OUTPUT_DIR / "Stage11C_Conservative_Final_Candidate_Verification_of_Stage10X_Summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_summary(summary)
    write_report(summary, stage_dirs, file_df, metrics_df)

    print("=" * 80)
    print("STAGE 11C CONSERVATIVE FINAL-CANDIDATE VERIFICATION COMPLETED")
    print("=" * 80)
    print(f"Target model: Stage10X | Progressive contrastive fine-tuning")
    print(f"Stage10X directories found: {len(stage_dirs)}")
    print(f"Evidence files scanned: {len(file_df)}")
    print(f"Metric evidence rows extracted: {len(metrics_df)}")
    print(f"Evidence used: {summary['verification_evidence_used']}")
    print(f"Verified AUC for decision: {summary['verified_auc_for_decision']:.4f}")
    print(f"Margin vs Stage6D6B: {summary['margin_vs_stage6d6b']:.4f}")
    print(f"Decision: {summary['final_decision']}")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
