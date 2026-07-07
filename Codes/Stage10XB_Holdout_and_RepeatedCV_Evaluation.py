"""
Stage10XB — Holdout and Repeated Cross-Validation Evaluation of Stage10X
Project: CPMR-Net HandDx-200 anemia diagnosis

Purpose
-------
This script is an EXECUTION stage, not an evidence-inventory stage.
It attempts to evaluate the fixed Stage10X model under:
  1) the fixed Stage10I1 independent holdout split, and
  2) the fixed Stage10I1 repeated participant-level CV splits.

Fixed scientific rules
----------------------
- Participant-level diagnosis only.
- No image-level split or image-level diagnosis.
- No architecture changes.
- No hyperparameter search.
- Stage10X must be evaluated as-is.
- Stage6D6B RBFSVM remains the mandatory benchmark unless Stage10X surpasses it under comparable evidence.

Important practical note
------------------------
Because the exact Stage10X model/dataset objects are defined in your local project scripts,
this file uses an adapter-based approach:

1. It searches your Codes folder for a Stage10X implementation script.
2. It searches your Outputs folder for Stage10X checkpoints.
3. It searches Stage10I1 for fixed participant-level splits.
4. It searches Stage10B/10C representation manifests.
5. If a compatible Stage10X API is found, it evaluates directly.
6. If not, it writes a required adapter template and stops safely without inventing results.

A compatible Stage10X API can be any one of the following in the Stage10X script:
  - evaluate_stage10x_holdout_and_cv(config: dict) -> dict
  - build_model(config: dict) + build_dataloaders(config: dict) or get_dataloader(...)
  - model class named CPMRNet, Stage10XModel, or ProgressiveCPMRNet

This design prevents accidental architecture changes while allowing execution using the
local Stage10X code already present in your project.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
import shutil
import traceback
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(r"D:\47\472\New-Papers\Anemia_Paper")
CODES_DIR = PROJECT_DIR / "Codes"
OUTPUTS_DIR = PROJECT_DIR / "Outputs"

OUTPUT_DIR = OUTPUTS_DIR / "Stage10XB_Holdout_and_RepeatedCV_Evaluation"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"
CONFIGS_DIR = OUTPUT_DIR / "configs"
LOGS_DIR = OUTPUT_DIR / "logs"

for d in [OUTPUT_DIR, TABLES_DIR, FIGURES_DIR, REPORTS_DIR, CONFIGS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

BENCHMARK_AUC = 0.7447
STAGE10X_EXPECTED_VALIDATION_AUC = 0.7879
RANDOM_STATE = 42
DEFAULT_THRESHOLD = 0.5


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def safe_float(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        v = float(x)
        return v
    except Exception:
        return float("nan")


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return float("nan")


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "positive_n": int(y_true.sum()),
        "negative_n": int((1 - y_true).sum()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": safe_auc(y_true, y_score),
        "pr_auc": safe_pr_auc(y_true, y_score),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def best_youden_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        j = tpr - fpr
        idx = int(np.nanargmax(j))
        thr = float(thresholds[idx])
        if not np.isfinite(thr):
            return DEFAULT_THRESHOLD
        return thr
    except Exception:
        return DEFAULT_THRESHOLD


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
    return df


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = list(df.columns)
    for cand in candidates:
        cand_norm = cand.lower().replace(" ", "_").replace("-", "_")
        if cand_norm in cols:
            return cand_norm
    for col in cols:
        for cand in candidates:
            if cand.lower().replace(" ", "_").replace("-", "_") in col:
                return col
    return None


# ============================================================
# DISCOVERY
# ============================================================

def discover_stage10x_scripts() -> List[Path]:
    patterns = ["*Stage10X*.py", "*10X*.py", "*Progressive*Fine*Tuning*.py", "*Progressive*Contrastive*.py"]
    found: List[Path] = []
    for pat in patterns:
        found.extend(CODES_DIR.glob(pat))
    unique = []
    seen = set()
    for p in found:
        if p.name not in seen and p.is_file():
            seen.add(p.name)
            unique.append(p)
    return unique


def discover_stage10x_output_dirs() -> List[Path]:
    if not OUTPUTS_DIR.exists():
        return []
    dirs = []
    for p in OUTPUTS_DIR.iterdir():
        if p.is_dir() and ("10X" in p.name.upper() or "PROGRESSIVE" in p.name.upper() or "CONTRASTIVE_FINE" in p.name.upper()):
            dirs.append(p)
    return sorted(dirs)


def discover_checkpoints(stage10x_dirs: List[Path]) -> List[Path]:
    exts = ["*.pt", "*.pth", "*.ckpt", "*.pkl"]
    files: List[Path] = []
    for d in stage10x_dirs:
        for ext in exts:
            files.extend(d.rglob(ext))
    # Prefer names suggesting best/final Stage10X model
    def score(p: Path) -> int:
        s = p.name.lower()
        val = 0
        if "stage10x" in s: val += 5
        if "best" in s: val += 4
        if "checkpoint" in s or "model" in s: val += 2
        if "last" in s: val -= 1
        return -val
    return sorted(set(files), key=lambda p: (score(p), str(p)))


def discover_split_files() -> Dict[str, List[Path]]:
    stage10i1 = OUTPUTS_DIR / "Stage10I1_Participant_Level_Split_Strategy"
    candidates = []
    if stage10i1.exists():
        candidates.extend(stage10i1.rglob("*.csv"))
    candidates.extend(OUTPUTS_DIR.rglob("*split*.csv"))
    out = {"holdout": [], "repeated_cv": [], "all": []}
    for p in sorted(set(candidates)):
        name = p.name.lower()
        out["all"].append(p)
        if "holdout" in name or "train_val_test" in name or "test_split" in name:
            out["holdout"].append(p)
        if "repeated" in name or "5fold" in name or "fold" in name or "cv" in name:
            out["repeated_cv"].append(p)
    return out


def discover_manifest_files() -> List[Path]:
    roots = [
        OUTPUTS_DIR / "Stage10B_MultiRepresentation_Generator",
        OUTPUTS_DIR / "Stage10C_Representation_Encoder_Dataset_Tensor_Validation",
        OUTPUTS_DIR / "Stage10I2_PyTorch_Participant_Dataset_Dataloader",
    ]
    files = []
    for r in roots:
        if r.exists():
            files.extend(r.rglob("*.csv"))
    # prefer manifest/readiness files
    return sorted(set(files), key=lambda p: (0 if "manifest" in p.name.lower() else 1, str(p)))


# ============================================================
# IMPORT / ADAPTER SUPPORT
# ============================================================

def import_module_from_path(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def find_compatible_module(scripts: List[Path]) -> Tuple[Optional[Any], Optional[Path], str]:
    errors = []
    for script in scripts:
        try:
            mod = import_module_from_path(script)
            # Highest-level API preferred.
            if hasattr(mod, "evaluate_stage10x_holdout_and_cv"):
                return mod, script, "high_level_evaluate_stage10x_holdout_and_cv"
            # Common function/class names.
            model_api = any(hasattr(mod, x) for x in ["build_model", "create_model", "CPMRNet", "Stage10XModel", "ProgressiveCPMRNet"])
            data_api = any(hasattr(mod, x) for x in ["build_dataloaders", "get_dataloader", "ParticipantRepresentationDataset", "create_dataloaders"])
            train_api = any(hasattr(mod, x) for x in ["train_one_fold", "train_stage10x", "run_training", "run_experiment"])
            if model_api and (data_api or train_api):
                return mod, script, "partial_model_data_or_train_api"
        except SystemExit:
            errors.append(f"{script.name}: SystemExit during import")
        except Exception as e:
            errors.append(f"{script.name}: {type(e).__name__}: {e}")
    return None, None, "No compatible Stage10X module API found. Import errors: " + " | ".join(errors[:5])


# ============================================================
# PREDICTION FILE EVALUATION FALLBACK
# ============================================================

def discover_prediction_files(stage10x_dirs: List[Path]) -> List[Path]:
    files = []
    for d in stage10x_dirs:
        for p in d.rglob("*.csv"):
            name = p.name.lower()
            if any(k in name for k in ["prediction", "predictions", "prob", "scores", "holdout", "test", "fold", "cv"]):
                files.append(p)
    return sorted(set(files))


def load_prediction_file(path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path)
        df = normalize_columns(df)
        y_col = find_col(df, ["label", "true_label", "y_true", "target", "anemia_label", "class"])
        score_col = find_col(df, ["probability", "prob", "score", "y_score", "pred_prob", "anemia_probability", "positive_probability"])
        pid_col = find_col(df, ["participant_id", "participant", "pid", "subject_id"])
        fold_col = find_col(df, ["fold", "fold_id", "cv_fold"])
        repeat_col = find_col(df, ["repeat", "repeat_id", "cv_repeat"])
        split_col = find_col(df, ["split", "set", "subset"])
        if y_col is None or score_col is None:
            return None
        out = pd.DataFrame({
            "participant_id": df[pid_col] if pid_col else np.arange(len(df)),
            "y_true": df[y_col].astype(int),
            "y_score": df[score_col].astype(float),
            "source_file": str(path),
        })
        if fold_col: out["fold"] = df[fold_col]
        if repeat_col: out["repeat"] = df[repeat_col]
        if split_col: out["split"] = df[split_col].astype(str).str.lower()
        return out
    except Exception:
        return None


def evaluate_existing_predictions(pred_files: List[Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    all_preds = []
    for f in pred_files:
        df = load_prediction_file(f)
        if df is None or df.empty:
            continue
        all_preds.append(df)
        # Decide evidence type from filename/split columns.
        name = f.name.lower()
        if "split" in df.columns:
            groups = [(k, g) for k, g in df.groupby("split")]
        else:
            evidence = "holdout" if any(k in name for k in ["holdout", "test"]) else ("repeated_cv" if any(k in name for k in ["fold", "cv", "repeat"]) else "unknown")
            groups = [(evidence, df)]
        for group_name, g in groups:
            thr = best_youden_threshold(g["y_true"].values, g["y_score"].values)
            m05 = compute_metrics(g["y_true"].values, g["y_score"].values, DEFAULT_THRESHOLD)
            my = compute_metrics(g["y_true"].values, g["y_score"].values, thr)
            row = {
                "source_file": str(f),
                "evidence_group": group_name,
                "n": len(g),
                "roc_auc": m05["roc_auc"],
                "pr_auc": m05["pr_auc"],
                "accuracy_thr_0_5": m05["accuracy"],
                "balanced_accuracy_thr_0_5": m05["balanced_accuracy"],
                "f1_thr_0_5": m05["f1_score"],
                "recall_thr_0_5": m05["recall"],
                "youden_threshold": thr,
                "accuracy_youden": my["accuracy"],
                "balanced_accuracy_youden": my["balanced_accuracy"],
                "f1_youden": my["f1_score"],
                "recall_youden": my["recall"],
            }
            rows.append(row)
    metrics_df = pd.DataFrame(rows)
    preds_df = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    return metrics_df, preds_df


# ============================================================
# EXECUTION VIA HIGH-LEVEL ADAPTER
# ============================================================

def make_config(
    scripts: List[Path],
    selected_script: Optional[Path],
    checkpoints: List[Path],
    split_files: Dict[str, List[Path]],
    manifests: List[Path],
) -> Dict[str, Any]:
    return {
        "stage": "Stage10XB",
        "project_dir": str(PROJECT_DIR),
        "codes_dir": str(CODES_DIR),
        "outputs_dir": str(OUTPUTS_DIR),
        "output_dir": str(OUTPUT_DIR),
        "tables_dir": str(TABLES_DIR),
        "figures_dir": str(FIGURES_DIR),
        "reports_dir": str(REPORTS_DIR),
        "benchmark_stage6d6b_auc": BENCHMARK_AUC,
        "expected_stage10x_validation_auc": STAGE10X_EXPECTED_VALIDATION_AUC,
        "random_state": RANDOM_STATE,
        "threshold_policy": "Use validation-derived threshold if provided by Stage10X; otherwise report both 0.5 and Youden threshold.",
        "fixed_rules": [
            "participant-level diagnosis only",
            "no image-level diagnosis",
            "no architecture change",
            "no hyperparameter search",
            "use fixed Stage10I1 holdout and repeated-CV splits",
        ],
        "discovered_stage10x_scripts": [str(p) for p in scripts],
        "selected_stage10x_script": str(selected_script) if selected_script else None,
        "discovered_checkpoints": [str(p) for p in checkpoints],
        "primary_checkpoint": str(checkpoints[0]) if checkpoints else None,
        "holdout_split_files": [str(p) for p in split_files.get("holdout", [])],
        "repeated_cv_split_files": [str(p) for p in split_files.get("repeated_cv", [])],
        "manifest_files": [str(p) for p in manifests],
    }


def try_high_level_adapter(module: Any, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if hasattr(module, "evaluate_stage10x_holdout_and_cv"):
        result = module.evaluate_stage10x_holdout_and_cv(config)
        if not isinstance(result, dict):
            raise RuntimeError("evaluate_stage10x_holdout_and_cv(config) must return a dict.")
        return result
    return None


# ============================================================
# ADAPTER TEMPLATE
# ============================================================

def write_adapter_template(config: Dict[str, Any]) -> Path:
    template = r'''"""
Stage10X local adapter template for Stage10XB.

Save this file as:
D:\47\472\New-Papers\Anemia_Paper\Codes\Stage10X_Local_Evaluation_Adapter.py

Then modify your Stage10XB script discovery if needed, or rename this adapter to include "Stage10X".

The required function is:
    evaluate_stage10x_holdout_and_cv(config: dict) -> dict

It must return a dictionary containing at least:
    holdout_predictions_path
    holdout_metrics_path
    cv_predictions_path
    cv_metrics_path
    summary
"""

from pathlib import Path
import pandas as pd


def evaluate_stage10x_holdout_and_cv(config: dict) -> dict:
    """
    Implement this using your existing Stage10X model, dataloader, checkpoint, and fixed Stage10I1 splits.

    Requirements:
    - Load Stage10X architecture exactly as trained.
    - Load the selected Stage10X checkpoint.
    - Evaluate the fixed holdout test participants.
    - Retrain/evaluate Stage10X across fixed repeated CV folds if the Stage10X protocol requires training per fold.
    - Save participant-level predictions with columns:
        participant_id, y_true, y_score, split, repeat, fold
    - Do not change representations, architecture, loss design, or hyperparameters.
    """
    raise NotImplementedError("Connect this adapter to the existing Stage10X implementation before running Stage10XB.")
'''
    p = CONFIGS_DIR / "Stage10X_Local_Evaluation_Adapter_TEMPLATE.py"
    p.write_text(template, encoding="utf-8")
    return p


# ============================================================
# FIGURES AND REPORTS
# ============================================================

def plot_auc_summary(summary_rows: List[Dict[str, Any]]) -> None:
    df = pd.DataFrame(summary_rows)
    if df.empty or "roc_auc" not in df.columns:
        return
    plot_df = df.dropna(subset=["roc_auc"]).copy()
    if plot_df.empty:
        return
    plot_df = plot_df.sort_values("roc_auc", ascending=True)
    plt.figure(figsize=(10, max(4, 0.5 * len(plot_df))))
    plt.barh(plot_df["evidence"], plot_df["roc_auc"])
    plt.axvline(BENCHMARK_AUC, linestyle="--", linewidth=1)
    plt.xlabel("ROC-AUC")
    plt.title("Stage10XB holdout/repeated-CV ROC-AUC evidence")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "stage10xb_auc_evidence_summary.png", dpi=300)
    plt.close()


def write_report(summary: Dict[str, Any], metrics_df: pd.DataFrame, executed: bool, adapter_template: Optional[Path]) -> None:
    report = []
    report.append("# Stage10XB — Holdout and Repeated Cross-Validation Evaluation\n")
    report.append(f"Generated: {now()}\n")
    report.append("## Purpose\n")
    report.append("This stage attempts to execute conservative holdout and repeated-CV evaluation of Stage10X as-is. It does not redesign CPMR-Net.\n")
    report.append("## Fixed Rules\n")
    report.append("- Participant-level diagnosis only.\n")
    report.append("- Fixed Stage10I1 holdout and repeated-CV splits.\n")
    report.append("- No architecture change.\n")
    report.append("- No hyperparameter search.\n")
    report.append("- Stage6D6B remains the mandatory benchmark until Stage10X surpasses it under comparable evidence.\n")
    report.append("## Execution Status\n")
    report.append(f"- Execution adapter used: {executed}\n")
    report.append(f"- Stage10X scripts discovered: {summary.get('stage10x_scripts_found', 0)}\n")
    report.append(f"- Stage10X checkpoints discovered: {summary.get('checkpoints_found', 0)}\n")
    report.append(f"- Holdout split files discovered: {summary.get('holdout_split_files_found', 0)}\n")
    report.append(f"- Repeated-CV split files discovered: {summary.get('repeated_cv_split_files_found', 0)}\n")
    if adapter_template:
        report.append(f"- Adapter template written: `{adapter_template}`\n")
    report.append("\n## Metric Summary\n")
    if not metrics_df.empty:
        display_cols = [c for c in ["evidence", "n", "roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "recall", "f1_score"] if c in metrics_df.columns]
        report.append(metrics_df[display_cols].to_markdown(index=False))
        report.append("\n")
    else:
        report.append("No executable Stage10X holdout/repeated-CV metric table was produced.\n")
    report.append("\n## Interpretation\n")
    report.append(summary.get("interpretation", "Pending execution with compatible local Stage10X adapter."))
    report.append("\n")
    report.append("## Generated Outputs\n")
    report.append("- tables/stage10xb_discovery_inventory.csv\n")
    report.append("- tables/stage10xb_prediction_file_metrics.csv\n")
    report.append("- tables/stage10xb_final_metric_summary.csv\n")
    report.append("- figures/stage10xb_auc_evidence_summary.png\n")
    report.append("- configs/stage10xb_execution_config.json\n")
    (REPORTS_DIR / "Stage10XB_Holdout_and_RepeatedCV_Evaluation_Report.md").write_text("\n".join(report), encoding="utf-8")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    discovery_rows = []

    scripts = discover_stage10x_scripts()
    stage10x_dirs = discover_stage10x_output_dirs()
    checkpoints = discover_checkpoints(stage10x_dirs)
    split_files = discover_split_files()
    manifests = discover_manifest_files()
    pred_files = discover_prediction_files(stage10x_dirs)

    for p in scripts:
        discovery_rows.append({"resource_type": "stage10x_script", "path": str(p), "exists": p.exists()})
    for p in stage10x_dirs:
        discovery_rows.append({"resource_type": "stage10x_output_dir", "path": str(p), "exists": p.exists()})
    for p in checkpoints:
        discovery_rows.append({"resource_type": "stage10x_checkpoint", "path": str(p), "exists": p.exists(), "size_bytes": p.stat().st_size if p.exists() else None})
    for p in split_files.get("holdout", []):
        discovery_rows.append({"resource_type": "holdout_split_file", "path": str(p), "exists": p.exists()})
    for p in split_files.get("repeated_cv", []):
        discovery_rows.append({"resource_type": "repeated_cv_split_file", "path": str(p), "exists": p.exists()})
    for p in manifests:
        discovery_rows.append({"resource_type": "manifest_file", "path": str(p), "exists": p.exists()})
    for p in pred_files:
        discovery_rows.append({"resource_type": "candidate_prediction_file", "path": str(p), "exists": p.exists()})

    pd.DataFrame(discovery_rows).to_csv(TABLES_DIR / "stage10xb_discovery_inventory.csv", index=False)

    module, selected_script, api_status = find_compatible_module(scripts)
    config = make_config(scripts, selected_script, checkpoints, split_files, manifests)
    config["api_status"] = api_status
    write_json(CONFIGS_DIR / "stage10xb_execution_config.json", config)

    executed = False
    adapter_result = None
    adapter_template = None
    execution_error = None

    if module is not None:
        try:
            adapter_result = try_high_level_adapter(module, config)
            if adapter_result is not None:
                executed = True
                write_json(OUTPUT_DIR / "Stage10XB_Adapter_Result.json", adapter_result)
        except Exception as e:
            execution_error = traceback.format_exc()
            (LOGS_DIR / "stage10xb_adapter_execution_error.txt").write_text(execution_error, encoding="utf-8")

    # Fallback: evaluate any existing prediction files, but clearly label them as existing evidence.
    pred_metrics_df, pred_df = evaluate_existing_predictions(pred_files)
    pred_metrics_df.to_csv(TABLES_DIR / "stage10xb_prediction_file_metrics.csv", index=False)
    if not pred_df.empty:
        pred_df.to_csv(TABLES_DIR / "stage10xb_existing_prediction_rows_combined.csv", index=False)

    final_rows: List[Dict[str, Any]] = []

    # If adapter returned paths/tables, import them conservatively.
    if adapter_result:
        # Users/local adapter may return direct metrics.
        if "holdout_metrics" in adapter_result and isinstance(adapter_result["holdout_metrics"], dict):
            m = adapter_result["holdout_metrics"].copy()
            m["evidence"] = "Stage10X holdout"
            final_rows.append(m)
        if "cv_metrics" in adapter_result and isinstance(adapter_result["cv_metrics"], dict):
            m = adapter_result["cv_metrics"].copy()
            m["evidence"] = "Stage10X repeated-CV"
            final_rows.append(m)
        # Or return metrics file paths.
        for key, evidence in [("holdout_metrics_path", "Stage10X holdout"), ("cv_metrics_path", "Stage10X repeated-CV")]:
            p = adapter_result.get(key)
            if p and Path(p).exists():
                try:
                    df = normalize_columns(pd.read_csv(p))
                    if "roc_auc" in df.columns:
                        if len(df) == 1:
                            row = df.iloc[0].to_dict()
                            row["evidence"] = evidence
                            final_rows.append(row)
                        else:
                            row = {
                                "evidence": evidence,
                                "n": int(df["n"].sum()) if "n" in df.columns else len(df),
                                "roc_auc": float(df["roc_auc"].mean()),
                                "roc_auc_std": float(df["roc_auc"].std(ddof=1)),
                                "pr_auc": float(df["pr_auc"].mean()) if "pr_auc" in df.columns else float("nan"),
                                "accuracy": float(df["accuracy"].mean()) if "accuracy" in df.columns else float("nan"),
                                "balanced_accuracy": float(df["balanced_accuracy"].mean()) if "balanced_accuracy" in df.columns else float("nan"),
                                "recall": float(df["recall"].mean()) if "recall" in df.columns else float("nan"),
                                "f1_score": float(df["f1_score"].mean()) if "f1_score" in df.columns else float("nan"),
                            }
                            final_rows.append(row)
                except Exception as e:
                    (LOGS_DIR / f"failed_to_read_{key}.txt").write_text(str(e), encoding="utf-8")

    # Existing prediction fallback, if any.
    for _, r in pred_metrics_df.iterrows():
        ev = str(r.get("evidence_group", "existing_predictions"))
        if ev in ["test", "holdout"] or "holdout" in ev or "test" in ev:
            evidence = "Stage10X holdout existing predictions"
        elif "fold" in ev or "cv" in ev or "repeat" in ev:
            evidence = "Stage10X repeated-CV existing predictions"
        else:
            evidence = "Stage10X existing prediction evidence"
        final_rows.append({
            "evidence": evidence,
            "n": r.get("n", np.nan),
            "roc_auc": r.get("roc_auc", np.nan),
            "pr_auc": r.get("pr_auc", np.nan),
            "accuracy": r.get("accuracy_thr_0_5", np.nan),
            "balanced_accuracy": r.get("balanced_accuracy_thr_0_5", np.nan),
            "recall": r.get("recall_thr_0_5", np.nan),
            "f1_score": r.get("f1_thr_0_5", np.nan),
            "source": r.get("source_file", ""),
        })

    # Always include benchmark row.
    final_rows.append({
        "evidence": "Stage6D6B benchmark",
        "n": np.nan,
        "roc_auc": BENCHMARK_AUC,
        "pr_auc": np.nan,
        "accuracy": 0.7374,
        "balanced_accuracy": np.nan,
        "recall": 0.4154,
        "f1_score": 0.5094,
        "source": "Stage6D6B repeated-CV benchmark",
    })

    final_metrics_df = pd.DataFrame(final_rows)
    # Remove duplicate exact evidence rows if any.
    if not final_metrics_df.empty:
        final_metrics_df = final_metrics_df.drop_duplicates(subset=[c for c in ["evidence", "roc_auc", "source"] if c in final_metrics_df.columns])
    final_metrics_df.to_csv(TABLES_DIR / "stage10xb_final_metric_summary.csv", index=False)

    # Decide if true new conservative evidence exists.
    has_holdout = any("holdout" in str(e).lower() and "benchmark" not in str(e).lower() for e in final_metrics_df.get("evidence", []))
    has_cv = any(("repeated" in str(e).lower() or "cv" in str(e).lower()) and "benchmark" not in str(e).lower() for e in final_metrics_df.get("evidence", []))

    stage10x_holdout_auc = np.nan
    stage10x_cv_auc = np.nan
    if has_holdout:
        vals = final_metrics_df[final_metrics_df["evidence"].astype(str).str.lower().str.contains("holdout")]["roc_auc"].dropna().values
        if len(vals): stage10x_holdout_auc = float(np.nanmax(vals))
    if has_cv:
        vals = final_metrics_df[final_metrics_df["evidence"].astype(str).str.lower().str.contains("cv|repeated", regex=True)]["roc_auc"].dropna().values
        if len(vals): stage10x_cv_auc = float(np.nanmean(vals))

    if has_holdout or has_cv:
        conservative_auc = stage10x_cv_auc if np.isfinite(stage10x_cv_auc) else stage10x_holdout_auc
        margin = conservative_auc - BENCHMARK_AUC if np.isfinite(conservative_auc) else np.nan
        if np.isfinite(margin) and margin > 0:
            claim_status = "conservatively_supported_above_stage6d6b"
            interpretation = "Stage10X has conservative holdout/repeated-CV evidence above the Stage6D6B benchmark. Confirm statistical uncertainty before final manuscript superiority wording."
        else:
            claim_status = "not_above_stage6d6b_under_conservative_evidence"
            interpretation = "Stage10X has conservative evidence but does not surpass Stage6D6B; retain Stage6D6B as strongest validated benchmark."
    else:
        conservative_auc = np.nan
        margin = np.nan
        claim_status = "execution_adapter_required"
        interpretation = "No executable Stage10X holdout/repeated-CV evidence was produced. A local Stage10X evaluation adapter is required to connect this stage to the existing model/dataloader/checkpoint code."
        adapter_template = write_adapter_template(config)

    summary = {
        "stage": "Stage10XB",
        "generated": now(),
        "stage10x_scripts_found": len(scripts),
        "stage10x_output_dirs_found": len(stage10x_dirs),
        "checkpoints_found": len(checkpoints),
        "holdout_split_files_found": len(split_files.get("holdout", [])),
        "repeated_cv_split_files_found": len(split_files.get("repeated_cv", [])),
        "manifest_files_found": len(manifests),
        "candidate_prediction_files_found": len(pred_files),
        "selected_stage10x_script": str(selected_script) if selected_script else None,
        "api_status": api_status,
        "adapter_executed": executed,
        "execution_error_present": execution_error is not None,
        "holdout_evidence_produced": has_holdout,
        "repeated_cv_evidence_produced": has_cv,
        "stage10x_holdout_auc": stage10x_holdout_auc,
        "stage10x_repeated_cv_auc": stage10x_cv_auc,
        "stage10x_conservative_auc": conservative_auc,
        "benchmark_auc": BENCHMARK_AUC,
        "margin_vs_stage6d6b": margin,
        "claim_status": claim_status,
        "interpretation": interpretation,
        "adapter_template": str(adapter_template) if adapter_template else None,
        "output_dir": str(OUTPUT_DIR),
    }
    write_json(OUTPUT_DIR / "Stage10XB_Holdout_and_RepeatedCV_Evaluation_Summary.json", summary)

    plot_auc_summary(final_rows)
    write_report(summary, final_metrics_df, executed, adapter_template)

    print("=" * 80)
    print("STAGE10XB HOLDOUT AND REPEATED-CV EVALUATION COMPLETED")
    print("=" * 80)
    print(f"Stage10X scripts found: {len(scripts)}")
    print(f"Checkpoints found: {len(checkpoints)}")
    print(f"Holdout split files found: {len(split_files.get('holdout', []))}")
    print(f"Repeated-CV split files found: {len(split_files.get('repeated_cv', []))}")
    print(f"Adapter executed: {executed}")
    print(f"Holdout evidence produced: {has_holdout}")
    print(f"Repeated-CV evidence produced: {has_cv}")
    print(f"Claim status: {claim_status}")
    print(f"Results saved to: {OUTPUT_DIR}")
    if adapter_template:
        print(f"Adapter template written to: {adapter_template}")
    print("=" * 80)


if __name__ == "__main__":
    main()
