# -*- coding: utf-8 -*-
r"""
Stage 10XD - Independent Holdout and Repeated-CV Evaluator for Stage10X

Purpose
-------
Conservatively evaluate the completed Stage10X Progressive Contrastive Fine-Tuning
CPMR-Net model without changing its architecture, representation set, loss design, or
hyperparameters.

This script does NOT modify Stage10X_Progressive_FineTuning_Contrastive_CPMRNet.py.
It imports the already-defined Stage10X model, dataset, collate function, optimizer,
metrics, and training utilities, then performs:

1) Independent holdout evaluation using the saved Stage10X best validation checkpoint.
2) Optional repeated-CV training/evaluation using the fixed Stage10I1 participant-level
   repeated-CV splits and the original Stage10X training protocol.

Important scientific rule
-------------------------
The saved Stage10X checkpoint can only be used for the fixed holdout test split because
it was trained on the fixed holdout train/validation split. Repeated-CV evidence requires
training a fresh Stage10X model per fold using the same protocol.

Expected location
-----------------
Save as:
D:\47\472\New-Papers\Anemia_Paper\Codes\Stage10XD_Independent_Holdout_and_RepeatedCV_Evaluator.py

Run:
python Stage10XD_Independent_Holdout_and_RepeatedCV_Evaluator.py

Environment controls, optional
------------------------------
STAGE10XD_RUN_REPEATED_CV=1       Run repeated CV. Default: 1.
STAGE10XD_MAX_CV_FOLDS=0          0 means all folds. Use e.g. 2 for smoke test.
STAGE10XD_NUM_WORKERS=0           Override dataloader workers. Default: config value.
"""

from pathlib import Path
import os
import sys
import json
import copy
import math
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import roc_curve

warnings.filterwarnings("ignore")


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(r"D:\47\472\New-Papers\Anemia_Paper")
CODES_DIR = BASE_DIR / "Codes"
OUTPUTS_DIR = BASE_DIR / "Outputs"

STAGE10X_SCRIPT = CODES_DIR / "Stage10X_Progressive_FineTuning_Contrastive_CPMRNet.py"
STAGE10X_OUT = OUTPUTS_DIR / "Stage10X_Progressive_FineTuning_Contrastive_CPMRNet"
STAGE10X_BEST_CKPT = STAGE10X_OUT / "models" / "ProgressiveContrastive_CPMRNet_best_val_auc.pt"
STAGE10X_LAST_CKPT = STAGE10X_OUT / "models" / "ProgressiveContrastive_CPMRNet_last.pt"

CONFIG_FILE = OUTPUTS_DIR / "Stage10I4_Training_Configuration_Experiment_Control" / "configs" / "CPMRNet_training_config_v1.json"
LOSS_CONFIG_FILE = OUTPUTS_DIR / "Stage10K_Loss_Functions_Class_Imbalance" / "configs" / "CPMRNet_loss_config_v1.json"
ENCODER_MAP_FILE = OUTPUTS_DIR / "Stage10V_SelfSupervised_Contrastive_Pretraining" / "models" / "contrastive_pretrained_encoder_checkpoint_map.json"

STAGE10I1_DIR = OUTPUTS_DIR / "Stage10I1_Participant_Level_Split_Strategy" / "tables"
HOLDOUT_SPLIT_FILE = STAGE10I1_DIR / "holdout_train_val_test_split.csv"
REPEATED_SPLIT_FILE = STAGE10I1_DIR / "repeated_stratified_5fold_train_val_test_splits.csv"

OUTPUT_DIR = OUTPUTS_DIR / "Stage10XD_Independent_Holdout_and_RepeatedCV_Evaluator"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"
MODELS_DIR = OUTPUT_DIR / "models_cv"
CONFIGS_DIR = OUTPUT_DIR / "configs"

for d in [OUTPUT_DIR, TABLES_DIR, FIGURES_DIR, REPORTS_DIR, MODELS_DIR, CONFIGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# FIXED SETTINGS
# =============================================================================

BENCHMARK_STAGE6D6B_AUC = 0.7447
EXPECTED_STAGE10X_VAL_AUC = 0.7879
RUN_REPEATED_CV = os.environ.get("STAGE10XD_RUN_REPEATED_CV", "1").strip() != "0"
MAX_CV_FOLDS = int(os.environ.get("STAGE10XD_MAX_CV_FOLDS", "0"))  # 0 = all
NUM_WORKERS_OVERRIDE = os.environ.get("STAGE10XD_NUM_WORKERS")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# IMPORT STAGE10X WITHOUT MODIFYING IT
# =============================================================================

if str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))

try:
    import Stage10X_Progressive_FineTuning_Contrastive_CPMRNet as s10x
except Exception as exc:
    raise RuntimeError(
        f"Could not import Stage10X script from {STAGE10X_SCRIPT}. Original error: {repr(exc)}"
    )


# =============================================================================
# HELPERS
# =============================================================================

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: Path):
    def convert(x):
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            if np.isnan(x) or np.isinf(x):
                return None
            return float(x)
        if isinstance(x, (np.ndarray,)):
            return x.tolist()
        if isinstance(x, Path):
            return str(x)
        return x

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False, default=convert)


def normalize_participant_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    candidates = ["participant_id", "participant", "pid", "id", "subject_id"]
    lower_map = {c.lower(): c for c in df.columns}
    if "participant_id" not in df.columns:
        for cand in candidates:
            if cand in lower_map:
                df = df.rename(columns={lower_map[cand]: "participant_id"})
                break
    if "participant_id" not in df.columns:
        raise ValueError(f"No participant_id-like column found. Columns: {list(df.columns)}")
    df["participant_id"] = df["participant_id"].astype(str)
    return df


def normalize_split_df(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_participant_id_columns(df)
    if "split" not in df.columns:
        # Try common alternatives.
        for c in df.columns:
            if c.lower() in ["set", "subset", "partition"]:
                df = df.rename(columns={c: "split"})
                break
    if "split" not in df.columns:
        raise ValueError(f"No split column found. Columns: {list(df.columns)}")
    df["split"] = df["split"].astype(str).str.lower().str.strip()
    # Normalize common names.
    df["split"] = df["split"].replace({
        "validation": "val", "valid": "val", "dev": "val",
        "test_set": "test", "train_set": "train", "val_set": "val"
    })
    return df


def find_repeat_fold_columns(df: pd.DataFrame):
    repeat_col = None
    fold_col = None
    for c in df.columns:
        cl = c.lower()
        if repeat_col is None and cl in ["repeat", "repeat_id", "repeat_index", "cv_repeat", "repetition"]:
            repeat_col = c
        if fold_col is None and cl in ["fold", "fold_id", "fold_index", "cv_fold"]:
            fold_col = c
    if repeat_col is None:
        # Single repeat fallback.
        df = df.copy()
        df["repeat"] = 1
        repeat_col = "repeat"
    if fold_col is None:
        raise ValueError(f"No fold column found in repeated-CV split file. Columns: {list(df.columns)}")
    return df, repeat_col, fold_col


def load_config_bundle():
    config = load_json(CONFIG_FILE)
    loss_config = load_json(LOSS_CONFIG_FILE)
    checkpoint_map = load_json(ENCODER_MAP_FILE)

    if NUM_WORKERS_OVERRIDE is not None:
        config["data"]["num_workers"] = int(NUM_WORKERS_OVERRIDE)

    return config, loss_config, checkpoint_map


def build_loaders(config, split_df, include_test=True, shuffle_train=True):
    pm = pd.read_csv(config["paths"]["participant_manifest"])
    rm = pd.read_csv(config["paths"]["representation_manifest"])
    split_df = normalize_split_df(split_df)

    batch_size = int(config["data"]["batch_size"])
    num_workers = int(config["data"].get("num_workers", 0))

    loaders = {}
    for split_name in ["train", "val"] + (["test"] if include_test else []):
        if split_name not in set(split_df["split"].unique()):
            continue
        ds = s10x.ParticipantDataset(pm, rm, split_df, split_name, config)
        loaders[split_name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(shuffle_train and split_name == "train"),
            num_workers=num_workers,
            collate_fn=s10x.collate_fn
        )
    return loaders


def instantiate_model(config, checkpoint_map=None, load_contrastive=True):
    model = s10x.ContrastiveFineTunedCPMRNet(config).to(DEVICE)
    if load_contrastive and checkpoint_map is not None:
        model.load_contrastive_encoders(checkpoint_map, DEVICE)
    return model


def load_model_checkpoint(model, checkpoint_path: Path):
    state = torch.load(checkpoint_path, map_location=DEVICE)
    # Support plain state_dict or wrapped dict.
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def predictions_dataframe(eval_out, split_name, repeat=None, fold=None):
    df = pd.DataFrame({
        "participant_id": [str(x) for x in eval_out["ids"]],
        "y_true": np.asarray(eval_out["labels"]).astype(int),
        "y_score": np.asarray(eval_out["probs"]).astype(float),
        "split": split_name,
    })
    if repeat is not None:
        df["repeat"] = repeat
    if fold is not None:
        df["fold"] = fold
    df["y_pred_0_5"] = (df["y_score"] >= 0.5).astype(int)
    return df


def safe_compute_metrics(y_true, y_prob, threshold=0.5):
    try:
        return s10x.compute_metrics(y_true, y_prob, threshold=threshold)
    except Exception as exc:
        return {"error": repr(exc)}


def youden_threshold(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j = tpr - fpr
    idx = int(np.nanargmax(j))
    thr = thresholds[idx]
    if not np.isfinite(thr):
        return 0.5
    return float(thr)


def flatten_metric_row(metrics, prefix=""):
    row = {}
    for k, v in metrics.items():
        key = f"{prefix}{k}" if prefix else k
        if isinstance(v, (int, float, np.integer, np.floating)):
            row[key] = float(v) if not isinstance(v, (int, np.integer)) else int(v)
        else:
            row[key] = v
    return row


def train_stage10x_fold(config, loss_config, checkpoint_map, split_df, repeat, fold):
    """Train Stage10X for one repeated-CV fold using the same protocol as Stage10X main()."""
    seed = int(config["optimization"].get("random_seed", 42)) + int(repeat) * 100 + int(fold)
    s10x.set_seed(seed)

    loaders = build_loaders(config, split_df, include_test=True, shuffle_train=True)
    if "train" not in loaders or "val" not in loaders:
        raise RuntimeError(f"Fold repeat={repeat}, fold={fold} lacks train or val split.")

    model = instantiate_model(config, checkpoint_map, load_contrastive=True)
    model.set_encoder_trainable(False)
    encoders_trainable = False

    base_lr = float(config["optimization"]["learning_rate"])
    weight_decay = float(config["optimization"]["weight_decay"])
    optimizer = s10x.make_optimizer(model, base_lr, weight_decay, encoders_trainable)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(config["optimization"]["scheduler_factor"]),
        patience=int(config["optimization"]["scheduler_patience"])
    )

    pos_weight = torch.tensor(
        [float(loss_config["positive_class_weight_for_bce"])],
        dtype=torch.float32,
        device=DEVICE
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    max_epochs = int(config["optimization"]["max_epochs"])
    patience = int(config["optimization"]["early_stopping_patience"])

    best_auc = -np.inf
    best_epoch = 0
    patience_counter = 0
    best_state = None
    best_val_out = None
    history = []

    for epoch in range(1, max_epochs + 1):
        if epoch == int(getattr(s10x, "FREEZE_ENCODERS_FIRST_EPOCHS", 5)) + 1:
            model.set_encoder_trainable(True)
            encoders_trainable = True
            optimizer = s10x.make_optimizer(model, base_lr, weight_decay, encoders_trainable)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=float(config["optimization"]["scheduler_factor"]),
                patience=int(config["optimization"]["scheduler_patience"])
            )

        train_out = s10x.run_epoch(model, loaders["train"], optimizer, criterion, DEVICE, train=True)
        val_out = s10x.run_epoch(model, loaders["val"], optimizer, criterion, DEVICE, train=False)
        val_auc = val_out["metrics"].get("roc_auc", np.nan)
        scheduler.step(0.0 if np.isnan(val_auc) else val_auc)

        row = {
            "repeat": repeat,
            "fold": fold,
            "epoch": epoch,
            "encoders_trainable": encoders_trainable,
            "train_loss": train_out["loss"],
            "val_loss": val_out["loss"],
        }
        row.update(flatten_metric_row(train_out["metrics"], "train_"))
        row.update(flatten_metric_row(val_out["metrics"], "val_"))
        history.append(row)

        improved = not np.isnan(val_auc) and val_auc > best_auc
        if improved:
            best_auc = float(val_auc)
            best_epoch = epoch
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
            best_val_out = copy.deepcopy(val_out)
        else:
            patience_counter += 1

        print(
            f"Stage10XD CV repeat={repeat} fold={fold} epoch={epoch:03d} | "
            f"train_loss={train_out['loss']:.4f} | val_loss={val_out['loss']:.4f} | "
            f"val_auc={val_auc:.4f} | patience={patience_counter}/{patience}"
        )

        if patience_counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, MODELS_DIR / f"stage10xd_repeat{repeat}_fold{fold}_best.pt")

    # Determine threshold from fold validation predictions only.
    val_threshold = 0.5
    if best_val_out is not None:
        val_threshold = youden_threshold(best_val_out["labels"], best_val_out["probs"])

    test_out = None
    test_metrics_0_5 = {}
    test_metrics_youden = {}
    test_pred_df = pd.DataFrame()
    if "test" in loaders:
        dummy_optimizer = None
        criterion_eval = criterion
        # run_epoch expects optimizer but does not use it when train=False.
        test_out = s10x.run_epoch(model, loaders["test"], dummy_optimizer, criterion_eval, DEVICE, train=False)
        test_metrics_0_5 = safe_compute_metrics(test_out["labels"], test_out["probs"], threshold=0.5)
        test_metrics_youden = safe_compute_metrics(test_out["labels"], test_out["probs"], threshold=val_threshold)
        test_pred_df = predictions_dataframe(test_out, "test", repeat, fold)
        test_pred_df["threshold_youden_from_val"] = val_threshold
        test_pred_df["y_pred_youden"] = (test_pred_df["y_score"] >= val_threshold).astype(int)

    fold_summary = {
        "repeat": repeat,
        "fold": fold,
        "best_epoch": best_epoch,
        "best_val_roc_auc": best_auc,
        "validation_youden_threshold": val_threshold,
        "test_roc_auc": test_metrics_0_5.get("roc_auc", np.nan),
        "test_pr_auc": test_metrics_0_5.get("pr_auc", np.nan),
        "test_accuracy_0_5": test_metrics_0_5.get("accuracy", np.nan),
        "test_balanced_accuracy_0_5": test_metrics_0_5.get("balanced_accuracy", np.nan),
        "test_recall_0_5": test_metrics_0_5.get("recall", np.nan),
        "test_f1_0_5": test_metrics_0_5.get("f1", np.nan),
        "test_accuracy_youden": test_metrics_youden.get("accuracy", np.nan),
        "test_balanced_accuracy_youden": test_metrics_youden.get("balanced_accuracy", np.nan),
        "test_recall_youden": test_metrics_youden.get("recall", np.nan),
        "test_f1_youden": test_metrics_youden.get("f1", np.nan),
    }

    return pd.DataFrame(history), test_pred_df, fold_summary


# =============================================================================
# HOLDOUT EVALUATION
# =============================================================================

def evaluate_holdout(config, loss_config, checkpoint_map):
    split_df = pd.read_csv(HOLDOUT_SPLIT_FILE)
    split_df = normalize_split_df(split_df)

    loaders = build_loaders(config, split_df, include_test=True, shuffle_train=False)
    if "test" not in loaders:
        raise RuntimeError("Holdout split file does not contain a test split.")

    ckpt = STAGE10X_BEST_CKPT if STAGE10X_BEST_CKPT.exists() else STAGE10X_LAST_CKPT
    if not ckpt.exists():
        raise FileNotFoundError(f"No Stage10X checkpoint found at {STAGE10X_BEST_CKPT} or {STAGE10X_LAST_CKPT}")

    model = instantiate_model(config, checkpoint_map, load_contrastive=False)
    model = load_model_checkpoint(model, ckpt)

    pos_weight = torch.tensor(
        [float(loss_config["positive_class_weight_for_bce"])],
        dtype=torch.float32,
        device=DEVICE
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Evaluate validation to derive threshold, if available.
    val_threshold = 0.5
    val_metrics = {}
    val_pred_df = pd.DataFrame()
    if "val" in loaders:
        val_out = s10x.run_epoch(model, loaders["val"], None, criterion, DEVICE, train=False)
        val_threshold = youden_threshold(val_out["labels"], val_out["probs"])
        val_metrics = safe_compute_metrics(val_out["labels"], val_out["probs"], threshold=0.5)
        val_pred_df = predictions_dataframe(val_out, "val")
        val_pred_df["threshold_youden_from_val"] = val_threshold
        val_pred_df["y_pred_youden"] = (val_pred_df["y_score"] >= val_threshold).astype(int)

    test_out = s10x.run_epoch(model, loaders["test"], None, criterion, DEVICE, train=False)
    test_metrics_0_5 = safe_compute_metrics(test_out["labels"], test_out["probs"], threshold=0.5)
    test_metrics_youden = safe_compute_metrics(test_out["labels"], test_out["probs"], threshold=val_threshold)

    test_pred_df = predictions_dataframe(test_out, "test")
    test_pred_df["threshold_youden_from_val"] = val_threshold
    test_pred_df["y_pred_youden"] = (test_pred_df["y_score"] >= val_threshold).astype(int)

    holdout_pred_all = pd.concat([val_pred_df, test_pred_df], ignore_index=True)
    holdout_pred_path = TABLES_DIR / "stage10xd_holdout_predictions.csv"
    holdout_pred_all.to_csv(holdout_pred_path, index=False)

    metric_row = {
        "evidence": "Stage10X independent holdout test",
        "checkpoint": str(ckpt),
        "validation_youden_threshold": val_threshold,
        "n_test": int(len(test_pred_df)),
    }
    metric_row.update(flatten_metric_row(test_metrics_0_5, "test_0_5_"))
    metric_row.update(flatten_metric_row(test_metrics_youden, "test_youden_"))
    metric_row.update(flatten_metric_row(val_metrics, "val_0_5_"))

    metrics_path = TABLES_DIR / "stage10xd_holdout_metrics.csv"
    pd.DataFrame([metric_row]).to_csv(metrics_path, index=False)

    return metric_row, holdout_pred_path, metrics_path


# =============================================================================
# REPEATED-CV EVALUATION
# =============================================================================

def evaluate_repeated_cv(config, loss_config, checkpoint_map):
    if not REPEATED_SPLIT_FILE.exists():
        raise FileNotFoundError(f"Repeated-CV split file not found: {REPEATED_SPLIT_FILE}")

    split_all = pd.read_csv(REPEATED_SPLIT_FILE)
    split_all = normalize_split_df(split_all)
    split_all, repeat_col, fold_col = find_repeat_fold_columns(split_all)

    fold_keys = (
        split_all[[repeat_col, fold_col]]
        .drop_duplicates()
        .sort_values([repeat_col, fold_col])
        .values.tolist()
    )
    if MAX_CV_FOLDS > 0:
        fold_keys = fold_keys[:MAX_CV_FOLDS]

    all_histories = []
    all_predictions = []
    fold_summaries = []

    for repeat, fold in fold_keys:
        fold_df = split_all[(split_all[repeat_col] == repeat) & (split_all[fold_col] == fold)].copy()
        # Normalize to expected columns for ParticipantDataset.
        if repeat_col != "repeat":
            fold_df["repeat"] = repeat
        if fold_col != "fold":
            fold_df["fold"] = fold

        history_df, pred_df, fold_summary = train_stage10x_fold(
            config=config,
            loss_config=loss_config,
            checkpoint_map=checkpoint_map,
            split_df=fold_df,
            repeat=repeat,
            fold=fold,
        )
        all_histories.append(history_df)
        if len(pred_df) > 0:
            all_predictions.append(pred_df)
        fold_summaries.append(fold_summary)

    histories_df = pd.concat(all_histories, ignore_index=True) if all_histories else pd.DataFrame()
    predictions_df = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    fold_metrics_df = pd.DataFrame(fold_summaries)

    histories_path = TABLES_DIR / "stage10xd_repeated_cv_training_history.csv"
    preds_path = TABLES_DIR / "stage10xd_repeated_cv_predictions.csv"
    metrics_path = TABLES_DIR / "stage10xd_repeated_cv_fold_metrics.csv"
    summary_path = TABLES_DIR / "stage10xd_repeated_cv_summary_metrics.csv"

    histories_df.to_csv(histories_path, index=False)
    predictions_df.to_csv(preds_path, index=False)
    fold_metrics_df.to_csv(metrics_path, index=False)

    summary = {}
    if len(fold_metrics_df) > 0:
        metric_cols = [c for c in fold_metrics_df.columns if c.startswith("test_") or c == "best_val_roc_auc"]
        rows = []
        for col in metric_cols:
            vals = pd.to_numeric(fold_metrics_df[col], errors="coerce").dropna()
            rows.append({
                "metric": col,
                "mean": vals.mean() if len(vals) else np.nan,
                "std": vals.std(ddof=1) if len(vals) > 1 else np.nan,
                "min": vals.min() if len(vals) else np.nan,
                "max": vals.max() if len(vals) else np.nan,
                "n_folds": int(len(vals)),
            })
        summary_df = pd.DataFrame(rows)
        summary_df.to_csv(summary_path, index=False)
        for _, r in summary_df.iterrows():
            summary[str(r["metric"])] = {
                "mean": r["mean"], "std": r["std"], "n_folds": int(r["n_folds"])
            }
    else:
        pd.DataFrame().to_csv(summary_path, index=False)

    return {
        "histories_path": histories_path,
        "predictions_path": preds_path,
        "fold_metrics_path": metrics_path,
        "summary_metrics_path": summary_path,
        "fold_metrics": fold_metrics_df,
        "summary": summary,
    }


# =============================================================================
# REPORTING
# =============================================================================

def make_figures(final_rows):
    df = pd.DataFrame(final_rows)
    if len(df) == 0:
        return []
    fig_paths = []

    auc_df = df[df["roc_auc"].notna()].copy()
    if len(auc_df):
        auc_df = auc_df.sort_values("roc_auc", ascending=True)
        plt.figure(figsize=(11, 5))
        plt.barh(auc_df["evidence"], auc_df["roc_auc"])
        plt.axvline(BENCHMARK_STAGE6D6B_AUC, linestyle="--", label="Stage6D6B benchmark")
        plt.xlabel("ROC-AUC")
        plt.title("Stage10XD conservative ROC-AUC evidence")
        plt.legend()
        plt.tight_layout()
        p = FIGURES_DIR / "stage10xd_conservative_auc_evidence.png"
        plt.savefig(p, dpi=300)
        plt.close()
        fig_paths.append(p)

    return fig_paths


def write_report(summary, final_metric_df):
    report = []
    report.append("# Stage10XD — Independent Holdout and Repeated-CV Evaluator\n")
    report.append(f"Generated: {summary['generated']}\n")
    report.append("## Purpose\n")
    report.append(
        "This stage evaluates the completed Stage10X Progressive Contrastive Fine-Tuning CPMR-Net without changing the architecture, representation set, loss design, or hyperparameters.\n"
    )
    report.append("## Fixed Rules\n")
    for rule in summary["fixed_rules"]:
        report.append(f"- {rule}")
    report.append("\n## Execution Summary\n")
    report.append(f"- Device: {summary['device']}")
    report.append(f"- Holdout evaluation completed: {summary['holdout_evaluation_completed']}")
    report.append(f"- Repeated-CV requested: {summary['repeated_cv_requested']}")
    report.append(f"- Repeated-CV completed: {summary['repeated_cv_completed']}")
    report.append(f"- Stage6D6B benchmark ROC-AUC: {BENCHMARK_STAGE6D6B_AUC:.4f}")
    report.append("\n## Final Metric Summary\n")
    if len(final_metric_df):
        report.append(final_metric_df.to_markdown(index=False))
    else:
        report.append("No metrics were generated.")
    report.append("\n## Interpretation\n")
    report.append(summary["interpretation"])
    report.append("\n## Generated Outputs\n")
    for key, val in summary.get("generated_outputs", {}).items():
        report.append(f"- {key}: `{val}`")

    report_path = REPORTS_DIR / "Stage10XD_Independent_Holdout_and_RepeatedCV_Evaluator_Report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    return report_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    config, loss_config, checkpoint_map = load_config_bundle()
    config_snapshot = copy.deepcopy(config)
    config_snapshot["stage10xd"] = {
        "run_repeated_cv": RUN_REPEATED_CV,
        "max_cv_folds": MAX_CV_FOLDS,
        "device": str(DEVICE),
        "stage10x_script": str(STAGE10X_SCRIPT),
        "stage10x_best_checkpoint": str(STAGE10X_BEST_CKPT),
        "holdout_split_file": str(HOLDOUT_SPLIT_FILE),
        "repeated_split_file": str(REPEATED_SPLIT_FILE),
    }
    save_json(config_snapshot, CONFIGS_DIR / "stage10xd_execution_config.json")

    final_rows = []
    generated_outputs = {}
    holdout_done = False
    cv_done = False
    errors = []

    # Holdout evaluation.
    try:
        holdout_metrics, holdout_pred_path, holdout_metrics_path = evaluate_holdout(config, loss_config, checkpoint_map)
        holdout_done = True
        generated_outputs["holdout_predictions"] = str(holdout_pred_path)
        generated_outputs["holdout_metrics"] = str(holdout_metrics_path)
        final_rows.append({
            "evidence": "Stage10X independent holdout test",
            "n": holdout_metrics.get("n_test", np.nan),
            "roc_auc": holdout_metrics.get("test_0_5_roc_auc", np.nan),
            "pr_auc": holdout_metrics.get("test_0_5_pr_auc", np.nan),
            "accuracy_0_5": holdout_metrics.get("test_0_5_accuracy", np.nan),
            "balanced_accuracy_0_5": holdout_metrics.get("test_0_5_balanced_accuracy", np.nan),
            "recall_0_5": holdout_metrics.get("test_0_5_recall", np.nan),
            "f1_0_5": holdout_metrics.get("test_0_5_f1", np.nan),
            "accuracy_youden": holdout_metrics.get("test_youden_accuracy", np.nan),
            "balanced_accuracy_youden": holdout_metrics.get("test_youden_balanced_accuracy", np.nan),
            "recall_youden": holdout_metrics.get("test_youden_recall", np.nan),
            "f1_youden": holdout_metrics.get("test_youden_f1", np.nan),
            "margin_vs_stage6d6b_auc": holdout_metrics.get("test_0_5_roc_auc", np.nan) - BENCHMARK_STAGE6D6B_AUC,
        })
    except Exception as exc:
        errors.append({"stage": "holdout", "error": repr(exc)})

    # Repeated CV evaluation.
    cv_result = None
    if RUN_REPEATED_CV:
        try:
            cv_result = evaluate_repeated_cv(config, loss_config, checkpoint_map)
            cv_done = True
            generated_outputs["repeated_cv_predictions"] = str(cv_result["predictions_path"])
            generated_outputs["repeated_cv_fold_metrics"] = str(cv_result["fold_metrics_path"])
            generated_outputs["repeated_cv_summary_metrics"] = str(cv_result["summary_metrics_path"])
            generated_outputs["repeated_cv_training_history"] = str(cv_result["histories_path"])

            fold_df = cv_result["fold_metrics"]
            if len(fold_df):
                vals = pd.to_numeric(fold_df["test_roc_auc"], errors="coerce").dropna()
                pr_vals = pd.to_numeric(fold_df["test_pr_auc"], errors="coerce").dropna()
                final_rows.append({
                    "evidence": "Stage10X repeated-CV test mean",
                    "n": int(len(vals)),
                    "roc_auc": float(vals.mean()) if len(vals) else np.nan,
                    "roc_auc_std": float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
                    "pr_auc": float(pr_vals.mean()) if len(pr_vals) else np.nan,
                    "accuracy_0_5": float(pd.to_numeric(fold_df["test_accuracy_0_5"], errors="coerce").mean()),
                    "balanced_accuracy_0_5": float(pd.to_numeric(fold_df["test_balanced_accuracy_0_5"], errors="coerce").mean()),
                    "recall_0_5": float(pd.to_numeric(fold_df["test_recall_0_5"], errors="coerce").mean()),
                    "f1_0_5": float(pd.to_numeric(fold_df["test_f1_0_5"], errors="coerce").mean()),
                    "accuracy_youden": float(pd.to_numeric(fold_df["test_accuracy_youden"], errors="coerce").mean()),
                    "balanced_accuracy_youden": float(pd.to_numeric(fold_df["test_balanced_accuracy_youden"], errors="coerce").mean()),
                    "recall_youden": float(pd.to_numeric(fold_df["test_recall_youden"], errors="coerce").mean()),
                    "f1_youden": float(pd.to_numeric(fold_df["test_f1_youden"], errors="coerce").mean()),
                    "margin_vs_stage6d6b_auc": (float(vals.mean()) - BENCHMARK_STAGE6D6B_AUC) if len(vals) else np.nan,
                })
        except Exception as exc:
            errors.append({"stage": "repeated_cv", "error": repr(exc)})

    # Add benchmark row.
    final_rows.append({
        "evidence": "Stage6D6B benchmark",
        "n": np.nan,
        "roc_auc": BENCHMARK_STAGE6D6B_AUC,
        "roc_auc_std": np.nan,
        "pr_auc": np.nan,
        "accuracy_0_5": 0.7374,
        "balanced_accuracy_0_5": np.nan,
        "recall_0_5": 0.4154,
        "f1_0_5": 0.5094,
        "accuracy_youden": np.nan,
        "balanced_accuracy_youden": np.nan,
        "recall_youden": np.nan,
        "f1_youden": np.nan,
        "margin_vs_stage6d6b_auc": 0.0,
    })

    final_metric_df = pd.DataFrame(final_rows)
    final_metric_path = TABLES_DIR / "stage10xd_final_metric_summary.csv"
    final_metric_df.to_csv(final_metric_path, index=False)
    generated_outputs["final_metric_summary"] = str(final_metric_path)

    fig_paths = make_figures(final_rows)
    for i, p in enumerate(fig_paths, start=1):
        generated_outputs[f"figure_{i}"] = str(p)

    # Conservative interpretation.
    holdout_auc = np.nan
    repeated_auc = np.nan
    if holdout_done:
        try:
            holdout_auc = float(final_metric_df.loc[final_metric_df["evidence"] == "Stage10X independent holdout test", "roc_auc"].iloc[0])
        except Exception:
            pass
    if cv_done:
        try:
            repeated_auc = float(final_metric_df.loc[final_metric_df["evidence"] == "Stage10X repeated-CV test mean", "roc_auc"].iloc[0])
        except Exception:
            pass

    if cv_done and np.isfinite(repeated_auc):
        conservative_auc = repeated_auc
        evidence_used = "repeated_cv"
    elif holdout_done and np.isfinite(holdout_auc):
        conservative_auc = holdout_auc
        evidence_used = "holdout"
    else:
        conservative_auc = np.nan
        evidence_used = "none"

    margin = conservative_auc - BENCHMARK_STAGE6D6B_AUC if np.isfinite(conservative_auc) else np.nan
    if np.isfinite(margin) and margin > 0:
        claim_status = "stage10x_conservatively_surpasses_stage6d6b"
        interpretation = (
            f"Stage10X has conservative {evidence_used} ROC-AUC evidence above Stage6D6B by {margin:.4f}. "
            "This supports promotion of Stage10X as the final CPMR-Net candidate, subject to reporting the full uncertainty and split protocol."
        )
    elif np.isfinite(margin):
        claim_status = "stage10x_does_not_surpass_stage6d6b_conservatively"
        interpretation = (
            f"Stage10X conservative {evidence_used} ROC-AUC does not exceed Stage6D6B. "
            "Stage6D6B remains the strongest validated model, while Stage10X remains a scientifically useful CPMR-Net architecture."
        )
    else:
        claim_status = "evaluation_incomplete"
        interpretation = (
            "No conservative Stage10X holdout or repeated-CV ROC-AUC was generated. "
            "Review errors and ensure all Stage10I1 splits, manifests, checkpoints, and contrastive encoder checkpoints are available."
        )

    summary = {
        "stage": "Stage10XD",
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(DEVICE),
        "stage10x_script": str(STAGE10X_SCRIPT),
        "stage10x_best_checkpoint": str(STAGE10X_BEST_CKPT),
        "holdout_split_file": str(HOLDOUT_SPLIT_FILE),
        "repeated_split_file": str(REPEATED_SPLIT_FILE),
        "benchmark_stage6d6b_auc": BENCHMARK_STAGE6D6B_AUC,
        "expected_stage10x_validation_auc": EXPECTED_STAGE10X_VAL_AUC,
        "holdout_evaluation_completed": holdout_done,
        "repeated_cv_requested": RUN_REPEATED_CV,
        "repeated_cv_completed": cv_done,
        "max_cv_folds": MAX_CV_FOLDS,
        "evidence_used_for_conservative_decision": evidence_used,
        "stage10x_conservative_auc": conservative_auc,
        "margin_vs_stage6d6b": margin,
        "claim_status": claim_status,
        "fixed_rules": [
            "participant-level diagnosis only",
            "fixed Stage10I1 holdout and repeated-CV splits",
            "no image-level diagnosis",
            "no architecture change",
            "no hyperparameter search",
            "same Stage10X contrastive encoder initialization and progressive fine-tuning protocol",
        ],
        "errors": errors,
        "interpretation": interpretation,
        "generated_outputs": generated_outputs,
        "output_dir": str(OUTPUT_DIR),
    }
    save_json(summary, OUTPUT_DIR / "Stage10XD_Independent_Holdout_and_RepeatedCV_Evaluator_Summary.json")

    report_path = write_report(summary, final_metric_df)
    generated_outputs["report"] = str(report_path)
    # Re-save summary with report path.
    summary["generated_outputs"] = generated_outputs
    save_json(summary, OUTPUT_DIR / "Stage10XD_Independent_Holdout_and_RepeatedCV_Evaluator_Summary.json")

    if errors:
        err_path = REPORTS_DIR / "stage10xd_errors.json"
        save_json(errors, err_path)
        generated_outputs["errors"] = str(err_path)

    print("=" * 80)
    print("STAGE10XD INDEPENDENT HOLDOUT AND REPEATED-CV EVALUATION COMPLETED")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Holdout evaluation completed: {holdout_done}")
    print(f"Repeated-CV requested/completed: {RUN_REPEATED_CV}/{cv_done}")
    print(f"Evidence used for decision: {evidence_used}")
    print(f"Stage10X conservative AUC: {conservative_auc if np.isfinite(conservative_auc) else 'NA'}")
    print(f"Margin vs Stage6D6B: {margin if np.isfinite(margin) else 'NA'}")
    print(f"Claim status: {claim_status}")
    if errors:
        print("Errors occurred. See reports/stage10xd_errors.json")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
