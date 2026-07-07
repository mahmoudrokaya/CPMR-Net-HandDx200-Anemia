"""
Stage10XC — Local Stage10X Evaluation Adapter
Project: CPMR-Net HandDx-200 anemia diagnosis

Save this file as:
D:\\47\\472\\New-Papers\\Anemia_Paper\\Codes\\Stage10X_Local_Evaluation_Adapter.py

Purpose
-------
This adapter connects Stage10XB to the local Stage10X implementation without changing
CPMR-Net architecture, representation set, loss design, split policy, or hyperparameters.

It exposes the required function:
    evaluate_stage10x_holdout_and_cv(config: dict) -> dict

The adapter is intentionally conservative:
- It first tries to call a high-level evaluation API if your Stage10X script already exposes one.
- It then tries a conventional PyTorch path: build_model/create_model + build_dataloaders.
- If no compatible local API is found, it stops with a clear diagnostic rather than inventing results.

Expected outputs, when execution succeeds:
- participant-level holdout predictions
- participant-level repeated-CV predictions
- metrics tables
- a JSON adapter summary

Required prediction columns:
    participant_id, y_true, y_score, split, repeat, fold
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
import importlib.util
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

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
)


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        if len(np.unique(y_true.astype(int))) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def _safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        if len(np.unique(y_true.astype(int))) < 2:
            return float("nan")
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return float("nan")


def _youden_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        fpr, tpr, thresholds = roc_curve(y_true.astype(int), y_score.astype(float))
        j = tpr - fpr
        idx = int(np.nanargmax(j))
        thr = float(thresholds[idx])
        if not np.isfinite(thr):
            return 0.5
        return thr
    except Exception:
        return 0.5


def _metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "positive_n": int(y_true.sum()),
        "negative_n": int((1 - y_true).sum()),
        "threshold": float(threshold),
        "roc_auc": _safe_auc(y_true, y_score),
        "pr_auc": _safe_pr_auc(y_true, y_score),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def _summarize_predictions(preds: pd.DataFrame, evidence: str) -> pd.DataFrame:
    if preds is None or preds.empty:
        return pd.DataFrame()
    required = {"y_true", "y_score"}
    if not required.issubset(set(preds.columns)):
        raise ValueError(f"Prediction table must include {required}; found {list(preds.columns)}")

    rows: List[Dict[str, Any]] = []
    df = preds.copy()
    df["y_true"] = df["y_true"].astype(int)
    df["y_score"] = df["y_score"].astype(float)

    grouping_cols = []
    if "split" in df.columns:
        grouping_cols.append("split")
    if evidence == "repeated_cv":
        for c in ["repeat", "fold"]:
            if c in df.columns:
                grouping_cols.append(c)

    if grouping_cols:
        for key, g in df.groupby(grouping_cols, dropna=False):
            key = key if isinstance(key, tuple) else (key,)
            base = {"evidence": evidence}
            for col, val in zip(grouping_cols, key):
                base[col] = val
            thr = _youden_threshold(g["y_true"].values, g["y_score"].values)
            base.update(_metrics(g["y_true"].values, g["y_score"].values, threshold=0.5))
            base["youden_threshold"] = thr
            ym = _metrics(g["y_true"].values, g["y_score"].values, threshold=thr)
            for k, v in ym.items():
                base[f"youden_{k}"] = v
            rows.append(base)

    base = {"evidence": evidence, "split": "overall"}
    thr = _youden_threshold(df["y_true"].values, df["y_score"].values)
    base.update(_metrics(df["y_true"].values, df["y_score"].values, threshold=0.5))
    base["youden_threshold"] = thr
    ym = _metrics(df["y_true"].values, df["y_score"].values, threshold=thr)
    for k, v in ym.items():
        base[f"youden_{k}"] = v
    rows.append(base)
    return pd.DataFrame(rows)


def _import_module_from_path(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def _candidate_stage10x_scripts(config: Dict[str, Any]) -> List[Path]:
    paths = []
    for p in config.get("discovered_stage10x_scripts", []) or []:
        path = Path(p)
        if path.exists() and path.name != Path(__file__).name:
            paths.append(path)
    codes_dir = Path(config.get("codes_dir", r"D:\47\472\New-Papers\Anemia_Paper\Codes"))
    preferred = codes_dir / "Stage10X_Progressive_FineTuning_Contrastive_CPMRNet.py"
    if preferred.exists():
        paths.insert(0, preferred)
    # Remove audit/adapter scripts from model candidates.
    filtered = []
    for p in paths:
        low = p.name.lower()
        if any(x in low for x in ["conservative", "verification", "stage10xb", "stage10xc", "adapter", "stage11"]):
            continue
        filtered.append(p)
    return list(dict.fromkeys(filtered))


def _load_checkpoint_into_model(model: Any, checkpoint_path: Path, device: str = "cpu") -> Any:
    try:
        import torch
    except Exception as e:
        raise RuntimeError("PyTorch is required to load the Stage10X checkpoint.") from e

    ckpt = torch.load(str(checkpoint_path), map_location=device)
    if isinstance(ckpt, dict):
        for key in ["model_state_dict", "state_dict", "model", "net"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {checkpoint_path}")

    # Remove DataParallel prefix if present.
    clean = {}
    for k, v in ckpt.items():
        nk = k[7:] if str(k).startswith("module.") else k
        clean[nk] = v
    missing, unexpected = model.load_state_dict(clean, strict=False)
    if len(unexpected) > 0:
        print(f"[Stage10XC] Warning: unexpected checkpoint keys: {list(unexpected)[:10]}")
    if len(missing) > 0:
        print(f"[Stage10XC] Warning: missing checkpoint keys: {list(missing)[:10]}")
    model.to(device)
    model.eval()
    return model


def _extract_batch(batch: Any) -> Tuple[Any, np.ndarray, List[Any]]:
    """Extract model inputs, labels, participant IDs from common batch formats."""
    try:
        import torch
    except Exception:
        torch = None

    if isinstance(batch, dict):
        y = None
        for k in ["label", "labels", "y", "y_true", "target", "targets", "anemia_label"]:
            if k in batch:
                y = batch[k]
                break
        pid = None
        for k in ["participant_id", "participant_ids", "pid", "subject_id"]:
            if k in batch:
                pid = batch[k]
                break
        exclude = {"label", "labels", "y", "y_true", "target", "targets", "anemia_label", "participant_id", "participant_ids", "pid", "subject_id"}
        x = {k: v for k, v in batch.items() if k not in exclude}
        if len(x) == 1:
            x = list(x.values())[0]
        if y is None:
            raise ValueError("Could not locate labels in batch dictionary.")
        y_np = y.detach().cpu().numpy() if torch is not None and hasattr(y, "detach") else np.asarray(y)
        if pid is None:
            pid_list = list(range(len(y_np)))
        elif torch is not None and hasattr(pid, "detach"):
            pid_list = pid.detach().cpu().numpy().tolist()
        else:
            pid_list = list(pid) if not isinstance(pid, (str, int)) else [pid]
        return x, y_np.astype(int), pid_list

    if isinstance(batch, (list, tuple)):
        if len(batch) < 2:
            raise ValueError("Tuple/list batch must contain at least inputs and labels.")
        x = batch[0]
        y = batch[1]
        pid = batch[2] if len(batch) > 2 else None
        y_np = y.detach().cpu().numpy() if torch is not None and hasattr(y, "detach") else np.asarray(y)
        if pid is None:
            pid_list = list(range(len(y_np)))
        elif torch is not None and hasattr(pid, "detach"):
            pid_list = pid.detach().cpu().numpy().tolist()
        else:
            pid_list = list(pid) if not isinstance(pid, (str, int)) else [pid]
        return x, y_np.astype(int), pid_list

    raise ValueError(f"Unsupported batch type: {type(batch)}")


def _move_to_device(obj: Any, device: str) -> Any:
    try:
        import torch
    except Exception:
        return obj
    if hasattr(obj, "to"):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_move_to_device(v, device) for v in obj)
    return obj


def _sigmoid_if_needed(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr).astype(float).reshape(-1)
    if arr.size == 0:
        return arr
    if np.nanmin(arr) < 0 or np.nanmax(arr) > 1:
        arr = 1.0 / (1.0 + np.exp(-arr))
    return arr


def _predict_loader(model: Any, loader: Iterable, split: str, repeat: Any = None, fold: Any = None, device: str = "cpu") -> pd.DataFrame:
    try:
        import torch
    except Exception as e:
        raise RuntimeError("PyTorch is required for dataloader evaluation.") from e

    rows = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x, y_np, pids = _extract_batch(batch)
            x = _move_to_device(x, device)
            try:
                out = model(x)
            except TypeError:
                if isinstance(x, dict):
                    out = model(**x)
                elif isinstance(x, (list, tuple)):
                    out = model(*x)
                else:
                    raise
            if isinstance(out, dict):
                for k in ["logits", "output", "outputs", "y_score", "score", "prob"]:
                    if k in out:
                        out = out[k]
                        break
            if isinstance(out, (list, tuple)):
                out = out[0]
            score = out.detach().cpu().numpy()
            if score.ndim == 2 and score.shape[1] > 1:
                # Binary 2-logit output.
                e = np.exp(score - np.max(score, axis=1, keepdims=True))
                score = e[:, 1] / e.sum(axis=1)
            else:
                score = _sigmoid_if_needed(score)
            for pid, yt, ys in zip(pids, y_np.reshape(-1), score.reshape(-1)):
                rows.append({
                    "participant_id": pid,
                    "y_true": int(yt),
                    "y_score": float(ys),
                    "split": split,
                    "repeat": repeat,
                    "fold": fold,
                })
    return pd.DataFrame(rows)


def _call_first_existing(module: Any, names: List[str], *args, **kwargs) -> Any:
    for name in names:
        if hasattr(module, name):
            fn = getattr(module, name)
            try:
                return fn(*args, **kwargs)
            except TypeError:
                try:
                    return fn(*args)
                except TypeError:
                    try:
                        return fn()
                    except TypeError:
                        continue
    raise AttributeError(f"None of these functions are callable with supported signatures: {names}")


def _try_high_level_local_api(module: Any, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Use a local high-level evaluator if the Stage10X script already has one."""
    names = [
        "evaluate_stage10x_holdout_and_cv",
        "evaluate_holdout_and_cv",
        "evaluate_stage10x_on_fixed_splits",
        "run_holdout_and_repeated_cv_evaluation",
        "run_evaluation",
    ]
    for name in names:
        if hasattr(module, name) and getattr(module, name) is not evaluate_stage10x_holdout_and_cv:
            fn = getattr(module, name)
            try:
                result = fn(config)
            except TypeError:
                continue
            if isinstance(result, dict):
                return result
    return None


def _try_model_dataloader_api(module: Any, config: Dict[str, Any], paths: Dict[str, Path]) -> Optional[Dict[str, Any]]:
    """Try conventional PyTorch evaluation via build_model/create_model and dataloader functions."""
    try:
        import torch
    except Exception:
        return None

    model_names = ["build_stage10x_model", "build_model", "create_model", "get_model"]
    class_names = ["ProgressiveCPMRNet", "Stage10XModel", "CPMRNet", "CooperativePhysiologicalMultiRepresentationNetwork"]

    model = None
    for name in model_names:
        if hasattr(module, name):
            fn = getattr(module, name)
            for args in [(config,), tuple()]:
                try:
                    model = fn(*args)
                    break
                except TypeError:
                    continue
            if model is not None:
                break
    if model is None:
        for name in class_names:
            if hasattr(module, name):
                cls = getattr(module, name)
                for args in [(config,), tuple()]:
                    try:
                        model = cls(*args)
                        break
                    except TypeError:
                        continue
                if model is not None:
                    break
    if model is None:
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = paths.get("checkpoint")
    if checkpoint is None or not checkpoint.exists():
        raise FileNotFoundError("Stage10X checkpoint not found in config['primary_checkpoint'].")
    model = _load_checkpoint_into_model(model, checkpoint, device=device)

    dataloader_names = [
        "build_stage10x_dataloaders",
        "build_dataloaders",
        "create_dataloaders",
        "get_dataloaders",
        "build_loaders",
    ]
    loaders = None
    for name in dataloader_names:
        if hasattr(module, name):
            fn = getattr(module, name)
            for args in [(config,), tuple()]:
                try:
                    loaders = fn(*args)
                    break
                except TypeError:
                    continue
            if loaders is not None:
                break
    if loaders is None:
        return None

    holdout_preds = pd.DataFrame()
    cv_preds = pd.DataFrame()

    if isinstance(loaders, dict):
        # Holdout/test loader.
        for key in ["test", "holdout", "holdout_test", "val_test"]:
            if key in loaders:
                holdout_preds = _predict_loader(model, loaders[key], split="holdout_test", device=device)
                break
        # CV loaders may be list of fold dicts or dict keyed by fold.
        cv_obj = None
        for key in ["cv", "repeated_cv", "folds", "repeated_folds"]:
            if key in loaders:
                cv_obj = loaders[key]
                break
        if cv_obj is not None:
            fold_frames = []
            if isinstance(cv_obj, dict):
                iterable = cv_obj.items()
            else:
                iterable = enumerate(cv_obj)
            for idx, item in iterable:
                fold_loader = item
                repeat = None
                fold = idx
                if isinstance(item, dict):
                    fold_loader = item.get("test") or item.get("val") or item.get("loader")
                    repeat = item.get("repeat")
                    fold = item.get("fold", idx)
                if fold_loader is not None:
                    fold_frames.append(_predict_loader(model, fold_loader, split="cv_test", repeat=repeat, fold=fold, device=device))
            if fold_frames:
                cv_preds = pd.concat(fold_frames, ignore_index=True)
    else:
        # Common tuple: train_loader, val_loader, test_loader
        try:
            seq = list(loaders)
            if len(seq) >= 3:
                holdout_preds = _predict_loader(model, seq[2], split="holdout_test", device=device)
        except Exception:
            pass

    if holdout_preds.empty and cv_preds.empty:
        return None

    return _save_adapter_outputs(holdout_preds, cv_preds, config, execution_mode="model_dataloader_api")


def _save_adapter_outputs(holdout_preds: pd.DataFrame, cv_preds: pd.DataFrame, config: Dict[str, Any], execution_mode: str) -> Dict[str, Any]:
    output_dir = Path(config.get("output_dir", r"D:\47\472\New-Papers\Anemia_Paper\Outputs\Stage10XB_Holdout_and_RepeatedCV_Evaluation"))
    tables_dir = _mkdir(output_dir / "tables")
    logs_dir = _mkdir(output_dir / "logs")

    holdout_predictions_path = tables_dir / "stage10xb_stage10x_holdout_predictions.csv"
    cv_predictions_path = tables_dir / "stage10xb_stage10x_repeated_cv_predictions.csv"
    holdout_metrics_path = tables_dir / "stage10xb_stage10x_holdout_metrics.csv"
    cv_metrics_path = tables_dir / "stage10xb_stage10x_repeated_cv_metrics.csv"
    summary_path = output_dir / "Stage10XC_Adapter_Execution_Summary.json"

    if holdout_preds is None:
        holdout_preds = pd.DataFrame()
    if cv_preds is None:
        cv_preds = pd.DataFrame()

    holdout_preds.to_csv(holdout_predictions_path, index=False)
    cv_preds.to_csv(cv_predictions_path, index=False)

    holdout_metrics = _summarize_predictions(holdout_preds, "holdout") if not holdout_preds.empty else pd.DataFrame()
    cv_metrics = _summarize_predictions(cv_preds, "repeated_cv") if not cv_preds.empty else pd.DataFrame()
    holdout_metrics.to_csv(holdout_metrics_path, index=False)
    cv_metrics.to_csv(cv_metrics_path, index=False)

    h_auc = float("nan")
    if not holdout_metrics.empty:
        overall = holdout_metrics[holdout_metrics["split"].astype(str).eq("overall")]
        if not overall.empty:
            h_auc = float(overall.iloc[0]["roc_auc"])
    cv_auc = float("nan")
    cv_std = float("nan")
    if not cv_metrics.empty:
        fold_rows = cv_metrics[~cv_metrics["split"].astype(str).eq("overall")]
        if not fold_rows.empty:
            cv_auc = float(fold_rows["roc_auc"].mean())
            cv_std = float(fold_rows["roc_auc"].std())
        else:
            cv_auc = float(cv_metrics.iloc[0]["roc_auc"])

    benchmark_auc = float(config.get("benchmark_stage6d6b_auc", 0.7447))
    conservative_auc = cv_auc if np.isfinite(cv_auc) else h_auc
    margin = conservative_auc - benchmark_auc if np.isfinite(conservative_auc) else float("nan")

    summary = {
        "stage": "Stage10XC Local Adapter Execution",
        "generated": _now(),
        "execution_mode": execution_mode,
        "holdout_evidence_produced": bool(not holdout_preds.empty),
        "repeated_cv_evidence_produced": bool(not cv_preds.empty),
        "holdout_n": int(len(holdout_preds)),
        "cv_prediction_rows": int(len(cv_preds)),
        "stage10x_holdout_auc": h_auc,
        "stage10x_repeated_cv_auc_mean": cv_auc,
        "stage10x_repeated_cv_auc_std": cv_std,
        "benchmark_stage6d6b_auc": benchmark_auc,
        "conservative_auc_for_decision": conservative_auc,
        "margin_vs_stage6d6b": margin,
        "holdout_predictions_path": str(holdout_predictions_path),
        "holdout_metrics_path": str(holdout_metrics_path),
        "cv_predictions_path": str(cv_predictions_path),
        "cv_metrics_path": str(cv_metrics_path),
        "summary_path": str(summary_path),
    }
    _write_json(summary_path, summary)
    (logs_dir / "Stage10XC_adapter_success.log").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    return {
        "holdout_predictions_path": str(holdout_predictions_path),
        "holdout_metrics_path": str(holdout_metrics_path),
        "cv_predictions_path": str(cv_predictions_path),
        "cv_metrics_path": str(cv_metrics_path),
        "summary": summary,
    }


def evaluate_stage10x_holdout_and_cv(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Required Stage10XB adapter function.

    Parameters
    ----------
    config : dict
        Stage10XB execution config. It should contain discovered scripts, checkpoint paths,
        split files, output folders, and fixed scientific rules.

    Returns
    -------
    dict
        Paths to prediction/metric files plus a summary dictionary.
    """
    output_dir = Path(config.get("output_dir", r"D:\47\472\New-Papers\Anemia_Paper\Outputs\Stage10XB_Holdout_and_RepeatedCV_Evaluation"))
    logs_dir = _mkdir(output_dir / "logs")

    checkpoint = config.get("primary_checkpoint")
    paths = {"checkpoint": Path(checkpoint) if checkpoint else None}

    diagnostics: Dict[str, Any] = {
        "stage": "Stage10XC",
        "generated": _now(),
        "status": "started",
        "candidate_scripts": [],
        "attempts": [],
        "fixed_rules": config.get("fixed_rules", []),
        "primary_checkpoint": checkpoint,
    }

    # Ensure local Codes directory is importable.
    codes_dir = Path(config.get("codes_dir", r"D:\47\472\New-Papers\Anemia_Paper\Codes"))
    if str(codes_dir) not in sys.path:
        sys.path.insert(0, str(codes_dir))

    os.environ.setdefault("STAGE10X_EVAL_ONLY", "1")
    os.environ.setdefault("CPMRNET_EVAL_ONLY", "1")

    scripts = _candidate_stage10x_scripts(config)
    diagnostics["candidate_scripts"] = [str(p) for p in scripts]

    for script in scripts:
        attempt = {"script": str(script), "status": "started"}
        try:
            module = _import_module_from_path(script)
            attempt["imported"] = True

            high = _try_high_level_local_api(module, config)
            if isinstance(high, dict):
                attempt["status"] = "success_high_level_api"
                diagnostics["attempts"].append(attempt)
                diagnostics["status"] = "success_high_level_api"
                _write_json(logs_dir / "Stage10XC_adapter_diagnostics.json", diagnostics)
                return high

            conventional = _try_model_dataloader_api(module, config, paths)
            if isinstance(conventional, dict):
                attempt["status"] = "success_model_dataloader_api"
                diagnostics["attempts"].append(attempt)
                diagnostics["status"] = "success_model_dataloader_api"
                _write_json(logs_dir / "Stage10XC_adapter_diagnostics.json", diagnostics)
                return conventional

            attempt["status"] = "no_compatible_api_in_script"
        except Exception as e:
            attempt["status"] = "failed"
            attempt["error_type"] = type(e).__name__
            attempt["error"] = str(e)
            attempt["traceback"] = traceback.format_exc(limit=8)
        diagnostics["attempts"].append(attempt)

    diagnostics["status"] = "manual_connection_required"
    diagnostics["message"] = (
        "Stage10XC adapter was installed and executed, but no compatible callable local "
        "Stage10X evaluation API was found. Add one high-level function to "
        "Stage10X_Progressive_FineTuning_Contrastive_CPMRNet.py named "
        "evaluate_holdout_and_cv(config), or expose build_model/create_model plus "
        "build_dataloaders/create_dataloaders. No results were invented."
    )
    _write_json(logs_dir / "Stage10XC_adapter_diagnostics.json", diagnostics)

    raise NotImplementedError(diagnostics["message"])


if __name__ == "__main__":
    # Standalone smoke test: it verifies importability and writes a small diagnostic file.
    default_config = {
        "codes_dir": r"D:\47\472\New-Papers\Anemia_Paper\Codes",
        "outputs_dir": r"D:\47\472\New-Papers\Anemia_Paper\Outputs",
        "output_dir": r"D:\47\472\New-Papers\Anemia_Paper\Outputs\Stage10XB_Holdout_and_RepeatedCV_Evaluation",
        "benchmark_stage6d6b_auc": 0.7447,
        "fixed_rules": [
            "participant-level diagnosis only",
            "no architecture change",
            "no hyperparameter search",
            "use fixed Stage10I1 holdout and repeated-CV splits",
        ],
        "discovered_stage10x_scripts": [
            r"D:\47\472\New-Papers\Anemia_Paper\Codes\Stage10X_Progressive_FineTuning_Contrastive_CPMRNet.py"
        ],
        "primary_checkpoint": r"D:\47\472\New-Papers\Anemia_Paper\Outputs\Stage10X_Progressive_FineTuning_Contrastive_CPMRNet\models\ProgressiveContrastive_CPMRNet_best_val_auc.pt",
    }
    print("Stage10XC adapter file is installed. Run Stage10XB to execute it with full config.")
    try:
        evaluate_stage10x_holdout_and_cv(default_config)
    except Exception as exc:
        print(f"Adapter smoke test stopped safely: {type(exc).__name__}: {exc}")
