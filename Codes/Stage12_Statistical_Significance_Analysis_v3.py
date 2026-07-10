"""
Stage12_Statistical_Significance_Analysis_v3.py

Purpose
-------
Perform publication-grade statistical validation using only explicitly
approved canonical prediction files.

Main analyses
-------------
1. Participant-level bootstrap 95% confidence intervals.
2. DeLong tests for matched ROC-AUC comparisons.
3. Paired bootstrap confidence intervals for ROC-AUC differences.
4. Exact McNemar tests for matched binary decisions.
5. Wilcoxon signed-rank tests for matched fold-level ROC-AUC values.
6. Paired Cohen's d and rank-biserial effect size.

Workflow
--------
First run:
    Creates Stage12_Canonical_Evidence_Manifest.csv

Then:
    Edit the manifest and set include=True only for canonical evidence files.

Second run:
    Performs the full statistical analysis.

Important
---------
Models are compared only when:
- evaluation_scope is identical;
- participant IDs are identical;
- participant labels are identical;
- evaluation_group is identical.

This prevents invalid comparisons across validation, holdout, and CV outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Callable
import itertools
import json
import math
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm, wilcoxon
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. Configuration
# ============================================================

OUTPUT_ROOT = Path(
    r"D:\47\472\New-Papers\Anemia_Paper\Experiments\Outputs"
)

OUT_DIR = (
    OUTPUT_ROOT
    / "Stage12_Statistical_Significance_Analysis_v3"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH = OUT_DIR / "Stage12_Canonical_Evidence_Manifest.csv"

RANDOM_SEED = 42
N_BOOTSTRAP = 5000
ALPHA = 0.05
MIN_PARTICIPANTS = 10

EXPECTED_MODELS = {
    "RBFSVM",
    "Stage10L",
    "Stage10P",
    "Stage10X",
    "Stage10Y",
    "Stage10Z",
    "Stage10XD",
}

VALID_SCOPES = {
    "validation",
    "holdout",
    "test",
    "repeated_cv",
}

EXCLUDED_FOLDER_TOKENS = {
    "_section4_collected_files",
    "_section4_compact_files",
    "_paper_final_evidence",
    "stage12_statistical_significance_analysis",
    "stage12_statistical_significance_analysis_v3",
}

PREDICTION_PATTERNS = (
    "*prediction*.csv",
    "*predictions*.csv",
    "*participant_level_predictions*.csv",
    "*best_validation_predictions*.csv",
    "*holdout_predictions*.csv",
    "*test_predictions*.csv",
    "*out_of_fold*.csv",
    "*oof*.csv",
)

PARTICIPANT_CANDIDATES = (
    "participant_id",
    "participant",
    "patient_id",
    "subject_id",
    "pid",
    "case_id",
    "sample_id",
)

LABEL_CANDIDATES = (
    "y_true",
    "true_label",
    "label",
    "target",
    "anemia_label",
    "ground_truth",
    "actual",
)

PROBABILITY_CANDIDATES = (
    "y_prob",
    "probability",
    "prediction_probability",
    "pred_prob",
    "predicted_probability",
    "score",
    "anemia_probability",
    "positive_probability",
    "prob_anemia",
    "pred_score",
)

PREDICTION_CANDIDATES = (
    "y_pred",
    "prediction",
    "predicted_label",
    "pred_label",
    "decision",
)

REPEAT_CANDIDATES = (
    "repeat",
    "repeat_id",
    "repetition",
    "run",
    "run_id",
)

FOLD_CANDIDATES = (
    "fold",
    "fold_id",
    "cv_fold",
    "split_id",
)


# ============================================================
# 2. Utilities
# ============================================================

def normalize_name(value: object) -> str:
    text = str(value).strip().lower()
    chars = []

    for char in text:
        chars.append(char if char.isalnum() else "_")

    return "_".join(
        token for token in "".join(chars).split("_") if token
    )


def path_is_excluded(path: Path) -> bool:
    normalized = normalize_name(str(path))
    return any(token in normalized for token in EXCLUDED_FOLDER_TOKENS)


def safe_read_csv(
    path: Path,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    errors = []

    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(
                path,
                nrows=nrows,
                encoding=encoding,
                low_memory=False,
            )
        except Exception as exc:
            errors.append(str(exc))

    raise ValueError(
        f"Could not read {path}. Errors: {' | '.join(errors)}"
    )


def find_column(
    df: pd.DataFrame,
    candidates: tuple[str, ...],
) -> Optional[str]:
    mapping = {
        normalize_name(column): column
        for column in df.columns
    }

    for candidate in candidates:
        key = normalize_name(candidate)

        if key in mapping:
            return mapping[key]

    return None


def coerce_binary(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")

    values = set(numeric.dropna().unique())

    if values and values.issubset({0, 1, 0.0, 1.0}):
        return numeric.astype("Int64")

    mapping = {
        "0": 0,
        "1": 1,
        "normal": 0,
        "control": 0,
        "negative": 0,
        "healthy": 0,
        "non_anemia": 0,
        "non_anemic": 0,
        "anemia": 1,
        "anemic": 1,
        "positive": 1,
        "case": 1,
    }

    return (
        series.astype(str)
        .map(normalize_name)
        .map(mapping)
        .astype("Int64")
    )


def infer_model(path: Path) -> Optional[str]:
    text = normalize_name(str(path))

    rules = (
        ("Stage10XD", ("stage10xd",)),
        ("Stage10X", ("stage10x",)),
        ("Stage10Z", ("stage10z", "distilled")),
        ("Stage10Y", ("stage10y", "consistency")),
        ("Stage10P", ("stage10p",)),
        ("Stage10L", ("stage10l",)),
        ("RBFSVM", ("rbfsvm", "rbf_svm")),
    )

    for model, tokens in rules:
        if any(token in text for token in tokens):
            return model

    return None


def infer_scope(path: Path) -> str:
    text = normalize_name(str(path))

    if any(
        token in text
        for token in (
            "repeated_cv",
            "repeated_cross",
            "out_of_fold",
            "oof",
            "stage10p",
            "stage10xd",
        )
    ):
        return "repeated_cv"

    if "holdout" in text or "independent" in text:
        return "holdout"

    if "validation" in text or "best_validation" in text:
        return "validation"

    if "test" in text:
        return "test"

    return "unknown"


def detect_prediction_columns(
    path: Path,
) -> dict[str, Optional[str]]:
    df = safe_read_csv(path, nrows=500)

    participant_col = find_column(df, PARTICIPANT_CANDIDATES)
    label_col = find_column(df, LABEL_CANDIDATES)
    probability_col = find_column(df, PROBABILITY_CANDIDATES)
    prediction_col = find_column(df, PREDICTION_CANDIDATES)
    repeat_col = find_column(df, REPEAT_CANDIDATES)
    fold_col = find_column(df, FOLD_CANDIDATES)

    return {
        "participant_col": participant_col,
        "label_col": label_col,
        "probability_col": probability_col,
        "prediction_col": prediction_col,
        "repeat_col": repeat_col,
        "fold_col": fold_col,
    }


# ============================================================
# 3. Create canonical evidence manifest
# ============================================================

def discover_candidate_files() -> list[Path]:
    files: set[Path] = set()

    for pattern in PREDICTION_PATTERNS:
        for path in OUTPUT_ROOT.rglob(pattern):
            if not path.is_file():
                continue

            if path.suffix.lower() != ".csv":
                continue

            if path_is_excluded(path):
                continue

            model = infer_model(path)

            if model not in EXPECTED_MODELS:
                continue

            files.add(path)

    return sorted(files)


def create_manifest_template() -> pd.DataFrame:
    rows = []

    for path in discover_candidate_files():
        try:
            columns = detect_prediction_columns(path)
            head = safe_read_csv(path, nrows=500)

            model = infer_model(path)
            scope = infer_scope(path)

            n_rows = len(safe_read_csv(path))
            n_participants = None

            if columns["participant_col"]:
                n_participants = (
                    head[columns["participant_col"]]
                    .astype(str)
                    .nunique()
                )

            rows.append(
                {
                    "include": False,
                    "model": model,
                    "evaluation_scope": scope,
                    "evaluation_group": "",
                    "file_path": str(path),
                    "participant_col": columns["participant_col"],
                    "label_col": columns["label_col"],
                    "probability_col": columns["probability_col"],
                    "prediction_col": columns["prediction_col"],
                    "repeat_col": columns["repeat_col"],
                    "fold_col": columns["fold_col"],
                    "aggregate_by_participant": True,
                    "expected_reported_auc": "",
                    "notes": "",
                    "n_rows_detected": n_rows,
                    "n_participants_preview": n_participants,
                }
            )

        except Exception as exc:
            rows.append(
                {
                    "include": False,
                    "model": infer_model(path),
                    "evaluation_scope": infer_scope(path),
                    "evaluation_group": "",
                    "file_path": str(path),
                    "participant_col": "",
                    "label_col": "",
                    "probability_col": "",
                    "prediction_col": "",
                    "repeat_col": "",
                    "fold_col": "",
                    "aggregate_by_participant": True,
                    "expected_reported_auc": "",
                    "notes": f"Detection failed: {exc}",
                    "n_rows_detected": "",
                    "n_participants_preview": "",
                }
            )

    manifest = pd.DataFrame(rows)

    if not manifest.empty:
        manifest = manifest.sort_values(
            [
                "model",
                "evaluation_scope",
                "file_path",
            ]
        )

    manifest.to_csv(MANIFEST_PATH, index=False)
    return manifest


# ============================================================
# 4. Manifest validation
# ============================================================

def to_bool(value: object) -> bool:
    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def validate_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "include",
        "model",
        "evaluation_scope",
        "evaluation_group",
        "file_path",
        "participant_col",
        "label_col",
        "probability_col",
        "prediction_col",
        "repeat_col",
        "fold_col",
        "aggregate_by_participant",
        "expected_reported_auc",
        "notes",
    }

    missing = required_columns.difference(manifest.columns)

    if missing:
        raise ValueError(
            f"Manifest is missing columns: {sorted(missing)}"
        )

    manifest = manifest.copy()
    manifest["include"] = manifest["include"].map(to_bool)

    selected = manifest[manifest["include"]].copy()

    if selected.empty:
        raise ValueError(
            "No manifest rows are marked include=True."
        )

    errors = []

    for index, row in selected.iterrows():
        path = Path(str(row["file_path"]))

        if not path.exists():
            errors.append(
                f"Row {index}: file not found: {path}"
            )

        if row["model"] not in EXPECTED_MODELS:
            errors.append(
                f"Row {index}: invalid model {row['model']}"
            )

        if row["evaluation_scope"] not in VALID_SCOPES:
            errors.append(
                f"Row {index}: invalid scope "
                f"{row['evaluation_scope']}"
            )

        if not str(row["evaluation_group"]).strip():
            errors.append(
                f"Row {index}: evaluation_group is required."
            )

        for field in (
            "participant_col",
            "label_col",
            "probability_col",
        ):
            if not str(row[field]).strip():
                errors.append(
                    f"Row {index}: {field} is required."
                )

    if errors:
        raise ValueError(
            "Manifest validation failed:\n"
            + "\n".join(errors)
        )

    return selected


# ============================================================
# 5. Load canonical evidence
# ============================================================

def load_manifest_evidence(
    row: pd.Series,
) -> tuple[pd.DataFrame, dict[str, object]]:
    path = Path(str(row["file_path"]))
    df = safe_read_csv(path)

    participant_col = str(row["participant_col"])
    label_col = str(row["label_col"])
    probability_col = str(row["probability_col"])

    prediction_col = (
        str(row["prediction_col"]).strip()
        if pd.notna(row["prediction_col"])
        else ""
    )

    repeat_col = (
        str(row["repeat_col"]).strip()
        if pd.notna(row["repeat_col"])
        else ""
    )

    fold_col = (
        str(row["fold_col"]).strip()
        if pd.notna(row["fold_col"])
        else ""
    )

    required = [
        participant_col,
        label_col,
        probability_col,
    ]

    if prediction_col:
        required.append(prediction_col)

    if repeat_col:
        required.append(repeat_col)

    if fold_col:
        required.append(fold_col)

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(
            f"{path} is missing columns: {missing}"
        )

    normalized = pd.DataFrame(
        {
            "participant_id": df[participant_col].astype(str),
            "y_true": coerce_binary(df[label_col]),
            "y_prob": pd.to_numeric(
                df[probability_col],
                errors="coerce",
            ),
        }
    )

    if prediction_col:
        normalized["y_pred"] = coerce_binary(
            df[prediction_col]
        )

    if repeat_col:
        normalized["repeat"] = df[repeat_col].astype(str)

    if fold_col:
        normalized["fold"] = df[fold_col].astype(str)

    normalized = normalized.dropna(
        subset=[
            "participant_id",
            "y_true",
            "y_prob",
        ]
    ).copy()

    normalized["y_true"] = normalized["y_true"].astype(int)

    if not normalized["y_prob"].between(0, 1).all():
        raise ValueError(
            f"{path}: probability values outside [0, 1]."
        )

    if normalized["y_true"].nunique() != 2:
        raise ValueError(
            f"{path}: target is not binary."
        )

    conflicts = (
        normalized.groupby("participant_id")["y_true"]
        .nunique()
    )

    if (conflicts > 1).any():
        raise ValueError(
            f"{path}: participant label conflicts detected."
        )

    aggregate = to_bool(row["aggregate_by_participant"])

    raw_for_fold_analysis = normalized.copy()

    if aggregate:
        aggregation: dict[str, object] = {
            "y_true": "first",
            "y_prob": "mean",
        }

        if "y_pred" in normalized.columns:
            aggregation["y_pred"] = (
                lambda values:
                int(
                    pd.to_numeric(
                        values,
                        errors="coerce",
                    ).dropna().mean() >= 0.5
                )
            )

        normalized = (
            normalized.groupby(
                "participant_id",
                as_index=False,
            )
            .agg(aggregation)
        )

    if "y_pred" not in normalized.columns:
        normalized["y_pred"] = (
            normalized["y_prob"] >= 0.5
        ).astype(int)
    else:
        normalized["y_pred"] = (
            normalized["y_pred"].fillna(
                (normalized["y_prob"] >= 0.5).astype(int)
            )
        ).astype(int)

    if len(normalized) < MIN_PARTICIPANTS:
        raise ValueError(
            f"{path}: fewer than {MIN_PARTICIPANTS} participants."
        )

    metadata = {
        "model": row["model"],
        "evaluation_scope": row["evaluation_scope"],
        "evaluation_group": row["evaluation_group"],
        "source_file": str(path),
        "expected_reported_auc": row["expected_reported_auc"],
        "notes": row["notes"],
        "raw_for_fold_analysis": raw_for_fold_analysis,
    }

    return normalized, metadata


# ============================================================
# 6. Metrics
# ============================================================

def specificity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    return float(tn / (tn + fp)) if (tn + fp) else np.nan


def calculate_metrics(df: pd.DataFrame) -> dict[str, float]:
    y_true = df["y_true"].to_numpy(dtype=int)
    y_prob = df["y_prob"].to_numpy(dtype=float)
    y_pred = df["y_pred"].to_numpy(dtype=int)

    return {
        "n": len(df),
        "positives": int((y_true == 1).sum()),
        "negatives": int((y_true == 0).sum()),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_pred,
        ),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "specificity": specificity(y_true, y_pred),
        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(
            y_true,
            y_prob,
        ),
    }


def stratified_bootstrap(
    y_true: np.ndarray,
    values: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    positive = np.where(y_true == 1)[0]
    negative = np.where(y_true == 0)[0]

    estimates = []

    for _ in range(N_BOOTSTRAP):
        pos_sample = rng.choice(
            positive,
            size=len(positive),
            replace=True,
        )
        neg_sample = rng.choice(
            negative,
            size=len(negative),
            replace=True,
        )

        indices = np.concatenate(
            [pos_sample, neg_sample]
        )
        rng.shuffle(indices)

        try:
            value = metric(
                y_true[indices],
                values[indices],
            )

            if np.isfinite(value):
                estimates.append(value)

        except Exception:
            continue

    return np.asarray(estimates, dtype=float)


def ci95(distribution: np.ndarray) -> tuple[float, float, float]:
    if len(distribution) == 0:
        return np.nan, np.nan, np.nan

    return (
        float(np.mean(distribution)),
        float(np.percentile(distribution, 2.5)),
        float(np.percentile(distribution, 97.5)),
    )


# ============================================================
# 7. DeLong test
# ============================================================

def compute_midrank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = values[order]

    ranks = np.zeros(len(values), dtype=float)
    start = 0

    while start < len(values):
        end = start

        while (
            end < len(values)
            and sorted_values[end] == sorted_values[start]
        ):
            end += 1

        ranks[start:end] = (
            0.5 * (start + end - 1) + 1
        )
        start = end

    result = np.empty(len(values), dtype=float)
    result[order] = ranks

    return result


def fast_delong(
    predictions: np.ndarray,
    positive_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    m = positive_count
    n = predictions.shape[1] - m
    k = predictions.shape[0]

    positive = predictions[:, :m]
    negative = predictions[:, m:]

    tx = np.empty((k, m))
    ty = np.empty((k, n))
    tz = np.empty((k, m + n))

    for row in range(k):
        tx[row] = compute_midrank(positive[row])
        ty[row] = compute_midrank(negative[row])
        tz[row] = compute_midrank(predictions[row])

    aucs = (
        tz[:, :m].sum(axis=1) / (m * n)
        - (m + 1) / (2 * n)
    )

    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m

    covariance = (
        np.atleast_2d(np.cov(v01)) / m
        + np.atleast_2d(np.cov(v10)) / n
    )

    return aucs, covariance


def delong_test(
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
) -> dict[str, float]:
    order = np.argsort(-y_true)
    positive_count = int(y_true.sum())

    predictions = np.vstack(
        [score_a, score_b]
    )[:, order]

    aucs, covariance = fast_delong(
        predictions,
        positive_count,
    )

    difference = aucs[0] - aucs[1]

    variance = (
        covariance[0, 0]
        + covariance[1, 1]
        - 2 * covariance[0, 1]
    )

    if variance <= 0 or not np.isfinite(variance):
        z_value = np.nan
        p_value = np.nan
    else:
        z_value = difference / math.sqrt(variance)
        p_value = 2 * norm.sf(abs(z_value))

    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "auc_difference": float(difference),
        "delong_z": z_value,
        "delong_p": p_value,
    }


# ============================================================
# 8. Bootstrap difference and McNemar
# ============================================================

def bootstrap_auc_difference(
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
) -> dict[str, float]:
    rng = np.random.default_rng(RANDOM_SEED)

    positive = np.where(y_true == 1)[0]
    negative = np.where(y_true == 0)[0]

    differences = []

    for _ in range(N_BOOTSTRAP):
        pos_sample = rng.choice(
            positive,
            size=len(positive),
            replace=True,
        )
        neg_sample = rng.choice(
            negative,
            size=len(negative),
            replace=True,
        )

        indices = np.concatenate(
            [pos_sample, neg_sample]
        )
        rng.shuffle(indices)

        differences.append(
            roc_auc_score(
                y_true[indices],
                score_a[indices],
            )
            - roc_auc_score(
                y_true[indices],
                score_b[indices],
            )
        )

    distribution = np.asarray(differences)

    p_value = 2 * min(
        np.mean(distribution <= 0),
        np.mean(distribution >= 0),
    )

    return {
        "bootstrap_mean_difference": distribution.mean(),
        "bootstrap_ci95_lower": np.percentile(
            distribution,
            2.5,
        ),
        "bootstrap_ci95_upper": np.percentile(
            distribution,
            97.5,
        ),
        "bootstrap_p": min(1.0, p_value),
    }


def mcnemar_exact(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
) -> dict[str, float]:
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true

    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))

    discordant = b + c

    if discordant == 0:
        p_value = 1.0
    else:
        p_value = binomtest(
            min(b, c),
            n=discordant,
            p=0.5,
            alternative="two-sided",
        ).pvalue

    return {
        "a_correct_b_wrong": b,
        "a_wrong_b_correct": c,
        "discordant_pairs": discordant,
        "mcnemar_exact_p": p_value,
    }


# ============================================================
# 9. Fold-level tests
# ============================================================

def calculate_fold_auc(
    raw_df: pd.DataFrame,
) -> pd.DataFrame:
    if "fold" not in raw_df.columns:
        return pd.DataFrame()

    group_columns = ["fold"]

    if "repeat" in raw_df.columns:
        group_columns.insert(0, "repeat")

    rows = []

    for keys, group in raw_df.groupby(group_columns):
        if group["y_true"].nunique() != 2:
            continue

        fold_auc = roc_auc_score(
            group["y_true"],
            group["y_prob"],
        )

        if not isinstance(keys, tuple):
            keys = (keys,)

        rows.append(
            {
                "fold_key": "_".join(map(str, keys)),
                "roc_auc": fold_auc,
            }
        )

    return pd.DataFrame(rows)


def paired_cohens_d(differences: np.ndarray) -> float:
    sd = np.std(differences, ddof=1)

    return (
        float(np.mean(differences) / sd)
        if sd > 0
        else np.nan
    )


def rank_biserial(differences: np.ndarray) -> float:
    nonzero = differences[differences != 0]

    if len(nonzero) == 0:
        return 0.0

    ranks = (
        pd.Series(np.abs(nonzero))
        .rank(method="average")
        .to_numpy()
    )

    positive_sum = ranks[nonzero > 0].sum()
    negative_sum = ranks[nonzero < 0].sum()

    return float(
        (positive_sum - negative_sum)
        / ranks.sum()
    )


# ============================================================
# 10. Main analysis
# ============================================================

def main() -> None:
    start = time.perf_counter()

    if not MANIFEST_PATH.exists():
        manifest = create_manifest_template()

        print("=" * 80)
        print("Canonical evidence manifest created")
        print("=" * 80)
        print(f"Manifest: {MANIFEST_PATH}")
        print(f"Candidate rows: {len(manifest)}")
        print()
        print("Next:")
        print("1. Open the CSV.")
        print("2. Mark only canonical files include=True.")
        print("3. Fill evaluation_group.")
        print("4. Confirm all column names.")
        print("5. Run this script again.")
        return

    manifest = pd.read_csv(MANIFEST_PATH)
    selected_manifest = validate_manifest(manifest)

    evidence = []

    load_audit_rows = []

    for _, row in selected_manifest.iterrows():
        try:
            df, metadata = load_manifest_evidence(row)

            evidence.append(
                {
                    "data": df,
                    "metadata": metadata,
                }
            )

            load_audit_rows.append(
                {
                    "model": metadata["model"],
                    "evaluation_scope": metadata[
                        "evaluation_scope"
                    ],
                    "evaluation_group": metadata[
                        "evaluation_group"
                    ],
                    "source_file": metadata["source_file"],
                    "n_participants": len(df),
                    "status": "Loaded",
                    "error": "",
                }
            )

        except Exception as exc:
            load_audit_rows.append(
                {
                    "model": row["model"],
                    "evaluation_scope": row[
                        "evaluation_scope"
                    ],
                    "evaluation_group": row[
                        "evaluation_group"
                    ],
                    "source_file": row["file_path"],
                    "n_participants": "",
                    "status": "Failed",
                    "error": str(exc),
                }
            )

    load_audit_df = pd.DataFrame(load_audit_rows)
    load_audit_df.to_csv(
        OUT_DIR / "canonical_evidence_load_audit.csv",
        index=False,
    )

    if not evidence:
        raise RuntimeError(
            "No canonical evidence files loaded successfully."
        )

    metric_rows = []
    fold_tables = []

    for item_index, item in enumerate(evidence):
        df = item["data"]
        metadata = item["metadata"]

        metrics = calculate_metrics(df)

        y_true = df["y_true"].to_numpy(dtype=int)
        y_prob = df["y_prob"].to_numpy(dtype=float)

        roc_distribution = stratified_bootstrap(
            y_true,
            y_prob,
            roc_auc_score,
            seed=RANDOM_SEED + item_index,
        )

        pr_distribution = stratified_bootstrap(
            y_true,
            y_prob,
            average_precision_score,
            seed=RANDOM_SEED + 1000 + item_index,
        )

        roc_mean, roc_low, roc_high = ci95(
            roc_distribution
        )

        pr_mean, pr_low, pr_high = ci95(
            pr_distribution
        )

        expected_auc = pd.to_numeric(
            pd.Series(
                [metadata["expected_reported_auc"]]
            ),
            errors="coerce",
        ).iloc[0]

        auc_difference_from_expected = (
            metrics["roc_auc"] - expected_auc
            if pd.notna(expected_auc)
            else np.nan
        )

        metric_rows.append(
            {
                "model": metadata["model"],
                "evaluation_scope": metadata[
                    "evaluation_scope"
                ],
                "evaluation_group": metadata[
                    "evaluation_group"
                ],
                "source_file": metadata["source_file"],
                **metrics,
                "roc_auc_bootstrap_mean": roc_mean,
                "roc_auc_ci95_lower": roc_low,
                "roc_auc_ci95_upper": roc_high,
                "pr_auc_bootstrap_mean": pr_mean,
                "pr_auc_ci95_lower": pr_low,
                "pr_auc_ci95_upper": pr_high,
                "expected_reported_auc": expected_auc,
                "auc_difference_from_expected": (
                    auc_difference_from_expected
                ),
                "auc_matches_expected_within_0_001": (
                    abs(auc_difference_from_expected) <= 0.001
                    if pd.notna(
                        auc_difference_from_expected
                    )
                    else np.nan
                ),
                "bootstrap_iterations": N_BOOTSTRAP,
            }
        )

        fold_auc_df = calculate_fold_auc(
            metadata["raw_for_fold_analysis"]
        )

        if not fold_auc_df.empty:
            fold_auc_df["model"] = metadata["model"]
            fold_auc_df["evaluation_scope"] = metadata[
                "evaluation_scope"
            ]
            fold_auc_df["evaluation_group"] = metadata[
                "evaluation_group"
            ]
            fold_auc_df["source_file"] = metadata[
                "source_file"
            ]

            fold_tables.append(fold_auc_df)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(
        OUT_DIR / "canonical_bootstrap_confidence_intervals.csv",
        index=False,
    )

    pairwise_rows = []

    for item_a, item_b in itertools.combinations(
        evidence,
        2,
    ):
        meta_a = item_a["metadata"]
        meta_b = item_b["metadata"]

        if meta_a["model"] == meta_b["model"]:
            continue

        if (
            meta_a["evaluation_scope"]
            != meta_b["evaluation_scope"]
        ):
            continue

        if (
            meta_a["evaluation_group"]
            != meta_b["evaluation_group"]
        ):
            continue

        df_a = item_a["data"]
        df_b = item_b["data"]

        merged = df_a.merge(
            df_b,
            on=["participant_id", "y_true"],
            suffixes=("_a", "_b"),
            how="inner",
            validate="one_to_one",
        )

        if (
            len(merged) != len(df_a)
            or len(merged) != len(df_b)
        ):
            continue

        y_true = merged["y_true"].to_numpy(dtype=int)
        score_a = merged["y_prob_a"].to_numpy(dtype=float)
        score_b = merged["y_prob_b"].to_numpy(dtype=float)
        pred_a = merged["y_pred_a"].to_numpy(dtype=int)
        pred_b = merged["y_pred_b"].to_numpy(dtype=int)

        delong_results = delong_test(
            y_true,
            score_a,
            score_b,
        )

        bootstrap_results = bootstrap_auc_difference(
            y_true,
            score_a,
            score_b,
        )

        mcnemar_results = mcnemar_exact(
            y_true,
            pred_a,
            pred_b,
        )

        pairwise_rows.append(
            {
                "model_a": meta_a["model"],
                "model_b": meta_b["model"],
                "evaluation_scope": meta_a[
                    "evaluation_scope"
                ],
                "evaluation_group": meta_a[
                    "evaluation_group"
                ],
                "n_matched_participants": len(merged),
                "source_file_a": meta_a["source_file"],
                "source_file_b": meta_b["source_file"],
                **delong_results,
                **bootstrap_results,
                **mcnemar_results,
                "delong_significant_0_05": (
                    delong_results["delong_p"] < ALPHA
                    if np.isfinite(
                        delong_results["delong_p"]
                    )
                    else False
                ),
                "bootstrap_significant_0_05": (
                    bootstrap_results["bootstrap_p"]
                    < ALPHA
                ),
                "mcnemar_significant_0_05": (
                    mcnemar_results[
                        "mcnemar_exact_p"
                    ] < ALPHA
                ),
            }
        )

    pairwise_df = pd.DataFrame(pairwise_rows)
    pairwise_df.to_csv(
        OUT_DIR / "canonical_pairwise_statistical_tests.csv",
        index=False,
    )

    if fold_tables:
        fold_values_df = pd.concat(
            fold_tables,
            ignore_index=True,
        )
    else:
        fold_values_df = pd.DataFrame()

    fold_values_df.to_csv(
        OUT_DIR / "canonical_fold_level_auc_values.csv",
        index=False,
    )

    fold_test_rows = []

    if not fold_values_df.empty:
        for (
            scope,
            group,
        ), group_df in fold_values_df.groupby(
            [
                "evaluation_scope",
                "evaluation_group",
            ]
        ):
            pivot = group_df.pivot_table(
                index="fold_key",
                columns="model",
                values="roc_auc",
                aggfunc="mean",
            )

            for model_a, model_b in itertools.combinations(
                pivot.columns,
                2,
            ):
                paired = pivot[
                    [model_a, model_b]
                ].dropna()

                if len(paired) < 3:
                    continue

                x = paired[model_a].to_numpy(dtype=float)
                y = paired[model_b].to_numpy(dtype=float)
                differences = x - y

                try:
                    statistic, p_value = wilcoxon(
                        differences,
                        alternative="two-sided",
                        zero_method="wilcox",
                    )
                except Exception:
                    statistic = np.nan
                    p_value = np.nan

                fold_test_rows.append(
                    {
                        "model_a": model_a,
                        "model_b": model_b,
                        "evaluation_scope": scope,
                        "evaluation_group": group,
                        "n_matched_folds": len(paired),
                        "mean_auc_a": np.mean(x),
                        "mean_auc_b": np.mean(y),
                        "mean_difference": np.mean(
                            differences
                        ),
                        "median_difference": np.median(
                            differences
                        ),
                        "wilcoxon_statistic": statistic,
                        "wilcoxon_p": p_value,
                        "paired_cohens_d": paired_cohens_d(
                            differences
                        ),
                        "rank_biserial_correlation": (
                            rank_biserial(differences)
                        ),
                        "significant_0_05": (
                            p_value < ALPHA
                            if np.isfinite(p_value)
                            else False
                        ),
                    }
                )

    fold_tests_df = pd.DataFrame(fold_test_rows)
    fold_tests_df.to_csv(
        OUT_DIR / "canonical_fold_level_tests.csv",
        index=False,
    )

    manuscript_rows = []

    for _, row in metrics_df.iterrows():
        manuscript_rows.append(
            {
                "analysis": "Bootstrap ROC-AUC CI",
                "comparison": (
                    f"{row['model']} "
                    f"({row['evaluation_scope']}; "
                    f"{row['evaluation_group']})"
                ),
                "estimate": row["roc_auc"],
                "confidence_interval_or_effect": (
                    f"[{row['roc_auc_ci95_lower']:.4f}, "
                    f"{row['roc_auc_ci95_upper']:.4f}]"
                ),
                "p_value": np.nan,
                "significant": np.nan,
            }
        )

    for _, row in pairwise_df.iterrows():
        manuscript_rows.append(
            {
                "analysis": "DeLong ROC-AUC comparison",
                "comparison": (
                    f"{row['model_a']} vs "
                    f"{row['model_b']} "
                    f"({row['evaluation_group']})"
                ),
                "estimate": row["auc_difference"],
                "confidence_interval_or_effect": (
                    f"z={row['delong_z']:.4f}"
                    if np.isfinite(row["delong_z"])
                    else "z=NA"
                ),
                "p_value": row["delong_p"],
                "significant": row[
                    "delong_significant_0_05"
                ],
            }
        )

        manuscript_rows.append(
            {
                "analysis": "Paired bootstrap AUC difference",
                "comparison": (
                    f"{row['model_a']} vs "
                    f"{row['model_b']} "
                    f"({row['evaluation_group']})"
                ),
                "estimate": row[
                    "bootstrap_mean_difference"
                ],
                "confidence_interval_or_effect": (
                    f"[{row['bootstrap_ci95_lower']:.4f}, "
                    f"{row['bootstrap_ci95_upper']:.4f}]"
                ),
                "p_value": row["bootstrap_p"],
                "significant": row[
                    "bootstrap_significant_0_05"
                ],
            }
        )

        manuscript_rows.append(
            {
                "analysis": "Exact McNemar test",
                "comparison": (
                    f"{row['model_a']} vs "
                    f"{row['model_b']} "
                    f"({row['evaluation_group']})"
                ),
                "estimate": row["discordant_pairs"],
                "confidence_interval_or_effect": (
                    f"b={row['a_correct_b_wrong']}; "
                    f"c={row['a_wrong_b_correct']}"
                ),
                "p_value": row["mcnemar_exact_p"],
                "significant": row[
                    "mcnemar_significant_0_05"
                ],
            }
        )

    for _, row in fold_tests_df.iterrows():
        manuscript_rows.append(
            {
                "analysis": "Wilcoxon fold-level comparison",
                "comparison": (
                    f"{row['model_a']} vs "
                    f"{row['model_b']} "
                    f"({row['evaluation_group']})"
                ),
                "estimate": row["mean_difference"],
                "confidence_interval_or_effect": (
                    f"d={row['paired_cohens_d']:.4f}; "
                    f"r_rb="
                    f"{row['rank_biserial_correlation']:.4f}"
                ),
                "p_value": row["wilcoxon_p"],
                "significant": row[
                    "significant_0_05"
                ],
            }
        )

    manuscript_df = pd.DataFrame(manuscript_rows)
    manuscript_df.to_csv(
        OUT_DIR / "canonical_manuscript_statistical_summary.csv",
        index=False,
    )

    excel_path = (
        OUT_DIR
        / "Stage12_Canonical_Statistical_Report.xlsx"
    )

    with pd.ExcelWriter(
        excel_path,
        engine="openpyxl",
    ) as writer:
        selected_manifest.to_excel(
            writer,
            sheet_name="Canonical_Manifest",
            index=False,
        )

        load_audit_df.to_excel(
            writer,
            sheet_name="Load_Audit",
            index=False,
        )

        metrics_df.to_excel(
            writer,
            sheet_name="Bootstrap_CI",
            index=False,
        )

        pairwise_df.to_excel(
            writer,
            sheet_name="Pairwise_Tests",
            index=False,
        )

        fold_values_df.to_excel(
            writer,
            sheet_name="Fold_AUC",
            index=False,
        )

        fold_tests_df.to_excel(
            writer,
            sheet_name="Fold_Tests",
            index=False,
        )

        manuscript_df.to_excel(
            writer,
            sheet_name="Manuscript_Summary",
            index=False,
        )

    elapsed = time.perf_counter() - start

    metadata = {
        "selected_manifest_rows": len(selected_manifest),
        "successfully_loaded_evidence": len(evidence),
        "pairwise_comparisons": len(pairwise_df),
        "fold_level_comparisons": len(fold_tests_df),
        "bootstrap_iterations": N_BOOTSTRAP,
        "random_seed": RANDOM_SEED,
        "alpha": ALPHA,
        "elapsed_seconds": elapsed,
        "elapsed_minutes": elapsed / 60,
    }

    with open(
        OUT_DIR / "stage12_v3_run_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=2)

    print("=" * 80)
    print("Stage 12 v3 Canonical Statistical Analysis Completed")
    print("=" * 80)
    print(f"Manifest rows selected: {len(selected_manifest)}")
    print(f"Evidence files loaded: {len(evidence)}")
    print(f"Pairwise comparisons: {len(pairwise_df)}")
    print(f"Fold comparisons: {len(fold_tests_df)}")
    print(f"Elapsed time: {elapsed / 60:.2f} minutes")
    print(f"Output directory: {OUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()