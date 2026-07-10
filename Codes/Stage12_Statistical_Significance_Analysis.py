"""
Stage12_Statistical_Significance_Analysis.py

Statistical validation for the HandDx-200 anemia study.

The script performs:

1. Controlled discovery of final prediction files only.
2. Safe inference of participant, label, probability, prediction,
   repeat, and fold columns.
3. Participant-level aggregation of repeated-CV predictions.
4. Bootstrap 95% confidence intervals for ROC-AUC and PR-AUC.
5. Paired DeLong tests for ROC-AUC.
6. Paired bootstrap tests for ROC-AUC differences.
7. McNemar tests for paired binary decisions.
8. Fold-level Wilcoxon signed-rank tests.
9. Paired Cohen's d and rank-biserial effect sizes.
10. Automatic Excel and CSV reporting.

Important:
Pairwise tests are performed only when two models contain the exact
same participant IDs and matching ground-truth labels within the same
evaluation scope.

Outputs:
D:\\47\\472\\New-Papers\\Anemia_Paper\\Experiments\\Outputs\\
Stage12_Statistical_Significance_Analysis
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Callable, Optional
import itertools
import json
import math
import re
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

OUT_DIR = OUTPUT_ROOT / "Stage12_Statistical_Significance_Analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
N_BOOTSTRAP = 5000
ALPHA = 0.05
MIN_PARTICIPANTS = 10

# Restrict discovery to models relevant to the paper.
ALLOWED_MODELS = {
    "RBFSVM",
    "Stage10L",
    "Stage10P",
    "Stage10X",
    "Stage10XD",
    "Stage10Y",
    "Stage10Z",
}

# Models most important for pairwise analysis.
PRIMARY_MODELS = {
    "RBFSVM",
    "Stage10L",
    "Stage10X",
    "Stage10XD",
    "Stage10Z",
}

# Folders that must never be rescanned.
EXCLUDED_PATH_PARTS = {
    "_section4_collected_files",
    "_section4_compact_files",
    "_paper_final_evidence",
    "stage12_statistical_significance_analysis",
    "__pycache__",
    ".git",
}

# Intermediate or unsuitable outputs that should not be treated as
# final prediction evidence.
EXCLUDED_FILE_KEYWORDS = {
    "training_history",
    "epoch",
    "threshold_search",
    "threshold_sweep",
    "calibration_curve",
    "embedding_manifest",
    "feature_importance",
    "parameter_summary",
    "metadata",
    "preview",
    "audit",
    "decision_matrix",
    "comparison",
    "summary",
    "classification_report",
    "confusion_matrix",
}

PREDICTION_PATTERNS = (
    "*prediction*.csv",
    "*predictions*.csv",
    "*best_validation_predictions*.csv",
    "*participant_level_predictions*.csv",
    "*holdout_predictions*.csv",
    "*test_predictions*.csv",
    "*out_of_fold*.csv",
    "*oof*.csv",
)

FOLD_RESULT_PATTERNS = (
    "*fold*result*.csv",
    "*repeated*cv*result*.csv",
    "*cross*validation*result*.csv",
    "*cv*summary*.csv",
)

PARTICIPANT_CANDIDATES = (
    "participant_id",
    "participant",
    "patient_id",
    "subject_id",
    "subject",
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
    "diagnosis",
    "class_label",
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
    "p_hat",
    "pred_score",
    "prob",
)

PREDICTION_CANDIDATES = (
    "y_pred",
    "prediction",
    "predicted_label",
    "pred_label",
    "class_prediction",
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
    "split",
    "split_id",
)

ROC_AUC_CANDIDATES = (
    "roc_auc",
    "test_roc_auc",
    "val_roc_auc",
    "validation_roc_auc",
    "auc",
)


# ============================================================
# 2. Data classes
# ============================================================

@dataclass
class PredictionEvidence:
    model: str
    scope: str
    source_file: Path
    participant_col: str
    label_col: str
    probability_col: str
    prediction_col: Optional[str]
    repeat_col: Optional[str]
    fold_col: Optional[str]
    dataframe: pd.DataFrame
    content_signature: str
    participant_signature: str
    selection_score: float


# ============================================================
# 3. Generic utilities
# ============================================================

def normalize_name(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def normalized_columns(df: pd.DataFrame) -> dict[str, str]:
    return {normalize_name(col): col for col in df.columns}


def find_named_column(
    df: pd.DataFrame,
    candidates: tuple[str, ...],
) -> Optional[str]:
    mapping = normalized_columns(df)

    for candidate in candidates:
        normalized = normalize_name(candidate)
        if normalized in mapping:
            return mapping[normalized]

    return None


def path_is_excluded(path: Path) -> bool:
    normalized_parts = {normalize_name(part) for part in path.parts}
    return bool(normalized_parts.intersection(EXCLUDED_PATH_PARTS))


def file_is_excluded(path: Path) -> bool:
    name = normalize_name(path.name)
    return any(keyword in name for keyword in EXCLUDED_FILE_KEYWORDS)


def safe_read_csv(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    errors: list[str] = []

    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(
                path,
                nrows=nrows,
                encoding=encoding,
                low_memory=False,
            )
        except Exception as exc:
            errors.append(f"{encoding}: {exc}")

    raise ValueError(
        f"Unable to read CSV: {path}\n" + "\n".join(errors)
    )


def coerce_binary(series: pd.Series) -> pd.Series:
    """
    Convert common binary labels to 0/1.

    Supported examples:
    0/1, normal/anemia, negative/positive, control/case,
    no/yes, false/true.
    """
    raw = series.copy()

    numeric = pd.to_numeric(raw, errors="coerce")
    non_null_numeric = numeric.dropna()

    if len(non_null_numeric) > 0:
        unique_numeric = set(non_null_numeric.unique().tolist())

        if unique_numeric.issubset({0, 1, 0.0, 1.0}):
            return numeric.astype("Int64")

    mapping = {
        "0": 0,
        "1": 1,
        "normal": 0,
        "control": 0,
        "negative": 0,
        "non_anemia": 0,
        "non_anemic": 0,
        "healthy": 0,
        "false": 0,
        "no": 0,
        "anemia": 1,
        "anemic": 1,
        "case": 1,
        "positive": 1,
        "true": 1,
        "yes": 1,
    }

    normalized = raw.astype(str).map(normalize_name)
    converted = normalized.map(mapping)
    return converted.astype("Int64")


def is_binary_label_column(series: pd.Series) -> bool:
    converted = coerce_binary(series).dropna()

    return (
        len(converted) >= MIN_PARTICIPANTS
        and converted.nunique() == 2
        and set(converted.unique()).issubset({0, 1})
    )


def infer_label_column(df: pd.DataFrame) -> Optional[str]:
    named = find_named_column(df, LABEL_CANDIDATES)

    if named is not None and is_binary_label_column(df[named]):
        return named

    # Strict fallback. Avoid IDs, folds, repetitions, and predictions.
    forbidden_tokens = {
        "id",
        "fold",
        "repeat",
        "run",
        "epoch",
        "index",
        "prediction",
        "pred",
        "probability",
        "score",
    }

    for column in df.columns:
        normalized = normalize_name(column)

        if any(token in normalized for token in forbidden_tokens):
            continue

        if is_binary_label_column(df[column]):
            return column

    return None


def infer_probability_column(
    df: pd.DataFrame,
    label_col: str,
) -> Optional[str]:
    named = find_named_column(df, PROBABILITY_CANDIDATES)

    if named is not None and named != label_col:
        values = pd.to_numeric(df[named], errors="coerce").dropna()

        if (
            len(values) >= MIN_PARTICIPANTS
            and values.nunique() > 2
            and values.between(0, 1).all()
        ):
            return named

    best_column: Optional[str] = None
    best_score = -math.inf

    for column in df.columns:
        if column == label_col:
            continue

        normalized = normalize_name(column)

        if any(
            token in normalized
            for token in (
                "id",
                "fold",
                "repeat",
                "run",
                "epoch",
                "index",
                "label",
                "target",
            )
        ):
            continue

        values = pd.to_numeric(df[column], errors="coerce").dropna()

        if len(values) < MIN_PARTICIPANTS:
            continue

        if not values.between(0, 1).all():
            continue

        if values.nunique() <= 2:
            continue

        score = float(values.nunique())

        if "prob" in normalized:
            score += 1000

        if "score" in normalized:
            score += 500

        if "anemia" in normalized:
            score += 200

        if score > best_score:
            best_score = score
            best_column = column

    return best_column


def infer_prediction_column(
    df: pd.DataFrame,
    label_col: str,
) -> Optional[str]:
    named = find_named_column(df, PREDICTION_CANDIDATES)

    if named is not None and named != label_col:
        values = coerce_binary(df[named]).dropna()

        if values.nunique() <= 2:
            return named

    for column in df.columns:
        if column == label_col:
            continue

        normalized = normalize_name(column)

        if not any(
            token in normalized
            for token in ("pred", "decision", "class")
        ):
            continue

        values = coerce_binary(df[column]).dropna()

        if len(values) >= MIN_PARTICIPANTS and values.nunique() <= 2:
            return column

    return None


def infer_model_name(path: Path) -> Optional[str]:
    text = normalize_name(str(path))

    # Longer names must be checked first.
    rules = (
        ("Stage10XD", ("stage10xd",)),
        ("Stage10X", ("stage10x",)),
        ("Stage10Z", ("stage10z", "distilled")),
        ("Stage10Y", ("stage10y", "consistency_regularized")),
        ("Stage10P", ("stage10p",)),
        ("Stage10L", ("stage10l",)),
        ("RBFSVM", ("rbfsvm", "rbf_svm")),
    )

    for model, tokens in rules:
        if any(token in text for token in tokens):
            return model

    return None


def infer_scope(path: Path, df: pd.DataFrame) -> str:
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

    if any(
        token in text
        for token in (
            "holdout",
            "independent_test",
            "independent_evaluation",
        )
    ):
        return "holdout"

    if "test" in text:
        return "test"

    if any(token in text for token in ("validation", "val_prediction")):
        return "validation"

    # Inspect optional split column.
    split_col = find_named_column(
        df,
        ("split", "subset", "partition", "dataset_split"),
    )

    if split_col is not None:
        values = {
            normalize_name(value)
            for value in df[split_col].dropna().unique()
        }

        if "holdout" in values:
            return "holdout"

        if "test" in values:
            return "test"

        if "validation" in values or "val" in values:
            return "validation"

    return "unknown"


def build_content_signature(df: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(
        df.sort_index(axis=1),
        index=True,
    ).values

    return sha256(hashed.tobytes()).hexdigest()


def build_participant_signature(df: pd.DataFrame) -> str:
    ordered = (
        df[["participant_id", "y_true"]]
        .sort_values("participant_id")
        .astype(str)
    )

    text = "\n".join(
        ordered["participant_id"] + "|" + ordered["y_true"]
    )

    return sha256(text.encode("utf-8")).hexdigest()


def evidence_selection_score(
    path: Path,
    model: str,
    scope: str,
    n_participants: int,
) -> float:
    text = normalize_name(str(path))
    score = 0.0

    if model in PRIMARY_MODELS:
        score += 100

    if scope == "holdout":
        score += 80
    elif scope == "repeated_cv":
        score += 70
    elif scope == "test":
        score += 60
    elif scope == "validation":
        score += 40

    if "final" in text:
        score += 30

    if "participant_level" in text:
        score += 20

    if "best" in text:
        score += 10

    if "prediction" in text:
        score += 10

    if "aggregated" in text or "out_of_fold" in text:
        score += 15

    if "tables" in text:
        score += 5

    score += min(n_participants, 1000) / 1000

    return score


# ============================================================
# 4. Normalize prediction evidence
# ============================================================

def normalize_prediction_file(
    path: Path,
) -> tuple[Optional[PredictionEvidence], str]:
    try:
        head = safe_read_csv(path, nrows=500)

        model = infer_model_name(path)

        if model is None:
            return None, "Model was not in the final allowed-model set."

        if model not in ALLOWED_MODELS:
            return None, f"Model {model} was excluded."

        participant_col = find_named_column(
            head,
            PARTICIPANT_CANDIDATES,
        )
        label_col = infer_label_column(head)

        if participant_col is None:
            return None, "No participant identifier column."

        if label_col is None:
            return None, "No safe binary label column."

        probability_col = infer_probability_column(head, label_col)
        prediction_col = infer_prediction_column(head, label_col)

        if probability_col is None:
            return None, "No valid probability column."

        repeat_col = find_named_column(head, REPEAT_CANDIDATES)
        fold_col = find_named_column(head, FOLD_CANDIDATES)

        df = safe_read_csv(path)

        required = [participant_col, label_col, probability_col]

        if prediction_col is not None:
            required.append(prediction_col)

        if repeat_col is not None:
            required.append(repeat_col)

        if fold_col is not None:
            required.append(fold_col)

        df = df[required].copy()

        rename_map = {
            participant_col: "participant_id",
            label_col: "y_true",
            probability_col: "y_prob",
        }

        if prediction_col is not None:
            rename_map[prediction_col] = "y_pred"

        if repeat_col is not None:
            rename_map[repeat_col] = "repeat"

        if fold_col is not None:
            rename_map[fold_col] = "fold"

        df = df.rename(columns=rename_map)

        df["participant_id"] = df["participant_id"].astype(str)
        df["y_true"] = coerce_binary(df["y_true"])
        df["y_prob"] = pd.to_numeric(df["y_prob"], errors="coerce")

        if "y_pred" in df.columns:
            df["y_pred"] = coerce_binary(df["y_pred"])

        df = df.dropna(
            subset=["participant_id", "y_true", "y_prob"]
        ).copy()

        df["y_true"] = df["y_true"].astype(int)

        if not df["y_prob"].between(0, 1).all():
            return None, "Probability values were outside [0, 1]."

        if df["y_true"].nunique() != 2:
            return None, "The target was not binary after normalization."

        # Verify label consistency across repeated predictions.
        label_counts = (
            df.groupby("participant_id")["y_true"].nunique()
        )

        if (label_counts > 1).any():
            return None, "Conflicting labels for one or more participants."

        scope = infer_scope(path, df)

        # Aggregate all repeated predictions to one probability per participant.
        aggregation: dict[str, object] = {
            "y_true": "first",
            "y_prob": "mean",
        }

        if "y_pred" in df.columns:
            aggregation["y_pred"] = (
                lambda values:
                int(
                    pd.to_numeric(
                        values,
                        errors="coerce",
                    ).dropna().mean() >= 0.5
                )
                if len(
                    pd.to_numeric(
                        values,
                        errors="coerce",
                    ).dropna()
                ) > 0
                else np.nan
            )

        participant_df = (
            df.groupby("participant_id", as_index=False)
            .agg(aggregation)
        )

        if "y_pred" not in participant_df.columns:
            participant_df["y_pred"] = (
                participant_df["y_prob"] >= 0.5
            ).astype(int)
        else:
            participant_df["y_pred"] = (
                participant_df["y_pred"]
                .fillna(
                    (participant_df["y_prob"] >= 0.5).astype(int)
                )
                .astype(int)
            )

        if len(participant_df) < MIN_PARTICIPANTS:
            return None, "Too few participant-level observations."

        content_signature = build_content_signature(participant_df)
        participant_signature = build_participant_signature(
            participant_df
        )

        score = evidence_selection_score(
            path,
            model,
            scope,
            len(participant_df),
        )

        evidence = PredictionEvidence(
            model=model,
            scope=scope,
            source_file=path,
            participant_col=participant_col,
            label_col=label_col,
            probability_col=probability_col,
            prediction_col=prediction_col,
            repeat_col=repeat_col,
            fold_col=fold_col,
            dataframe=participant_df,
            content_signature=content_signature,
            participant_signature=participant_signature,
            selection_score=score,
        )

        return evidence, "Accepted"

    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


# ============================================================
# 5. Controlled discovery and deduplication
# ============================================================

def discover_prediction_candidates() -> list[Path]:
    discovered: set[Path] = set()

    for pattern in PREDICTION_PATTERNS:
        for path in OUTPUT_ROOT.rglob(pattern):
            if not path.is_file():
                continue

            if path.suffix.lower() != ".csv":
                continue

            if path_is_excluded(path):
                continue

            if file_is_excluded(path):
                continue

            if infer_model_name(path) is None:
                continue

            discovered.add(path)

    return sorted(discovered)


def select_final_evidence(
    accepted: list[PredictionEvidence],
) -> tuple[list[PredictionEvidence], pd.DataFrame]:
    """
    Keep one highest-ranked unique file per:
    model + scope + participant set.

    Exact content duplicates are also removed.
    """
    audit_rows: list[dict[str, object]] = []
    selected: list[PredictionEvidence] = []

    groups: dict[
        tuple[str, str, str],
        list[PredictionEvidence],
    ] = {}

    for evidence in accepted:
        key = (
            evidence.model,
            evidence.scope,
            evidence.participant_signature,
        )
        groups.setdefault(key, []).append(evidence)

    seen_content_signatures: set[str] = set()

    for key, candidates in groups.items():
        ranked = sorted(
            candidates,
            key=lambda item: item.selection_score,
            reverse=True,
        )

        chosen: Optional[PredictionEvidence] = None

        for candidate in ranked:
            if candidate.content_signature in seen_content_signatures:
                audit_rows.append(
                    {
                        "model": candidate.model,
                        "scope": candidate.scope,
                        "source_file": str(candidate.source_file),
                        "selection_score": candidate.selection_score,
                        "selected": False,
                        "reason": "Exact prediction-content duplicate.",
                    }
                )
                continue

            if chosen is None:
                chosen = candidate
                selected.append(candidate)
                seen_content_signatures.add(
                    candidate.content_signature
                )

                audit_rows.append(
                    {
                        "model": candidate.model,
                        "scope": candidate.scope,
                        "source_file": str(candidate.source_file),
                        "selection_score": candidate.selection_score,
                        "selected": True,
                        "reason": "Highest-ranked evidence for model, scope, and participant set.",
                    }
                )
            else:
                audit_rows.append(
                    {
                        "model": candidate.model,
                        "scope": candidate.scope,
                        "source_file": str(candidate.source_file),
                        "selection_score": candidate.selection_score,
                        "selected": False,
                        "reason": "Lower-ranked duplicate evidence for the same model, scope, and participant set.",
                    }
                )

    return selected, pd.DataFrame(audit_rows)


# ============================================================
# 6. Metrics and confidence intervals
# ============================================================

def specificity_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    denominator = tn + fp
    return float(tn / denominator) if denominator > 0 else np.nan


def compute_metrics(df: pd.DataFrame) -> dict[str, float]:
    y_true = df["y_true"].to_numpy(dtype=int)
    y_prob = df["y_prob"].to_numpy(dtype=float)
    y_pred = df["y_pred"].to_numpy(dtype=int)

    return {
        "n": int(len(df)),
        "positives": int((y_true == 1).sum()),
        "negatives": int((y_true == 0).sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "specificity": specificity_score(y_true, y_pred),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(
            average_precision_score(y_true, y_prob)
        ),
    }


def stratified_bootstrap_distribution(
    y_true: np.ndarray,
    values: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """
    Stratified participant-level bootstrap preserving class presence.
    """
    rng = np.random.default_rng(seed)

    y_true = np.asarray(y_true, dtype=int)
    values = np.asarray(values)

    positive_indices = np.where(y_true == 1)[0]
    negative_indices = np.where(y_true == 0)[0]

    if len(positive_indices) == 0 or len(negative_indices) == 0:
        return np.array([], dtype=float)

    estimates: list[float] = []

    for _ in range(n_bootstrap):
        sampled_positive = rng.choice(
            positive_indices,
            size=len(positive_indices),
            replace=True,
        )
        sampled_negative = rng.choice(
            negative_indices,
            size=len(negative_indices),
            replace=True,
        )

        indices = np.concatenate(
            [sampled_positive, sampled_negative]
        )
        rng.shuffle(indices)

        try:
            estimate = metric(
                y_true[indices],
                values[indices],
            )

            if np.isfinite(estimate):
                estimates.append(float(estimate))
        except Exception:
            continue

    return np.asarray(estimates, dtype=float)


def confidence_interval_from_distribution(
    distribution: np.ndarray,
) -> tuple[float, float, float]:
    if len(distribution) == 0:
        return np.nan, np.nan, np.nan

    lower = 100 * (ALPHA / 2)
    upper = 100 * (1 - ALPHA / 2)

    return (
        float(np.mean(distribution)),
        float(np.percentile(distribution, lower)),
        float(np.percentile(distribution, upper)),
    )


# ============================================================
# 7. DeLong implementation
# ============================================================

def compute_midrank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = values[order]
    n = len(values)

    ranks = np.zeros(n, dtype=float)
    start = 0

    while start < n:
        end = start

        while (
            end < n
            and sorted_values[end] == sorted_values[start]
        ):
            end += 1

        ranks[start:end] = 0.5 * (start + end - 1) + 1
        start = end

    output = np.empty(n, dtype=float)
    output[order] = ranks
    return output


def fast_delong(
    predictions_sorted_transposed: np.ndarray,
    positive_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    m = positive_count
    n = predictions_sorted_transposed.shape[1] - m
    k = predictions_sorted_transposed.shape[0]

    positive = predictions_sorted_transposed[:, :m]
    negative = predictions_sorted_transposed[:, m:]

    tx = np.empty((k, m))
    ty = np.empty((k, n))
    tz = np.empty((k, m + n))

    for row in range(k):
        tx[row] = compute_midrank(positive[row])
        ty[row] = compute_midrank(negative[row])
        tz[row] = compute_midrank(
            predictions_sorted_transposed[row]
        )

    aucs = (
        tz[:, :m].sum(axis=1) / (m * n)
        - (m + 1.0) / (2.0 * n)
    )

    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m

    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))

    covariance = sx / m + sy / n
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

    difference = float(aucs[0] - aucs[1])

    variance = (
        covariance[0, 0]
        + covariance[1, 1]
        - 2 * covariance[0, 1]
    )

    if not np.isfinite(variance) or variance <= 0:
        z_value = np.nan
        p_value = np.nan
    else:
        z_value = difference / math.sqrt(variance)
        p_value = 2 * norm.sf(abs(z_value))

    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "auc_difference_a_minus_b": difference,
        "delong_z": float(z_value)
        if np.isfinite(z_value)
        else np.nan,
        "delong_p_value": float(p_value)
        if np.isfinite(p_value)
        else np.nan,
    }


# ============================================================
# 8. Pairwise bootstrap and McNemar
# ============================================================

def paired_bootstrap_auc_difference(
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)

    positive_indices = np.where(y_true == 1)[0]
    negative_indices = np.where(y_true == 0)[0]

    differences: list[float] = []

    for _ in range(n_bootstrap):
        sampled_positive = rng.choice(
            positive_indices,
            size=len(positive_indices),
            replace=True,
        )
        sampled_negative = rng.choice(
            negative_indices,
            size=len(negative_indices),
            replace=True,
        )

        indices = np.concatenate(
            [sampled_positive, sampled_negative]
        )
        rng.shuffle(indices)

        try:
            auc_a = roc_auc_score(
                y_true[indices],
                score_a[indices],
            )
            auc_b = roc_auc_score(
                y_true[indices],
                score_b[indices],
            )
            differences.append(float(auc_a - auc_b))
        except Exception:
            continue

    distribution = np.asarray(differences, dtype=float)

    if len(distribution) == 0:
        return {
            "bootstrap_mean_auc_difference": np.nan,
            "bootstrap_ci95_lower": np.nan,
            "bootstrap_ci95_upper": np.nan,
            "bootstrap_two_sided_p": np.nan,
        }

    lower = float(np.percentile(distribution, 2.5))
    upper = float(np.percentile(distribution, 97.5))

    proportion_nonpositive = np.mean(distribution <= 0)
    proportion_nonnegative = np.mean(distribution >= 0)

    p_value = min(
        1.0,
        2 * min(
            proportion_nonpositive,
            proportion_nonnegative,
        ),
    )

    return {
        "bootstrap_mean_auc_difference": float(
            distribution.mean()
        ),
        "bootstrap_ci95_lower": lower,
        "bootstrap_ci95_upper": upper,
        "bootstrap_two_sided_p": float(p_value),
    }


def exact_mcnemar_test(
    y_true: np.ndarray,
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
) -> dict[str, float]:
    correct_a = prediction_a == y_true
    correct_b = prediction_b == y_true

    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    discordant = b + c

    if discordant == 0:
        p_value = 1.0
    else:
        p_value = float(
            binomtest(
                min(b, c),
                n=discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )

    return {
        "mcnemar_a_correct_b_wrong": b,
        "mcnemar_a_wrong_b_correct": c,
        "mcnemar_discordant_pairs": discordant,
        "mcnemar_exact_p_value": p_value,
    }


# ============================================================
# 9. Run controlled prediction analysis
# ============================================================

start_time = time.perf_counter()

candidate_paths = discover_prediction_candidates()

discovery_rows: list[dict[str, object]] = []
accepted_evidence: list[PredictionEvidence] = []

for index, path in enumerate(candidate_paths, start=1):
    evidence, reason = normalize_prediction_file(path)

    discovery_rows.append(
        {
            "candidate_number": index,
            "source_file": str(path),
            "accepted": evidence is not None,
            "reason": reason,
            "model": evidence.model if evidence else None,
            "scope": evidence.scope if evidence else None,
            "n_participants": (
                len(evidence.dataframe)
                if evidence
                else None
            ),
            "selection_score": (
                evidence.selection_score
                if evidence
                else None
            ),
        }
    )

    if evidence is not None:
        accepted_evidence.append(evidence)

selected_evidence, selection_audit = select_final_evidence(
    accepted_evidence
)

discovery_df = pd.DataFrame(discovery_rows)
discovery_df.to_csv(
    OUT_DIR / "prediction_discovery_audit.csv",
    index=False,
)

selection_audit.to_csv(
    OUT_DIR / "selected_prediction_evidence_audit.csv",
    index=False,
)


# ============================================================
# 10. Per-model confidence intervals
# ============================================================

metric_rows: list[dict[str, object]] = []

for evidence in selected_evidence:
    df = evidence.dataframe.sort_values(
        "participant_id"
    ).reset_index(drop=True)

    metrics = compute_metrics(df)

    y_true = df["y_true"].to_numpy(dtype=int)
    y_prob = df["y_prob"].to_numpy(dtype=float)

    roc_distribution = stratified_bootstrap_distribution(
        y_true,
        y_prob,
        roc_auc_score,
    )

    pr_distribution = stratified_bootstrap_distribution(
        y_true,
        y_prob,
        average_precision_score,
    )

    roc_mean, roc_lower, roc_upper = (
        confidence_interval_from_distribution(
            roc_distribution
        )
    )

    pr_mean, pr_lower, pr_upper = (
        confidence_interval_from_distribution(
            pr_distribution
        )
    )

    metric_rows.append(
        {
            "model": evidence.model,
            "evaluation_scope": evidence.scope,
            "source_file": str(evidence.source_file),
            "participant_signature": (
                evidence.participant_signature
            ),
            **metrics,
            "roc_auc_bootstrap_mean": roc_mean,
            "roc_auc_ci95_lower": roc_lower,
            "roc_auc_ci95_upper": roc_upper,
            "pr_auc_bootstrap_mean": pr_mean,
            "pr_auc_ci95_lower": pr_lower,
            "pr_auc_ci95_upper": pr_upper,
            "bootstrap_iterations": N_BOOTSTRAP,
        }
    )

metrics_df = pd.DataFrame(metric_rows)

if not metrics_df.empty:
    metrics_df = metrics_df.sort_values(
        ["evaluation_scope", "roc_auc"],
        ascending=[True, False],
    )

metrics_df.to_csv(
    OUT_DIR / "final_model_bootstrap_confidence_intervals.csv",
    index=False,
)


# ============================================================
# 11. Valid pairwise comparisons
# ============================================================

pairwise_rows: list[dict[str, object]] = []

for evidence_a, evidence_b in itertools.combinations(
    selected_evidence,
    2,
):
    if evidence_a.model == evidence_b.model:
        continue

    # Never compare validation against holdout or repeated CV.
    if evidence_a.scope != evidence_b.scope:
        continue

    # Require the exact same participant and label set.
    if (
        evidence_a.participant_signature
        != evidence_b.participant_signature
    ):
        continue

    merged = evidence_a.dataframe.merge(
        evidence_b.dataframe,
        on=["participant_id", "y_true"],
        suffixes=("_a", "_b"),
        how="inner",
        validate="one_to_one",
    )

    if len(merged) < MIN_PARTICIPANTS:
        continue

    y_true = merged["y_true"].to_numpy(dtype=int)
    score_a = merged["y_prob_a"].to_numpy(dtype=float)
    score_b = merged["y_prob_b"].to_numpy(dtype=float)
    prediction_a = merged["y_pred_a"].to_numpy(dtype=int)
    prediction_b = merged["y_pred_b"].to_numpy(dtype=int)

    delong_results = delong_test(
        y_true,
        score_a,
        score_b,
    )

    bootstrap_results = paired_bootstrap_auc_difference(
        y_true,
        score_a,
        score_b,
    )

    mcnemar_results = exact_mcnemar_test(
        y_true,
        prediction_a,
        prediction_b,
    )

    pairwise_rows.append(
        {
            "model_a": evidence_a.model,
            "model_b": evidence_b.model,
            "evaluation_scope": evidence_a.scope,
            "n_matched_participants": len(merged),
            "source_file_a": str(
                evidence_a.source_file
            ),
            "source_file_b": str(
                evidence_b.source_file
            ),
            **delong_results,
            **bootstrap_results,
            **mcnemar_results,
            "statistically_significant_delong_0_05": (
                delong_results["delong_p_value"] < ALPHA
                if np.isfinite(
                    delong_results["delong_p_value"]
                )
                else False
            ),
            "statistically_significant_bootstrap_0_05": (
                bootstrap_results[
                    "bootstrap_two_sided_p"
                ] < ALPHA
                if np.isfinite(
                    bootstrap_results[
                        "bootstrap_two_sided_p"
                    ]
                )
                else False
            ),
            "statistically_significant_mcnemar_0_05": (
                mcnemar_results[
                    "mcnemar_exact_p_value"
                ] < ALPHA
            ),
        }
    )

pairwise_df = pd.DataFrame(pairwise_rows)

pairwise_df.to_csv(
    OUT_DIR / "valid_pairwise_statistical_tests.csv",
    index=False,
)


# ============================================================
# 12. Fold-level repeated-CV analysis
# ============================================================

def discover_fold_result_files() -> list[Path]:
    discovered: set[Path] = set()

    for pattern in FOLD_RESULT_PATTERNS:
        for path in OUTPUT_ROOT.rglob(pattern):
            if not path.is_file():
                continue

            if path.suffix.lower() != ".csv":
                continue

            if path_is_excluded(path):
                continue

            if infer_model_name(path) is None:
                continue

            discovered.add(path)

    return sorted(discovered)


def normalize_fold_result_file(
    path: Path,
) -> tuple[Optional[pd.DataFrame], str]:
    try:
        model = infer_model_name(path)

        if model is None:
            return None, "Unknown model."

        df = safe_read_csv(path)

        auc_col = find_named_column(
            df,
            ROC_AUC_CANDIDATES,
        )
        fold_col = find_named_column(df, FOLD_CANDIDATES)
        repeat_col = find_named_column(
            df,
            REPEAT_CANDIDATES,
        )

        if auc_col is None:
            return None, "No ROC-AUC column."

        normalized = pd.DataFrame()
        normalized["roc_auc"] = pd.to_numeric(
            df[auc_col],
            errors="coerce",
        )

        if fold_col is not None:
            normalized["fold"] = df[fold_col].astype(str)
        else:
            normalized["fold"] = np.arange(len(df)).astype(str)

        if repeat_col is not None:
            normalized["repeat"] = df[repeat_col].astype(str)
        else:
            normalized["repeat"] = "1"

        normalized["model"] = model
        normalized["source_file"] = str(path)

        normalized = normalized.dropna(
            subset=["roc_auc"]
        )

        normalized["fold_key"] = (
            normalized["repeat"].astype(str)
            + "_"
            + normalized["fold"].astype(str)
        )

        if len(normalized) < 3:
            return None, "Fewer than three fold results."

        return normalized, "Accepted"

    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


fold_audit_rows: list[dict[str, object]] = []
fold_tables: list[pd.DataFrame] = []

for path in discover_fold_result_files():
    normalized, reason = normalize_fold_result_file(path)

    fold_audit_rows.append(
        {
            "source_file": str(path),
            "accepted": normalized is not None,
            "reason": reason,
            "model": (
                normalized["model"].iloc[0]
                if normalized is not None
                else infer_model_name(path)
            ),
            "n_rows": (
                len(normalized)
                if normalized is not None
                else None
            ),
        }
    )

    if normalized is not None:
        fold_tables.append(normalized)

fold_audit_df = pd.DataFrame(fold_audit_rows)
fold_audit_df.to_csv(
    OUT_DIR / "fold_result_discovery_audit.csv",
    index=False,
)

if fold_tables:
    fold_values_df = pd.concat(
        fold_tables,
        ignore_index=True,
    )

    # Remove exact duplicate model/fold/value records.
    fold_values_df = fold_values_df.drop_duplicates(
        subset=[
            "model",
            "fold_key",
            "roc_auc",
        ]
    )
else:
    fold_values_df = pd.DataFrame(
        columns=[
            "roc_auc",
            "fold",
            "repeat",
            "model",
            "source_file",
            "fold_key",
        ]
    )

fold_values_df.to_csv(
    OUT_DIR / "selected_fold_level_roc_auc_values.csv",
    index=False,
)


def rank_biserial_from_wilcoxon(
    differences: np.ndarray,
) -> float:
    nonzero = differences[differences != 0]

    if len(nonzero) == 0:
        return 0.0

    ranks = pd.Series(
        np.abs(nonzero)
    ).rank(method="average").to_numpy()

    positive_rank_sum = ranks[nonzero > 0].sum()
    negative_rank_sum = ranks[nonzero < 0].sum()
    total_rank_sum = ranks.sum()

    if total_rank_sum == 0:
        return 0.0

    return float(
        (positive_rank_sum - negative_rank_sum)
        / total_rank_sum
    )


fold_test_rows: list[dict[str, object]] = []

if not fold_values_df.empty:
    # Aggregate duplicate measurements per model/fold.
    fold_aggregated = (
        fold_values_df.groupby(
            ["model", "fold_key"],
            as_index=False,
        )["roc_auc"]
        .mean()
    )

    model_names = sorted(
        fold_aggregated["model"].unique()
    )

    for model_a, model_b in itertools.combinations(
        model_names,
        2,
    ):
        a = fold_aggregated[
            fold_aggregated["model"] == model_a
        ][["fold_key", "roc_auc"]].rename(
            columns={"roc_auc": "roc_auc_a"}
        )

        b = fold_aggregated[
            fold_aggregated["model"] == model_b
        ][["fold_key", "roc_auc"]].rename(
            columns={"roc_auc": "roc_auc_b"}
        )

        paired = a.merge(
            b,
            on="fold_key",
            how="inner",
        )

        if len(paired) < 3:
            continue

        x = paired["roc_auc_a"].to_numpy(dtype=float)
        y = paired["roc_auc_b"].to_numpy(dtype=float)
        differences = x - y

        try:
            statistic, p_value = wilcoxon(
                differences,
                zero_method="wilcox",
                alternative="two-sided",
                mode="auto",
            )
        except Exception:
            statistic = np.nan
            p_value = np.nan

        difference_sd = np.std(differences, ddof=1)

        if difference_sd > 0:
            paired_cohens_d = (
                float(np.mean(differences) / difference_sd)
            )
        else:
            paired_cohens_d = np.nan

        rank_biserial = rank_biserial_from_wilcoxon(
            differences
        )

        fold_test_rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "n_matched_folds": len(paired),
                "mean_roc_auc_a": float(np.mean(x)),
                "mean_roc_auc_b": float(np.mean(y)),
                "mean_difference_a_minus_b": float(
                    np.mean(differences)
                ),
                "median_difference_a_minus_b": float(
                    np.median(differences)
                ),
                "wilcoxon_statistic": (
                    float(statistic)
                    if np.isfinite(statistic)
                    else np.nan
                ),
                "wilcoxon_p_value": (
                    float(p_value)
                    if np.isfinite(p_value)
                    else np.nan
                ),
                "paired_cohens_d": paired_cohens_d,
                "rank_biserial_correlation": rank_biserial,
                "statistically_significant_0_05": (
                    p_value < ALPHA
                    if np.isfinite(p_value)
                    else False
                ),
            }
        )

fold_tests_df = pd.DataFrame(fold_test_rows)
fold_tests_df.to_csv(
    OUT_DIR / "paired_fold_wilcoxon_effect_sizes.csv",
    index=False,
)


# ============================================================
# 13. Manuscript-ready summary
# ============================================================

summary_rows: list[dict[str, object]] = []

for _, row in metrics_df.iterrows():
    summary_rows.append(
        {
            "analysis_type": "Bootstrap confidence interval",
            "comparison": (
                f"{row['model']} ({row['evaluation_scope']})"
            ),
            "estimate": row["roc_auc"],
            "ci_or_test": (
                f"95% CI "
                f"[{row['roc_auc_ci95_lower']:.4f}, "
                f"{row['roc_auc_ci95_upper']:.4f}]"
            ),
            "p_value": np.nan,
            "interpretation": (
                "Participant-level bootstrap confidence interval "
                "for ROC-AUC."
            ),
        }
    )

for _, row in pairwise_df.iterrows():
    summary_rows.append(
        {
            "analysis_type": "DeLong test",
            "comparison": (
                f"{row['model_a']} vs {row['model_b']} "
                f"({row['evaluation_scope']})"
            ),
            "estimate": row[
                "auc_difference_a_minus_b"
            ],
            "ci_or_test": (
                f"z={row['delong_z']:.4f}"
                if np.isfinite(row["delong_z"])
                else "z=NA"
            ),
            "p_value": row["delong_p_value"],
            "interpretation": (
                "Statistically significant ROC-AUC difference."
                if row[
                    "statistically_significant_delong_0_05"
                ]
                else "No statistically significant ROC-AUC difference."
            ),
        }
    )

    summary_rows.append(
        {
            "analysis_type": "Paired bootstrap AUC difference",
            "comparison": (
                f"{row['model_a']} vs {row['model_b']} "
                f"({row['evaluation_scope']})"
            ),
            "estimate": row[
                "bootstrap_mean_auc_difference"
            ],
            "ci_or_test": (
                f"95% CI "
                f"[{row['bootstrap_ci95_lower']:.4f}, "
                f"{row['bootstrap_ci95_upper']:.4f}]"
            ),
            "p_value": row[
                "bootstrap_two_sided_p"
            ],
            "interpretation": (
                "Bootstrap evidence of an ROC-AUC difference."
                if row[
                    "statistically_significant_bootstrap_0_05"
                ]
                else "Bootstrap interval did not support a reliable ROC-AUC difference."
            ),
        }
    )

    summary_rows.append(
        {
            "analysis_type": "Exact McNemar test",
            "comparison": (
                f"{row['model_a']} vs {row['model_b']} "
                f"({row['evaluation_scope']})"
            ),
            "estimate": row[
                "mcnemar_discordant_pairs"
            ],
            "ci_or_test": (
                f"discordant={int(row['mcnemar_discordant_pairs'])}"
            ),
            "p_value": row[
                "mcnemar_exact_p_value"
            ],
            "interpretation": (
                "Significant difference in paired classification errors."
                if row[
                    "statistically_significant_mcnemar_0_05"
                ]
                else "No significant difference in paired classification errors."
            ),
        }
    )

for _, row in fold_tests_df.iterrows():
    summary_rows.append(
        {
            "analysis_type": "Fold-level Wilcoxon test",
            "comparison": (
                f"{row['model_a']} vs {row['model_b']}"
            ),
            "estimate": row[
                "mean_difference_a_minus_b"
            ],
            "ci_or_test": (
                f"paired d={row['paired_cohens_d']:.4f}; "
                f"rank-biserial="
                f"{row['rank_biserial_correlation']:.4f}"
            ),
            "p_value": row["wilcoxon_p_value"],
            "interpretation": (
                "Significant paired fold-level difference."
                if row[
                    "statistically_significant_0_05"
                ]
                else "No significant paired fold-level difference."
            ),
        }
    )

manuscript_summary_df = pd.DataFrame(summary_rows)
manuscript_summary_df.to_csv(
    OUT_DIR / "manuscript_ready_statistical_summary.csv",
    index=False,
)


# ============================================================
# 14. Excel workbook and metadata
# ============================================================

excel_path = (
    OUT_DIR
    / "Stage12_Statistical_Significance_Report.xlsx"
)

with pd.ExcelWriter(
    excel_path,
    engine="openpyxl",
) as writer:
    discovery_df.to_excel(
        writer,
        sheet_name="Discovery_Audit",
        index=False,
    )

    selection_audit.to_excel(
        writer,
        sheet_name="Selection_Audit",
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

    fold_audit_df.to_excel(
        writer,
        sheet_name="Fold_Audit",
        index=False,
    )

    fold_values_df.to_excel(
        writer,
        sheet_name="Fold_Values",
        index=False,
    )

    fold_tests_df.to_excel(
        writer,
        sheet_name="Fold_Wilcoxon",
        index=False,
    )

    manuscript_summary_df.to_excel(
        writer,
        sheet_name="Manuscript_Summary",
        index=False,
    )

elapsed_seconds = time.perf_counter() - start_time

run_metadata = {
    "output_root": str(OUTPUT_ROOT),
    "output_directory": str(OUT_DIR),
    "random_seed": RANDOM_SEED,
    "bootstrap_iterations": N_BOOTSTRAP,
    "alpha": ALPHA,
    "candidate_prediction_files": len(candidate_paths),
    "accepted_prediction_files_before_deduplication": len(
        accepted_evidence
    ),
    "selected_final_prediction_evidence": len(
        selected_evidence
    ),
    "valid_pairwise_comparisons": len(pairwise_df),
    "fold_level_comparisons": len(fold_tests_df),
    "elapsed_seconds": elapsed_seconds,
    "elapsed_minutes": elapsed_seconds / 60,
}

with open(
    OUT_DIR / "stage12_run_metadata.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        run_metadata,
        file,
        indent=2,
    )


# ============================================================
# 15. Console summary
# ============================================================

print("=" * 88)
print("Stage 12 Statistical Significance Analysis Completed")
print("=" * 88)
print(f"Candidate prediction files: {len(candidate_paths)}")
print(
    "Accepted before deduplication: "
    f"{len(accepted_evidence)}"
)
print(
    "Selected final prediction evidence: "
    f"{len(selected_evidence)}"
)
print(
    "Valid matched pairwise comparisons: "
    f"{len(pairwise_df)}"
)
print(
    "Valid paired fold comparisons: "
    f"{len(fold_tests_df)}"
)
print(f"Bootstrap iterations: {N_BOOTSTRAP:,}")
print(f"Elapsed time: {elapsed_seconds / 60:.2f} minutes")
print()
print(f"Output directory:\n{OUT_DIR}")
print()
print("Generated files:")
print("  1. prediction_discovery_audit.csv")
print("  2. selected_prediction_evidence_audit.csv")
print("  3. final_model_bootstrap_confidence_intervals.csv")
print("  4. valid_pairwise_statistical_tests.csv")
print("  5. fold_result_discovery_audit.csv")
print("  6. selected_fold_level_roc_auc_values.csv")
print("  7. paired_fold_wilcoxon_effect_sizes.csv")
print("  8. manuscript_ready_statistical_summary.csv")
print("  9. Stage12_Statistical_Significance_Report.xlsx")
print(" 10. stage12_run_metadata.json")
print("=" * 88)