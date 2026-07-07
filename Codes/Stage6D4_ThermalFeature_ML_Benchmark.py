from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

GLOBAL_FEATURE_FILE = Path(
    r"D:\47\472\New-Papers\Anemia_Paper\Outputs\Stage5A_Global_Feature_Extraction\tables\participant_global_features.csv"
)

PATCH_FEATURE_FILE = Path(
    r"D:\47\472\New-Papers\Anemia_Paper\Outputs\Stage5B_Local_Patch_Feature_Extraction\tables\participant_patch_features.csv"
)

TOP_THERMAL_FEATURE_FILE = Path(
    r"D:\47\472\New-Papers\Anemia_Paper\Outputs\Stage6C_Global_vs_Local_Feature_Integration_and_Filtering\tables\top50_thermal_features.csv"
)

OUTPUT_DIR = Path(
    r"D:\47\472\New-Papers\Anemia_Paper\Outputs\Stage6D4_ThermalFeature_ML_Benchmark"
)

TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"

for d in [OUTPUT_DIR, TABLES_DIR, FIGURES_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIG
# ============================================================

RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 10
FEATURE_SET_NAME = "Top50_Thermal_Features"


# ============================================================
# HELPERS
# ============================================================

def safe_auc(y_true, y_score):
    try:
        return roc_auc_score(y_true, y_score)
    except Exception:
        return np.nan


def get_feature_list():
    top_df = pd.read_csv(TOP_THERMAL_FEATURE_FILE)

    if "feature" not in top_df.columns:
        raise ValueError("top50_thermal_features.csv must contain a 'feature' column.")

    return top_df["feature"].dropna().astype(str).tolist()


def evaluate_predictions(y_true, y_pred, y_score):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": safe_auc(y_true, y_score),
    }


def model_scores(clf, X_test):
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X_test)[:, 1]

    if hasattr(clf, "decision_function"):
        scores = clf.decision_function(X_test)
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)

    return clf.predict(X_test)


def plot_bar(summary_df, metric, output_file, title):
    plt.figure(figsize=(8, 5))
    plt.bar(summary_df["model"], summary_df[metric])
    plt.ylabel(metric)
    plt.xlabel("Model")
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def plot_confusion(cm, labels, output_file, title):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title(title)
    plt.colorbar()
    plt.xticks([0, 1], labels)
    plt.yticks([0, 1], labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def plot_roc(y_true, y_score, output_file, title):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def merge_global_and_patch_features(global_df, patch_df):
    """
    Merge global and local patch feature matrices using participant_id.
    Metadata columns from the patch file are removed to avoid duplicate conflicts.
    """
    key_cols = ["participant_id", "label", "class_name", "sex", "age"]
    patch_drop_cols = [c for c in key_cols if c in patch_df.columns and c != "participant_id"]

    patch_features_only = patch_df.drop(columns=patch_drop_cols, errors="ignore")

    merged = global_df.merge(
        patch_features_only,
        on="participant_id",
        how="inner"
    )

    return merged


# ============================================================
# LOAD DATA
# ============================================================

if not GLOBAL_FEATURE_FILE.exists():
    raise FileNotFoundError(f"Missing file: {GLOBAL_FEATURE_FILE}")

if not PATCH_FEATURE_FILE.exists():
    raise FileNotFoundError(f"Missing file: {PATCH_FEATURE_FILE}")

if not TOP_THERMAL_FEATURE_FILE.exists():
    raise FileNotFoundError(f"Missing file: {TOP_THERMAL_FEATURE_FILE}")

global_df = pd.read_csv(GLOBAL_FEATURE_FILE)
patch_df = pd.read_csv(PATCH_FEATURE_FILE)

df = merge_global_and_patch_features(global_df, patch_df)

selected_features = get_feature_list()

available_features = [f for f in selected_features if f in df.columns]
missing_features = [f for f in selected_features if f not in df.columns]

if len(available_features) == 0:
    raise ValueError("No selected thermal features were found in the merged global + patch feature matrix.")

X = df[available_features].copy()
y = df["label"].astype(int).values

X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median(numeric_only=True))

feature_status = pd.DataFrame({
    "feature": selected_features,
    "available_in_merged_features": [f in available_features for f in selected_features]
})

feature_status.to_csv(
    TABLES_DIR / "selected_thermal_feature_availability.csv",
    index=False
)

df[
    ["participant_id", "label", "class_name", "sex", "age"] + available_features
].to_csv(
    TABLES_DIR / "thermal_feature_dataset.csv",
    index=False
)


# ============================================================
# MODELS
# ============================================================

models = {
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ]),

    "LinearSVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            kernel="linear",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ]),

    "RBFSVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ]),

    "RandomForest": RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "GradientBoosting": GradientBoostingClassifier(
        random_state=RANDOM_STATE
    )
}


cv = RepeatedStratifiedKFold(
    n_splits=N_SPLITS,
    n_repeats=N_REPEATS,
    random_state=RANDOM_STATE
)


# ============================================================
# REPEATED CV EVALUATION
# ============================================================

summary_rows = []
all_fold_rows = []

for model_name, model in models.items():

    print("=" * 80)
    print(f"Evaluating model: {model_name}")
    print("=" * 80)

    model_fold_rows = []
    all_predictions = []

    for fold_id, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        participant_test = df.iloc[test_idx]["participant_id"].values

        clf = clone(model)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_score = model_scores(clf, X_test)

        metrics = evaluate_predictions(y_test, y_pred, y_score)

        fold_record = {
            "model": model_name,
            "fold_id": fold_id,
            **metrics
        }

        model_fold_rows.append(fold_record)
        all_fold_rows.append(fold_record)

        for pid, yt, yp, ys in zip(participant_test, y_test, y_pred, y_score):
            all_predictions.append({
                "model": model_name,
                "fold_id": fold_id,
                "participant_id": pid,
                "true_label": int(yt),
                "predicted_label": int(yp),
                "predicted_probability_anemia": float(ys)
            })

    fold_df = pd.DataFrame(model_fold_rows)
    pred_df = pd.DataFrame(all_predictions)

    fold_df.to_csv(
        TABLES_DIR / f"fold_results_{model_name}.csv",
        index=False
    )

    pred_df.to_csv(
        TABLES_DIR / f"predictions_{model_name}.csv",
        index=False
    )

    participant_pred = (
        pred_df
        .groupby(["participant_id", "true_label"])
        .agg(
            mean_probability=("predicted_probability_anemia", "mean")
        )
        .reset_index()
    )

    participant_pred["predicted_label"] = (
        participant_pred["mean_probability"] >= 0.5
    ).astype(int)

    overall_metrics = evaluate_predictions(
        participant_pred["true_label"].values,
        participant_pred["predicted_label"].values,
        participant_pred["mean_probability"].values
    )

    summary_rows.append({
        "model": model_name,
        "feature_set": FEATURE_SET_NAME,
        "n_features": len(available_features),
        "cv_folds_total": N_SPLITS * N_REPEATS,
        **overall_metrics,
        "fold_accuracy_mean": fold_df["accuracy"].mean(),
        "fold_accuracy_std": fold_df["accuracy"].std(),
        "fold_f1_mean": fold_df["f1_score"].mean(),
        "fold_f1_std": fold_df["f1_score"].std(),
        "fold_auc_mean": fold_df["roc_auc"].mean(),
        "fold_auc_std": fold_df["roc_auc"].std(),
    })

    participant_pred.to_csv(
        TABLES_DIR / f"participant_level_predictions_{model_name}.csv",
        index=False
    )

    report_dict = classification_report(
        participant_pred["true_label"],
        participant_pred["predicted_label"],
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(report_dict).transpose().to_csv(
        TABLES_DIR / f"classification_report_{model_name}.csv"
    )


fold_results_all = pd.DataFrame(all_fold_rows)

fold_results_all.to_csv(
    TABLES_DIR / "fold_results_all_models.csv",
    index=False
)

performance_summary = pd.DataFrame(summary_rows)

performance_summary = performance_summary.sort_values(
    by=["roc_auc", "f1_score", "accuracy"],
    ascending=False
)

performance_summary.to_csv(
    TABLES_DIR / "model_performance_summary.csv",
    index=False
)


# ============================================================
# BEST MODEL FIGURES
# ============================================================

best_model_name = performance_summary.iloc[0]["model"]

best_pred = pd.read_csv(
    TABLES_DIR / f"participant_level_predictions_{best_model_name}.csv"
)

cm = confusion_matrix(
    best_pred["true_label"],
    best_pred["predicted_label"]
)

plot_confusion(
    cm,
    labels=["Normal", "Anemia"],
    output_file=FIGURES_DIR / "confusion_matrix_best_model.png",
    title=f"Confusion Matrix - {best_model_name}"
)

plot_roc(
    best_pred["true_label"],
    best_pred["mean_probability"],
    output_file=FIGURES_DIR / "roc_curve_best_model.png",
    title=f"ROC Curve - {best_model_name}"
)

plot_bar(
    performance_summary,
    "accuracy",
    FIGURES_DIR / "model_accuracy_comparison.png",
    "Model Accuracy Comparison"
)

plot_bar(
    performance_summary,
    "roc_auc",
    FIGURES_DIR / "model_auc_comparison.png",
    "Model ROC-AUC Comparison"
)

plot_bar(
    performance_summary,
    "f1_score",
    FIGURES_DIR / "model_f1_comparison.png",
    "Model F1-score Comparison"
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

rf.fit(X, y)

rf_importance = pd.DataFrame({
    "feature": available_features,
    "importance": rf.feature_importances_
}).sort_values(by="importance", ascending=False)

rf_importance.to_csv(
    TABLES_DIR / "feature_importance_random_forest.csv",
    index=False
)

gb = GradientBoostingClassifier(random_state=RANDOM_STATE)
gb.fit(X, y)

gb_importance = pd.DataFrame({
    "feature": available_features,
    "importance": gb.feature_importances_
}).sort_values(by="importance", ascending=False)

gb_importance.to_csv(
    TABLES_DIR / "feature_importance_gradient_boosting.csv",
    index=False
)

top20 = rf_importance.head(20)

plt.figure(figsize=(8, 7))
plt.barh(top20["feature"][::-1], top20["importance"][::-1])
plt.xlabel("Importance")
plt.title("Top 20 Random Forest Feature Importances")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "top20_feature_importance_random_forest.png", dpi=300)
plt.close()


# ============================================================
# REPORT
# ============================================================

best_row = performance_summary.iloc[0]

report = f"""# Stage 6D4 Thermal Feature ML Benchmark Report

Stage 6D4 evaluated classical machine-learning models using the top 50 thermal features selected from the integrated Stage 6C ranking. These features may originate from either global thermal descriptors or local thermal patch descriptors.

## Inputs

Global feature matrix:

`{GLOBAL_FEATURE_FILE}`

Local patch feature matrix:

`{PATCH_FEATURE_FILE}`

Top thermal feature ranking:

`{TOP_THERMAL_FEATURE_FILE}`

## Dataset

Participants: {len(df)}

Anemia participants: {int(np.sum(y == 1))}

Normal participants: {int(np.sum(y == 0))}

Selected features requested: {len(selected_features)}

Selected thermal features available: {len(available_features)}

Missing selected features: {len(missing_features)}

## Validation

Repeated Stratified K-Fold Cross-Validation was used.

Folds: {N_SPLITS}

Repeats: {N_REPEATS}

Total validation rounds: {N_SPLITS * N_REPEATS}

## Models Evaluated

- Logistic Regression
- Linear SVM
- RBF SVM
- Random Forest
- Gradient Boosting

## Best Model

Best model: {best_row["model"]}

Accuracy: {best_row["accuracy"]:.4f}

Precision: {best_row["precision"]:.4f}

Recall: {best_row["recall"]:.4f}

F1-score: {best_row["f1_score"]:.4f}

ROC-AUC: {best_row["roc_auc"]:.4f}

## Output Tables

- thermal_feature_dataset.csv
- model_performance_summary.csv
- fold_results_all_models.csv
- predictions_[model].csv
- participant_level_predictions_[model].csv
- classification_report_[model].csv
- selected_thermal_feature_availability.csv
- feature_importance_random_forest.csv
- feature_importance_gradient_boosting.csv

## Output Figures

- model_accuracy_comparison.png
- model_auc_comparison.png
- model_f1_comparison.png
- confusion_matrix_best_model.png
- roc_curve_best_model.png
- top20_feature_importance_random_forest.png

## Interpretation

This stage tests whether thermal-derived features alone can support binary anemia classification. The results should be compared directly with the RGB-only benchmark from Stage 6D3. If thermal-only performance is substantially weaker than RGB-only performance, the final model should treat thermal information as auxiliary rather than primary.
"""

with open(
    REPORTS_DIR / "Stage6D4_ThermalFeature_ML_Benchmark_Report.md",
    "w",
    encoding="utf-8"
) as f:
    f.write(report)


# ============================================================
# CONSOLE OUTPUT
# ============================================================

print("=" * 80)
print("STAGE 6D4 THERMAL FEATURE ML BENCHMARK COMPLETED")
print("=" * 80)
print(f"Participants: {len(df)}")
print(f"Features used: {len(available_features)}")
print(f"Best model: {best_row['model']}")
print(f"Accuracy: {best_row['accuracy']:.4f}")
print(f"Precision: {best_row['precision']:.4f}")
print(f"Recall: {best_row['recall']:.4f}")
print(f"F1-score: {best_row['f1_score']:.4f}")
print(f"ROC-AUC: {best_row['roc_auc']:.4f}")
print(f"Results saved to: {OUTPUT_DIR}")
print("=" * 80)