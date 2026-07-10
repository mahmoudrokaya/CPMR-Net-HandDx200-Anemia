# -*- coding: utf-8 -*-
"""
Stage 10V2 - Contrastive Embedding Quality Evaluation

Purpose:
Evaluate whether Stage 10V contrastive-pretrained encoders learned useful
participant-level and anemia-relevant embeddings before supervised CPMR-Net retraining.
"""

from pathlib import Path
import json
from datetime import datetime
import warnings

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix, silhouette_score
)
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

warnings.filterwarnings("ignore")


BASE_DIR = Path(r"D:\47\472\New-Papers\Anemia_Paper")
OUTPUTS_DIR = BASE_DIR / "Outputs"

CONFIG_FILE = OUTPUTS_DIR / "Stage10I4_Training_Configuration_Experiment_Control" / "configs" / "CPMRNet_training_config_v1.json"
ENCODER_MAP_FILE = OUTPUTS_DIR / "Stage10V_SelfSupervised_Contrastive_Pretraining" / "models" / "contrastive_pretrained_encoder_checkpoint_map.json"

STAGE_OUT = OUTPUTS_DIR / "Stage10V2_Contrastive_Embedding_Quality_Evaluation"
TABLES_OUT = STAGE_OUT / "tables"
REPORTS_OUT = STAGE_OUT / "reports"
EMB_OUT = STAGE_OUT / "embeddings"

for p in [TABLES_OUT, REPORTS_OUT, EMB_OUT]:
    p.mkdir(parents=True, exist_ok=True)


BATCH_SIZE = 64
NUM_WORKERS = 0
EMBEDDING_DIM = 128

ENCODER_GROUPS = ["rgb_3ch", "rgb_1ch", "thermal_1ch"]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_encoder_group(modality, representation):
    if modality == "rgb":
        if representation == "rgb_texture":
            return "rgb_1ch"
        return "rgb_3ch"
    return "thermal_1ch"


def read_tensor(path, modality, representation):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if img is None:
        raise RuntimeError(f"Could not read image: {path}")

    group = infer_encoder_group(modality, representation)

    if group == "rgb_3ch":
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    else:
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        img = img[:, :, None]

    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return torch.tensor(img, dtype=torch.float32)


class RepresentationEmbeddingDataset(Dataset):
    def __init__(self, representation_manifest, participant_manifest, encoder_group):
        self.representation_manifest = representation_manifest.copy()
        self.participant_manifest = participant_manifest.copy()
        self.encoder_group = encoder_group

        self.representation_manifest["encoder_group"] = self.representation_manifest.apply(
            lambda r: infer_encoder_group(r["modality"], r["representation"]),
            axis=1
        )

        self.df = self.representation_manifest[
            (self.representation_manifest["status"] == "saved")
            & (self.representation_manifest["encoder_group"] == encoder_group)
        ].copy().reset_index(drop=True)

        self.participant_manifest["participant_id"] = self.participant_manifest["participant_id"].astype(str)

        self.label_lookup = {
            str(r["participant_id"]): int(r["label"])
            for _, r in self.participant_manifest.iterrows()
        }

        self.class_lookup = {
            str(r["participant_id"]): str(r["class_name"])
            for _, r in self.participant_manifest.iterrows()
        }

        self.df["participant_id"] = self.df["participant_id"].astype(str)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = str(row["participant_id"])

        x = read_tensor(row["path"], row["modality"], row["representation"])

        return {
            "participant_id": pid,
            "label": torch.tensor(self.label_lookup[pid], dtype=torch.long),
            "class_name": self.class_lookup[pid],
            "modality": row["modality"],
            "view": row["view"],
            "representation": row["representation"],
            "path": row["path"],
            "x": x
        }


def collate_fn(batch):
    return {
        "participant_id": [b["participant_id"] for b in batch],
        "label": torch.stack([b["label"] for b in batch]),
        "class_name": [b["class_name"] for b in batch],
        "modality": [b["modality"] for b in batch],
        "view": [b["view"] for b in batch],
        "representation": [b["representation"] for b in batch],
        "path": [b["path"] for b in batch],
        "x": torch.stack([b["x"] for b in batch], dim=0)
    }


class LightweightEncoder(nn.Module):
    def __init__(self, in_channels, embedding_dim=128, dropout=0.10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96), nn.ReLU(inplace=True),
            nn.Conv2d(96, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, x):
        return self.projection(self.features(x))


def load_pretrained_encoder(group, checkpoint_map, device):
    in_channels = 3 if group == "rgb_3ch" else 1
    encoder = LightweightEncoder(in_channels, EMBEDDING_DIM).to(device)
    checkpoint = checkpoint_map[group]

    state = torch.load(checkpoint, map_location=device)
    encoder.load_state_dict(state)
    encoder.eval()

    return encoder


def extract_group_embeddings(group, dataset, checkpoint_map, device):
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )

    encoder = load_pretrained_encoder(group, checkpoint_map, device)

    rows = []
    emb_list = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            emb = encoder(x).detach().cpu().numpy()
            emb_list.append(emb)

            for i in range(len(batch["participant_id"])):
                rows.append({
                    "encoder_group": group,
                    "participant_id": batch["participant_id"][i],
                    "label": int(batch["label"][i].item()),
                    "class_name": batch["class_name"][i],
                    "modality": batch["modality"][i],
                    "view": batch["view"][i],
                    "representation": batch["representation"][i],
                    "path": batch["path"][i],
                })

    manifest = pd.DataFrame(rows)
    embeddings = np.vstack(emb_list).astype(np.float32)

    return manifest, embeddings


def participant_mean_embeddings(manifest, embeddings):
    df = manifest.copy()
    emb_cols = [f"e{i}" for i in range(embeddings.shape[1])]
    emb_df = pd.DataFrame(embeddings, columns=emb_cols)
    full = pd.concat([df.reset_index(drop=True), emb_df], axis=1)

    grouped = full.groupby(["participant_id", "label", "class_name"])[emb_cols].mean().reset_index()
    X = grouped[emb_cols].values.astype(np.float32)
    y = grouped["label"].values.astype(int)

    return grouped[["participant_id", "label", "class_name"]], X, y


def compute_distance_quality(manifest, embeddings, max_pairs=50000):
    rng = np.random.default_rng(42)

    n = len(manifest)
    if n < 3:
        return {}

    pair_count = min(max_pairs, n * 20)

    intra_participant = []
    inter_participant = []
    same_class = []
    different_class = []

    pids = manifest["participant_id"].astype(str).values
    labels = manifest["label"].astype(int).values

    for _ in range(pair_count):
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue

        d = float(cosine_distances(embeddings[i:i+1], embeddings[j:j+1])[0, 0])

        if pids[i] == pids[j]:
            intra_participant.append(d)
        else:
            inter_participant.append(d)

        if labels[i] == labels[j]:
            same_class.append(d)
        else:
            different_class.append(d)

    return {
        "mean_intra_participant_cosine_distance": float(np.mean(intra_participant)) if intra_participant else np.nan,
        "mean_inter_participant_cosine_distance": float(np.mean(inter_participant)) if inter_participant else np.nan,
        "participant_distance_margin": (
            float(np.mean(inter_participant) - np.mean(intra_participant))
            if intra_participant and inter_participant else np.nan
        ),
        "mean_same_class_cosine_distance": float(np.mean(same_class)) if same_class else np.nan,
        "mean_different_class_cosine_distance": float(np.mean(different_class)) if different_class else np.nan,
        "class_distance_margin": (
            float(np.mean(different_class) - np.mean(same_class))
            if same_class and different_class else np.nan
        ),
        "sampled_pairs": int(pair_count),
    }


def safe_silhouette(X, y):
    try:
        if len(np.unique(y)) < 2 or len(X) <= len(np.unique(y)):
            return np.nan
        return float(silhouette_score(X, y, metric="cosine"))
    except Exception:
        return np.nan


def metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan,
        "pr_auc": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan,
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
    }


def evaluate_simple_classifiers(split_df, participant_meta, X, y, embedding_name):
    meta = participant_meta.copy()
    meta["participant_id"] = meta["participant_id"].astype(str)

    split_map = split_df[["participant_id", "split"]].copy()
    split_map["participant_id"] = split_map["participant_id"].astype(str)

    meta = meta.merge(split_map, on="participant_id", how="left")

    train_idx = meta["split"] == "train"
    val_idx = meta["split"] == "val"
    test_idx = meta["split"] == "test"

    X_train, y_train = X[train_idx.values], y[train_idx.values]
    X_val, y_val = X[val_idx.values], y[val_idx.values]
    X_test, y_test = X[test_idx.values], y[test_idx.values]

    models = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42))
        ]),
        "LinearSVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="linear", probability=True, class_weight="balanced", random_state=42))
        ]),
        "RBFSVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42))
        ]),
    }

    rows = []

    for model_name, model in models.items():
        model.fit(X_train, y_train)

        val_prob = model.predict_proba(X_val)[:, 1]
        test_prob = model.predict_proba(X_test)[:, 1]

        val_m = metrics(y_val, val_prob)
        test_m = metrics(y_test, test_prob)

        row = {
            "embedding_name": embedding_name,
            "model": model_name,
            "train_n": int(len(y_train)),
            "val_n": int(len(y_val)),
            "test_n": int(len(y_test)),
        }

        for k, v in val_m.items():
            row[f"val_{k}"] = v
        for k, v in test_m.items():
            row[f"test_{k}"] = v

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    config = load_json(CONFIG_FILE)
    checkpoint_map = load_json(ENCODER_MAP_FILE)

    participant_manifest = pd.read_csv(config["paths"]["participant_manifest"])
    representation_manifest = pd.read_csv(config["paths"]["representation_manifest"])
    holdout_split = pd.read_csv(config["paths"]["holdout_split"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_group_manifests = []
    group_embeddings = {}
    group_participant_embeddings = {}
    quality_rows = []
    classifier_results = []

    for group in ENCODER_GROUPS:
        print("=" * 80)
        print(f"Extracting contrastive embeddings: {group}")
        print("=" * 80)

        dataset = RepresentationEmbeddingDataset(
            representation_manifest,
            participant_manifest,
            group
        )

        manifest, embeddings = extract_group_embeddings(
            group,
            dataset,
            checkpoint_map,
            device
        )

        np.save(EMB_OUT / f"{group}_representation_embeddings.npy", embeddings)
        manifest.to_csv(TABLES_OUT / f"{group}_representation_embedding_manifest.csv", index=False)

        participant_meta, participant_X, participant_y = participant_mean_embeddings(manifest, embeddings)

        np.save(EMB_OUT / f"{group}_participant_mean_embeddings.npy", participant_X)
        participant_meta.to_csv(TABLES_OUT / f"{group}_participant_embedding_manifest.csv", index=False)

        dist_quality = compute_distance_quality(manifest, embeddings)
        sil = safe_silhouette(participant_X, participant_y)

        quality = {
            "encoder_group": group,
            "representation_records": int(len(manifest)),
            "participants": int(participant_meta["participant_id"].nunique()),
            "participant_embedding_shape": str(list(participant_X.shape)),
            "class_silhouette_cosine": sil,
        }
        quality.update(dist_quality)

        quality_rows.append(quality)

        clf_df = evaluate_simple_classifiers(
            holdout_split,
            participant_meta,
            participant_X,
            participant_y,
            group
        )
        classifier_results.append(clf_df)

        group_embeddings[group] = embeddings
        group_participant_embeddings[group] = (participant_meta, participant_X, participant_y)
        all_group_manifests.append(manifest)

    # Combined participant embeddings
    base_meta = None
    combined_X_list = []
    combined_y = None

    for group in ENCODER_GROUPS:
        meta, X, y = group_participant_embeddings[group]
        if base_meta is None:
            base_meta = meta.copy()
            combined_y = y.copy()
        combined_X_list.append(X)

    combined_X = np.concatenate(combined_X_list, axis=1)
    np.save(EMB_OUT / "combined_contrastive_participant_embeddings.npy", combined_X)
    base_meta.to_csv(TABLES_OUT / "combined_contrastive_participant_embedding_manifest.csv", index=False)

    combined_quality = {
        "encoder_group": "combined_rgb3_rgb1_thermal",
        "representation_records": np.nan,
        "participants": int(base_meta["participant_id"].nunique()),
        "participant_embedding_shape": str(list(combined_X.shape)),
        "class_silhouette_cosine": safe_silhouette(combined_X, combined_y),
        "mean_intra_participant_cosine_distance": np.nan,
        "mean_inter_participant_cosine_distance": np.nan,
        "participant_distance_margin": np.nan,
        "mean_same_class_cosine_distance": np.nan,
        "mean_different_class_cosine_distance": np.nan,
        "class_distance_margin": np.nan,
        "sampled_pairs": np.nan,
    }

    quality_rows.append(combined_quality)

    combined_clf = evaluate_simple_classifiers(
        holdout_split,
        base_meta,
        combined_X,
        combined_y,
        "combined_rgb3_rgb1_thermal"
    )

    classifier_results.append(combined_clf)

    quality_df = pd.DataFrame(quality_rows)
    classifier_df = pd.concat(classifier_results, ignore_index=True)

    quality_df.to_csv(TABLES_OUT / "contrastive_embedding_quality_summary.csv", index=False)
    classifier_df.to_csv(TABLES_OUT / "contrastive_embedding_classifier_results.csv", index=False)

    best_val = classifier_df.sort_values("val_roc_auc", ascending=False).iloc[0].to_dict()
    best_test = classifier_df.sort_values("test_roc_auc", ascending=False).iloc[0].to_dict()

    summary = {
        "stage": "Stage10V2",
        "title": "Contrastive Embedding Quality Evaluation",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "encoder_groups": ENCODER_GROUPS,
        "best_validation_model": {
            "embedding_name": best_val["embedding_name"],
            "model": best_val["model"],
            "val_roc_auc": best_val["val_roc_auc"],
            "test_roc_auc": best_val["test_roc_auc"],
        },
        "best_test_model": {
            "embedding_name": best_test["embedding_name"],
            "model": best_test["model"],
            "val_roc_auc": best_test["val_roc_auc"],
            "test_roc_auc": best_test["test_roc_auc"],
        },
        "outputs_saved_to": str(STAGE_OUT)
    }

    with open(STAGE_OUT / "Stage10V2_Contrastive_Embedding_Quality_Evaluation_Summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    report = []
    report.append("# Stage 10V2 Contrastive Embedding Quality Evaluation\n")
    report.append(f"Generated at: {summary['created_at']}\n")
    report.append("## Purpose\n")
    report.append(
        "This stage evaluates whether Stage 10V contrastive-pretrained encoders produce useful participant-level "
        "and anemia-relevant embeddings before supervised CPMR-Net retraining.\n"
    )
    report.append("## Best Validation-Selected Classifier\n")
    report.append(f"- Embedding: {summary['best_validation_model']['embedding_name']}")
    report.append(f"- Model: {summary['best_validation_model']['model']}")
    report.append(f"- Validation ROC-AUC: {summary['best_validation_model']['val_roc_auc']:.4f}")
    report.append(f"- Test ROC-AUC: {summary['best_validation_model']['test_roc_auc']:.4f}\n")
    report.append("## Best Test-Ranked Diagnostic Classifier\n")
    report.append(f"- Embedding: {summary['best_test_model']['embedding_name']}")
    report.append(f"- Model: {summary['best_test_model']['model']}")
    report.append(f"- Validation ROC-AUC: {summary['best_test_model']['val_roc_auc']:.4f}")
    report.append(f"- Test ROC-AUC: {summary['best_test_model']['test_roc_auc']:.4f}\n")
    report.append("## Output Files\n")
    report.append("- `contrastive_embedding_quality_summary.csv`")
    report.append("- `contrastive_embedding_classifier_results.csv`")
    report.append("- `*_representation_embeddings.npy`")
    report.append("- `*_participant_mean_embeddings.npy`")
    report.append("- `combined_contrastive_participant_embeddings.npy`")
    report.append("- `Stage10V2_Contrastive_Embedding_Quality_Evaluation_Summary.json`")

    with open(REPORTS_OUT / "Stage10V2_Contrastive_Embedding_Quality_Evaluation_Report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("=" * 80)
    print("STAGE 10V2 CONTRASTIVE EMBEDDING QUALITY EVALUATION COMPLETED")
    print("=" * 80)
    print(f"Best validation-selected model: {summary['best_validation_model']}")
    print(f"Best test-ranked diagnostic model: {summary['best_test_model']}")
    print(f"Outputs saved to: {STAGE_OUT}")
    print("=" * 80)


if __name__ == "__main__":
    main()