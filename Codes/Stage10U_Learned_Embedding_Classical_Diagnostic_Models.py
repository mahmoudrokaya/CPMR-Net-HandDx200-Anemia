# -*- coding: utf-8 -*-
"""
Stage 10U - Learned Embedding Classical Diagnostic Models

Purpose:
Extract learned embeddings from the best Stage 10L CPMR-Net checkpoint and train
classical low-variance classifiers on those embeddings.

Goal:
Determine whether the weak performance is due to:
1) poor learned representations, or
2) neural classification/fusion instability.

Inputs:
- Stage 10I4 config
- Stage 10L best CPMR-Net checkpoint
- Stage 10I1 holdout split

Outputs:
- Extracted train/val/test embeddings
- Logistic Regression, Linear SVM, RBF SVM, Random Forest, Gradient Boosting results
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

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, balanced_accuracy_score, matthews_corrcoef, confusion_matrix

warnings.filterwarnings("ignore")


BASE_DIR = Path(r"D:\47\472\New-Papers\Anemia_Paper")
OUTPUTS_DIR = BASE_DIR / "Outputs"

CONFIG_FILE = OUTPUTS_DIR / "Stage10I4_Training_Configuration_Experiment_Control" / "configs" / "CPMRNet_training_config_v1.json"
BEST_MODEL_FILE = OUTPUTS_DIR / "Stage10L_CPMRNet_Training_Engine" / "models" / "CPMRNet_best_val_auc.pt"

STAGE_OUT = OUTPUTS_DIR / "Stage10U_Learned_Embedding_Classical_Diagnostic_Models"
TABLES_OUT = STAGE_OUT / "tables"
REPORTS_OUT = STAGE_OUT / "reports"
EMB_OUT = STAGE_OUT / "embeddings"

for p in [TABLES_OUT, REPORTS_OUT, EMB_OUT]:
    p.mkdir(parents=True, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_image_as_tensor(path, modality, representation):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")

    if modality == "rgb" and representation != "rgb_texture":
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img[:, :, None]

    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return torch.tensor(img, dtype=torch.float32)


class CPMRParticipantDataset(Dataset):
    def __init__(self, participant_manifest, representation_manifest, split_df, split_name, config):
        self.participant_manifest = participant_manifest.copy()
        self.representation_manifest = representation_manifest.copy()
        self.split_df = split_df[split_df["split"] == split_name].copy()
        self.config = config

        self.modalities = config["data"]["modalities"]
        self.views = config["data"]["views"]
        self.rgb_reps = config["data"]["rgb_representations"]
        self.thermal_reps = config["data"]["thermal_representations"]

        self.participant_ids = self.split_df["participant_id"].astype(str).tolist()
        self.participant_manifest["participant_id"] = self.participant_manifest["participant_id"].astype(str)
        self.representation_manifest["participant_id"] = self.representation_manifest["participant_id"].astype(str)

        self.participant_lookup = {
            str(r["participant_id"]): r for _, r in self.participant_manifest.iterrows()
        }

        self.rep_groups = {
            pid: g.copy() for pid, g in self.representation_manifest.groupby("participant_id")
        }

    def __len__(self):
        return len(self.participant_ids)

    def __getitem__(self, idx):
        pid = self.participant_ids[idx]
        p_row = self.participant_lookup[pid]
        rep_df = self.rep_groups[pid]

        sample = {
            "participant_id": pid,
            "label": torch.tensor(int(p_row["label"]), dtype=torch.float32),
            "representations": {"rgb": {}, "thermal": {}},
        }

        for modality in self.modalities:
            reps = self.rgb_reps if modality == "rgb" else self.thermal_reps

            for view in self.views:
                sample["representations"][modality][view] = {}
                view_df = rep_df[
                    (rep_df["modality"] == modality)
                    & (rep_df["view"] == view)
                    & (rep_df["status"] == "saved")
                ]

                for rep in reps:
                    row = view_df[view_df["representation"] == rep]
                    if len(row) == 0:
                        raise RuntimeError(f"Missing representation: {pid} {modality} {view} {rep}")
                    sample["representations"][modality][view][rep] = read_image_as_tensor(
                        row.iloc[0]["path"], modality, rep
                    )

        return sample


def collate_fn(batch):
    first = batch[0]
    output = {
        "participant_id": [b["participant_id"] for b in batch],
        "label": torch.stack([b["label"] for b in batch]),
        "representations": {"rgb": {}, "thermal": {}},
    }

    for modality in ["rgb", "thermal"]:
        for view in first["representations"][modality]:
            output["representations"][modality][view] = {}
            for rep in first["representations"][modality][view]:
                output["representations"][modality][view][rep] = torch.stack(
                    [b["representations"][modality][view][rep] for b in batch],
                    dim=0
                )

    return output


def move_batch_to_device(batch, device):
    batch["label"] = batch["label"].to(device)
    for modality in ["rgb", "thermal"]:
        for view in batch["representations"][modality]:
            for rep in batch["representations"][modality][view]:
                batch["representations"][modality][view][rep] = batch["representations"][modality][view][rep].to(device)
    return batch


class LightweightEncoder(nn.Module):
    def __init__(self, in_channels, embedding_dim=128, dropout=0.25):
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


class AttentionAggregator(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.Tanh(),
            nn.Linear(embedding_dim // 2, 1),
        )

    def forward(self, x):
        logits = self.score(x).squeeze(-1)
        weights = torch.softmax(logits, dim=1)
        return torch.sum(x * weights.unsqueeze(-1), dim=1), weights


class AdaptiveRGBThermalFusion(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, 2),
        )

    def forward(self, rgb_vec, thermal_vec):
        weights = torch.softmax(self.gate(torch.cat([rgb_vec, thermal_vec], dim=-1)), dim=-1)
        fused = weights[:, 0:1] * rgb_vec + weights[:, 1:2] * thermal_vec
        return fused, weights


class CPMRNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding_dim = int(config["model"]["embedding_dim"])
        dropout = float(config["model"]["dropout"])

        self.views = config["data"]["views"]
        self.rgb_reps = config["data"]["rgb_representations"]
        self.thermal_reps = config["data"]["thermal_representations"]

        self.rgb_3ch_encoder = LightweightEncoder(3, self.embedding_dim, dropout)
        self.rgb_1ch_encoder = LightweightEncoder(1, self.embedding_dim, dropout)
        self.thermal_encoder = LightweightEncoder(1, self.embedding_dim, dropout)

        self.representation_attention = AttentionAggregator(self.embedding_dim)
        self.rgb_view_attention = AttentionAggregator(self.embedding_dim)
        self.thermal_view_attention = AttentionAggregator(self.embedding_dim)
        self.fusion = AdaptiveRGBThermalFusion(self.embedding_dim)

        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.embedding_dim, 1),
        )

    def encode_representation(self, modality, representation, tensor):
        if modality == "rgb":
            if representation == "rgb_texture":
                return self.rgb_1ch_encoder(tensor)
            return self.rgb_3ch_encoder(tensor)
        return self.thermal_encoder(tensor)

    def encode_specialist(self, batch, modality, view):
        reps = self.rgb_reps if modality == "rgb" else self.thermal_reps
        embeddings = []
        for rep in reps:
            embeddings.append(
                self.encode_representation(
                    modality,
                    rep,
                    batch["representations"][modality][view][rep]
                )
            )
        return self.representation_attention(torch.stack(embeddings, dim=1))

    def forward(self, batch):
        rgb_specialists = []
        thermal_specialists = []

        for view in self.views:
            rgb_spec, _ = self.encode_specialist(batch, "rgb", view)
            th_spec, _ = self.encode_specialist(batch, "thermal", view)
            rgb_specialists.append(rgb_spec)
            thermal_specialists.append(th_spec)

        rgb_branch, _ = self.rgb_view_attention(torch.stack(rgb_specialists, dim=1))
        thermal_branch, _ = self.thermal_view_attention(torch.stack(thermal_specialists, dim=1))
        fused, modality_weights = self.fusion(rgb_branch, thermal_branch)

        logits = self.classifier(fused).squeeze(-1)
        probs = torch.sigmoid(logits)

        return {
            "logits": logits,
            "probabilities": probs,
            "fused_embedding": fused,
            "rgb_branch_embedding": rgb_branch,
            "thermal_branch_embedding": thermal_branch,
            "modality_weights": modality_weights,
        }


def extract_embeddings(model, loader, device, split_name):
    model.eval()
    rows = []
    fused_list = []
    rgb_list = []
    thermal_list = []

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            out = model(batch)

            fused = out["fused_embedding"].detach().cpu().numpy()
            rgb = out["rgb_branch_embedding"].detach().cpu().numpy()
            thermal = out["thermal_branch_embedding"].detach().cpu().numpy()
            probs = out["probabilities"].detach().cpu().numpy()
            labels = batch["label"].detach().cpu().numpy()

            for i, pid in enumerate(batch["participant_id"]):
                rows.append({
                    "participant_id": pid,
                    "split": split_name,
                    "label": int(labels[i]),
                    "deep_probability": float(probs[i])
                })

            fused_list.append(fused)
            rgb_list.append(rgb)
            thermal_list.append(thermal)

    manifest = pd.DataFrame(rows)
    return (
        manifest,
        np.vstack(fused_list).astype(np.float32),
        np.vstack(rgb_list).astype(np.float32),
        np.vstack(thermal_list).astype(np.float32),
    )


def metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

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
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def get_models():
    return {
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
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
    }


def evaluate_embedding_set(name, X_train, y_train, X_val, y_val, X_test, y_test):
    rows = []
    pred_rows = []

    for model_name, model in get_models().items():
        model.fit(X_train, y_train)

        val_prob = model.predict_proba(X_val)[:, 1]
        test_prob = model.predict_proba(X_test)[:, 1]

        val_m = metrics(y_val, val_prob)
        test_m = metrics(y_test, test_prob)

        row = {
            "embedding_set": name,
            "model": model_name,
        }

        for k, v in val_m.items():
            row[f"val_{k}"] = v

        for k, v in test_m.items():
            row[f"test_{k}"] = v

        rows.append(row)

        for split, y_true, probs in [
            ("val", y_val, val_prob),
            ("test", y_test, test_prob)
        ]:
            for i in range(len(y_true)):
                pred_rows.append({
                    "embedding_set": name,
                    "model": model_name,
                    "split": split,
                    "label": int(y_true[i]),
                    "probability": float(probs[i]),
                    "prediction_0_5": int(probs[i] >= 0.5),
                })

    return pd.DataFrame(rows), pd.DataFrame(pred_rows)


def main():
    config = load_json(CONFIG_FILE)

    participant_manifest = pd.read_csv(config["paths"]["participant_manifest"])
    representation_manifest = pd.read_csv(config["paths"]["representation_manifest"])
    holdout_split = pd.read_csv(config["paths"]["holdout_split"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = CPMRParticipantDataset(participant_manifest, representation_manifest, holdout_split, "train", config)
    val_ds = CPMRParticipantDataset(participant_manifest, representation_manifest, holdout_split, "val", config)
    test_ds = CPMRParticipantDataset(participant_manifest, representation_manifest, holdout_split, "test", config)

    train_loader = DataLoader(train_ds, batch_size=int(config["data"]["batch_size"]), shuffle=False, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=int(config["data"]["batch_size"]), shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=int(config["data"]["batch_size"]), shuffle=False, num_workers=0, collate_fn=collate_fn)

    model = CPMRNet(config).to(device)
    model.load_state_dict(torch.load(BEST_MODEL_FILE, map_location=device))

    train_manifest, train_fused, train_rgb, train_thermal = extract_embeddings(model, train_loader, device, "train")
    val_manifest, val_fused, val_rgb, val_thermal = extract_embeddings(model, val_loader, device, "val")
    test_manifest, test_fused, test_rgb, test_thermal = extract_embeddings(model, test_loader, device, "test")

    manifest = pd.concat([train_manifest, val_manifest, test_manifest], ignore_index=True)
    manifest.to_csv(TABLES_OUT / "learned_embedding_manifest.csv", index=False)

    np.save(EMB_OUT / "train_fused_embeddings.npy", train_fused)
    np.save(EMB_OUT / "val_fused_embeddings.npy", val_fused)
    np.save(EMB_OUT / "test_fused_embeddings.npy", test_fused)

    np.save(EMB_OUT / "train_rgb_embeddings.npy", train_rgb)
    np.save(EMB_OUT / "val_rgb_embeddings.npy", val_rgb)
    np.save(EMB_OUT / "test_rgb_embeddings.npy", test_rgb)

    np.save(EMB_OUT / "train_thermal_embeddings.npy", train_thermal)
    np.save(EMB_OUT / "val_thermal_embeddings.npy", val_thermal)
    np.save(EMB_OUT / "test_thermal_embeddings.npy", test_thermal)

    y_train = train_manifest["label"].values
    y_val = val_manifest["label"].values
    y_test = test_manifest["label"].values

    embedding_sets = {
        "fused": (train_fused, val_fused, test_fused),
        "rgb_branch": (train_rgb, val_rgb, test_rgb),
        "thermal_branch": (train_thermal, val_thermal, test_thermal),
        "rgb_plus_thermal_concat": (
            np.concatenate([train_rgb, train_thermal], axis=1),
            np.concatenate([val_rgb, val_thermal], axis=1),
            np.concatenate([test_rgb, test_thermal], axis=1),
        ),
    }

    result_tables = []
    prediction_tables = []

    for emb_name, (X_train, X_val, X_test) in embedding_sets.items():
        results, preds = evaluate_embedding_set(
            emb_name,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test
        )
        result_tables.append(results)
        prediction_tables.append(preds)

    results_df = pd.concat(result_tables, ignore_index=True)
    preds_df = pd.concat(prediction_tables, ignore_index=True)

    results_df.to_csv(TABLES_OUT / "learned_embedding_classical_model_results.csv", index=False)
    preds_df.to_csv(TABLES_OUT / "learned_embedding_classical_model_predictions.csv", index=False)

    best_val = results_df.sort_values("val_roc_auc", ascending=False).iloc[0].to_dict()
    best_test = results_df.sort_values("test_roc_auc", ascending=False).iloc[0].to_dict()

    summary = {
        "stage": "Stage10U",
        "title": "Learned Embedding Classical Diagnostic Models",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_checkpoint": str(BEST_MODEL_FILE),
        "train_participants": int(len(train_manifest)),
        "val_participants": int(len(val_manifest)),
        "test_participants": int(len(test_manifest)),
        "embedding_sets": list(embedding_sets.keys()),
        "models": list(get_models().keys()),
        "best_validation_model": {
            "embedding_set": best_val["embedding_set"],
            "model": best_val["model"],
            "val_roc_auc": best_val["val_roc_auc"],
            "test_roc_auc": best_val["test_roc_auc"],
        },
        "best_test_model": {
            "embedding_set": best_test["embedding_set"],
            "model": best_test["model"],
            "val_roc_auc": best_test["val_roc_auc"],
            "test_roc_auc": best_test["test_roc_auc"],
        },
        "outputs_saved_to": str(STAGE_OUT),
    }

    with open(STAGE_OUT / "Stage10U_Learned_Embedding_Classical_Diagnostic_Models_Summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    report = []
    report.append("# Stage 10U Learned Embedding Classical Diagnostic Models\n")
    report.append(f"Generated at: {summary['created_at']}\n")
    report.append("## Purpose\n")
    report.append(
        "This stage extracts learned embeddings from the best Stage 10L CPMR-Net checkpoint and evaluates "
        "whether low-variance classical classifiers can exploit those embeddings better than the neural classifier.\n"
    )
    report.append("## Best Validation-Selected Model\n")
    report.append(f"- Embedding set: {summary['best_validation_model']['embedding_set']}")
    report.append(f"- Classifier: {summary['best_validation_model']['model']}")
    report.append(f"- Validation ROC-AUC: {summary['best_validation_model']['val_roc_auc']:.4f}")
    report.append(f"- Test ROC-AUC: {summary['best_validation_model']['test_roc_auc']:.4f}\n")
    report.append("## Best Test-Ranked Diagnostic Model\n")
    report.append(f"- Embedding set: {summary['best_test_model']['embedding_set']}")
    report.append(f"- Classifier: {summary['best_test_model']['model']}")
    report.append(f"- Validation ROC-AUC: {summary['best_test_model']['val_roc_auc']:.4f}")
    report.append(f"- Test ROC-AUC: {summary['best_test_model']['test_roc_auc']:.4f}\n")
    report.append("## Output Files\n")
    report.append("- `tables/learned_embedding_manifest.csv`")
    report.append("- `tables/learned_embedding_classical_model_results.csv`")
    report.append("- `tables/learned_embedding_classical_model_predictions.csv`")
    report.append("- `embeddings/*.npy`")
    report.append("- `Stage10U_Learned_Embedding_Classical_Diagnostic_Models_Summary.json`")

    with open(REPORTS_OUT / "Stage10U_Learned_Embedding_Classical_Diagnostic_Models_Report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("=" * 80)
    print("STAGE 10U LEARNED EMBEDDING CLASSICAL DIAGNOSTIC MODELS COMPLETED")
    print("=" * 80)
    print(f"Best validation-selected model: {summary['best_validation_model']}")
    print(f"Best test-ranked diagnostic model: {summary['best_test_model']}")
    print(f"Outputs saved to: {STAGE_OUT}")
    print("=" * 80)


if __name__ == "__main__":
    main()