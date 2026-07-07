# -*- coding: utf-8 -*-
"""
Stage 10B - Multi-Representation Generator

Purpose:
Generate core CPMR-Net image representations from the Stage 10A participant-level manifest.

Generated representations:
RGB:
- original resized RGB
- HSV
- LAB
- texture-enhanced grayscale
- local anatomical patches

Thermal:
- normalized thermal map
- texture-enhanced thermal
- local thermal patches

No model training is performed in this stage.
"""

from pathlib import Path
import json
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm

BASE_DIR = Path(r"D:\47\472\New-Papers\Anemia_Paper")
OUTPUTS_DIR = BASE_DIR / "Outputs"

STAGE10A_DIR = OUTPUTS_DIR / "Stage10A_ParticipantLevel_Dataset_Loader"
INPUT_MANIFEST = STAGE10A_DIR / "tables" / "valid_participant_level_manifest.csv"

STAGE_OUT = OUTPUTS_DIR / "Stage10B_MultiRepresentation_Generator"
TABLES_OUT = STAGE_OUT / "tables"
REPORTS_OUT = STAGE_OUT / "reports"
REPR_OUT = STAGE_OUT / "representations"

TABLES_OUT.mkdir(parents=True, exist_ok=True)
REPORTS_OUT.mkdir(parents=True, exist_ok=True)
REPR_OUT.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224
PATCH_SIZE = 112

RGB_COLUMNS = [
    "rgb_l_dorsal",
    "rgb_l_palmar",
    "rgb_r_dorsal",
    "rgb_r_palmar",
]

THERMAL_COLUMNS = [
    "thermal_l_dorsal",
    "thermal_l_palmar",
    "thermal_r_dorsal",
    "thermal_r_palmar",
]


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def read_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return img


def read_thermal(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return img


def normalize_uint8(img):
    img = img.astype(np.float32)
    min_val = float(np.min(img))
    max_val = float(np.max(img))
    if max_val - min_val < 1e-8:
        return np.zeros_like(img, dtype=np.uint8)
    norm = (img - min_val) / (max_val - min_val)
    return (norm * 255).astype(np.uint8)


def texture_enhance_gray(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = normalize_uint8(img)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    edges = cv2.Laplacian(enhanced, cv2.CV_64F)
    edges = normalize_uint8(np.abs(edges))
    combined = cv2.addWeighted(enhanced, 0.75, edges, 0.25, 0)
    return combined


def extract_patches(img):
    h, w = img.shape[:2]
    ps = PATCH_SIZE

    coords = {
        "patch_center": (w // 2 - ps // 2, h // 2 - ps // 2),
        "patch_upper": (w // 2 - ps // 2, h // 4 - ps // 2),
        "patch_lower": (w // 2 - ps // 2, 3 * h // 4 - ps // 2),
        "patch_left": (w // 4 - ps // 2, h // 2 - ps // 2),
        "patch_right": (3 * w // 4 - ps // 2, h // 2 - ps // 2),
    }

    patches = {}

    for name, (x, y) in coords.items():
        x = max(0, min(x, w - ps))
        y = max(0, min(y, h - ps))
        patches[name] = img[y:y + ps, x:x + ps].copy()

    return patches


def save_image(path, img):
    path = Path(path)
    ensure_dir(path.parent)

    if len(img.shape) == 3:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), img_bgr)
    else:
        cv2.imwrite(str(path), img)


def process_rgb_image(participant_id, label, view_name, image_path):
    rows = []
    img = read_rgb(image_path)

    if img is None:
        rows.append({
            "participant_id": participant_id,
            "label": label,
            "modality": "rgb",
            "view": view_name,
            "representation": "FAILED",
            "path": str(image_path),
            "status": "failed_read"
        })
        return rows

    base = REPR_OUT / "rgb" / str(participant_id) / view_name

    reps = {
        "rgb_original": img,
        "rgb_hsv": cv2.cvtColor(img, cv2.COLOR_RGB2HSV),
        "rgb_lab": cv2.cvtColor(img, cv2.COLOR_RGB2LAB),
        "rgb_texture": texture_enhance_gray(img),
    }

    for rep_name, rep_img in reps.items():
        out_path = base / f"{rep_name}.png"
        save_image(out_path, rep_img)
        rows.append({
            "participant_id": participant_id,
            "label": label,
            "modality": "rgb",
            "view": view_name,
            "representation": rep_name,
            "path": str(out_path),
            "status": "saved"
        })

    patches = extract_patches(img)

    for patch_name, patch_img in patches.items():
        out_path = base / "patches" / f"{patch_name}.png"
        save_image(out_path, patch_img)
        rows.append({
            "participant_id": participant_id,
            "label": label,
            "modality": "rgb",
            "view": view_name,
            "representation": patch_name,
            "path": str(out_path),
            "status": "saved"
        })

    return rows


def process_thermal_image(participant_id, label, view_name, image_path):
    rows = []
    img = read_thermal(image_path)

    if img is None:
        rows.append({
            "participant_id": participant_id,
            "label": label,
            "modality": "thermal",
            "view": view_name,
            "representation": "FAILED",
            "path": str(image_path),
            "status": "failed_read"
        })
        return rows

    base = REPR_OUT / "thermal" / str(participant_id) / view_name

    norm = normalize_uint8(img)
    texture = texture_enhance_gray(norm)

    reps = {
        "thermal_normalized": norm,
        "thermal_texture": texture,
    }

    for rep_name, rep_img in reps.items():
        out_path = base / f"{rep_name}.png"
        save_image(out_path, rep_img)
        rows.append({
            "participant_id": participant_id,
            "label": label,
            "modality": "thermal",
            "view": view_name,
            "representation": rep_name,
            "path": str(out_path),
            "status": "saved"
        })

    patches = extract_patches(norm)

    for patch_name, patch_img in patches.items():
        out_path = base / "patches" / f"{patch_name}.png"
        save_image(out_path, patch_img)
        rows.append({
            "participant_id": participant_id,
            "label": label,
            "modality": "thermal",
            "view": view_name,
            "representation": patch_name,
            "path": str(out_path),
            "status": "saved"
        })

    return rows


def main():
    if not INPUT_MANIFEST.exists():
        raise FileNotFoundError(f"Missing Stage 10A manifest: {INPUT_MANIFEST}")

    manifest = pd.read_csv(INPUT_MANIFEST)

    all_rows = []

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Generating representations"):
        participant_id = row["participant_id"]
        label = row["label"]

        rgb_view_map = {
            "l_dorsal": row["rgb_l_dorsal"],
            "l_palmar": row["rgb_l_palmar"],
            "r_dorsal": row["rgb_r_dorsal"],
            "r_palmar": row["rgb_r_palmar"],
        }

        thermal_view_map = {
            "l_dorsal": row["thermal_l_dorsal"],
            "l_palmar": row["thermal_l_palmar"],
            "r_dorsal": row["thermal_r_dorsal"],
            "r_palmar": row["thermal_r_palmar"],
        }

        for view_name, path in rgb_view_map.items():
            all_rows.extend(
                process_rgb_image(participant_id, label, view_name, Path(path))
            )

        for view_name, path in thermal_view_map.items():
            all_rows.extend(
                process_thermal_image(participant_id, label, view_name, Path(path))
            )

    repr_manifest = pd.DataFrame(all_rows)
    repr_manifest.to_csv(TABLES_OUT / "multi_representation_manifest.csv", index=False)

    failed = repr_manifest[repr_manifest["status"] != "saved"].copy()
    failed.to_csv(TABLES_OUT / "failed_representation_generation.csv", index=False)

    summary_by_modality = (
        repr_manifest[repr_manifest["status"] == "saved"]
        .groupby(["modality", "representation"])
        .size()
        .reset_index(name="count")
    )
    summary_by_modality.to_csv(TABLES_OUT / "representation_count_summary.csv", index=False)

    participant_counts = (
        repr_manifest[repr_manifest["status"] == "saved"]
        .groupby(["participant_id", "modality"])
        .size()
        .reset_index(name="representation_count")
    )
    participant_counts.to_csv(TABLES_OUT / "participant_representation_counts.csv", index=False)

    expected_rgb_per_participant = 4 * (4 + 5)
    expected_thermal_per_participant = 4 * (2 + 5)
    expected_total_per_participant = expected_rgb_per_participant + expected_thermal_per_participant

    participant_total = (
        repr_manifest[repr_manifest["status"] == "saved"]
        .groupby("participant_id")
        .size()
        .reset_index(name="total_representations")
    )

    participant_total["expected_total_representations"] = expected_total_per_participant
    participant_total["complete_representations"] = (
        participant_total["total_representations"] == expected_total_per_participant
    )

    participant_total.to_csv(TABLES_OUT / "participant_representation_completeness.csv", index=False)

    incomplete = participant_total[~participant_total["complete_representations"]].copy()
    incomplete.to_csv(TABLES_OUT / "participants_with_incomplete_representations.csv", index=False)

    summary = {
        "stage": "Stage10B",
        "title": "Multi-Representation Generator",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_manifest": str(INPUT_MANIFEST),
        "participants_processed": int(len(manifest)),
        "image_size": IMAGE_SIZE,
        "patch_size": PATCH_SIZE,
        "saved_representations": int((repr_manifest["status"] == "saved").sum()),
        "failed_representations": int((repr_manifest["status"] != "saved").sum()),
        "expected_rgb_representations_per_participant": expected_rgb_per_participant,
        "expected_thermal_representations_per_participant": expected_thermal_per_participant,
        "expected_total_representations_per_participant": expected_total_per_participant,
        "complete_participants": int(participant_total["complete_representations"].sum()),
        "incomplete_participants": int((~participant_total["complete_representations"]).sum()),
        "outputs_saved_to": str(STAGE_OUT)
    }

    with open(STAGE_OUT / "Stage10B_MultiRepresentation_Generator_Summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    report = []
    report.append("# Stage 10B Multi-Representation Generator\n")
    report.append(f"Generated at: {summary['created_at']}\n")

    report.append("## Purpose\n")
    report.append(
        "This stage generates the core CPMR-Net image representations from the participant-level manifest. "
        "It transforms each image into multiple complementary evidence sources before deep-learning encoding.\n"
    )

    report.append("## Generated Representations\n")
    report.append("### RGB")
    report.append("- Original RGB")
    report.append("- HSV")
    report.append("- LAB")
    report.append("- Texture-enhanced grayscale")
    report.append("- Five local anatomical patches\n")

    report.append("### Thermal")
    report.append("- Normalized thermal intensity map")
    report.append("- Texture-enhanced thermal map")
    report.append("- Five local thermal patches\n")

    report.append("## Summary\n")
    report.append(f"- Participants processed: {summary['participants_processed']}")
    report.append(f"- Image size: {summary['image_size']} x {summary['image_size']}")
    report.append(f"- Patch size: {summary['patch_size']} x {summary['patch_size']}")
    report.append(f"- Saved representations: {summary['saved_representations']}")
    report.append(f"- Failed representations: {summary['failed_representations']}")
    report.append(f"- Expected RGB representations per participant: {summary['expected_rgb_representations_per_participant']}")
    report.append(f"- Expected thermal representations per participant: {summary['expected_thermal_representations_per_participant']}")
    report.append(f"- Expected total representations per participant: {summary['expected_total_representations_per_participant']}")
    report.append(f"- Complete participants: {summary['complete_participants']}")
    report.append(f"- Incomplete participants: {summary['incomplete_participants']}\n")

    report.append("## Output Files\n")
    report.append("- `multi_representation_manifest.csv`")
    report.append("- `representation_count_summary.csv`")
    report.append("- `participant_representation_counts.csv`")
    report.append("- `participant_representation_completeness.csv`")
    report.append("- `failed_representation_generation.csv`")
    report.append("- `participants_with_incomplete_representations.csv`")
    report.append("- `Stage10B_MultiRepresentation_Generator_Summary.json`\n")

    report.append("## Implementation Role\n")
    report.append(
        "This stage provides the representation-level inputs for the next CPMR-Net module: "
        "Stage 10C Representation Encoders."
    )

    with open(REPORTS_OUT / "Stage10B_MultiRepresentation_Generator_Report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("=" * 80)
    print("STAGE 10B MULTI-REPRESENTATION GENERATOR COMPLETED")
    print("=" * 80)
    print(f"Participants processed: {summary['participants_processed']}")
    print(f"Saved representations: {summary['saved_representations']}")
    print(f"Failed representations: {summary['failed_representations']}")
    print(f"Expected representations per participant: {summary['expected_total_representations_per_participant']}")
    print(f"Complete participants: {summary['complete_participants']}")
    print(f"Incomplete participants: {summary['incomplete_participants']}")
    print(f"Outputs saved to: {STAGE_OUT}")
    print("=" * 80)


if __name__ == "__main__":
    main()