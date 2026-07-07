# -*- coding: utf-8 -*-
"""
Stage 10C - Representation Encoder Dataset and Tensor Validation

Purpose:
Validate Stage 10B generated representations before encoder implementation.

This stage:
- Reads multi_representation_manifest.csv
- Loads every representation image
- Checks shape, channels, dtype, intensity range
- Assigns encoder input type
- Creates encoder-ready manifest
- Reports failed or inconsistent tensors

No model training is performed.
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

STAGE10B_DIR = OUTPUTS_DIR / "Stage10B_MultiRepresentation_Generator"
INPUT_MANIFEST = STAGE10B_DIR / "tables" / "multi_representation_manifest.csv"

STAGE_OUT = OUTPUTS_DIR / "Stage10C_Representation_Encoder_Dataset_Tensor_Validation"
TABLES_OUT = STAGE_OUT / "tables"
REPORTS_OUT = STAGE_OUT / "reports"

TABLES_OUT.mkdir(parents=True, exist_ok=True)
REPORTS_OUT.mkdir(parents=True, exist_ok=True)


RGB_LIKE_REPRESENTATIONS = {
    "rgb_original",
    "rgb_hsv",
    "rgb_lab",
    "patch_center",
    "patch_upper",
    "patch_lower",
    "patch_left",
    "patch_right",
}

GRAY_LIKE_REPRESENTATIONS = {
    "rgb_texture",
    "thermal_normalized",
    "thermal_texture",
}


def infer_encoder_input_type(modality, representation):
    if modality == "rgb":
        if representation in ["rgb_original", "rgb_hsv", "rgb_lab"]:
            return "three_channel_full_image"
        if representation.startswith("patch_"):
            return "three_channel_patch"
        if representation == "rgb_texture":
            return "single_channel_texture"
        return "unknown_rgb"

    if modality == "thermal":
        if representation in ["thermal_normalized", "thermal_texture"]:
            return "single_channel_full_image"
        if representation.startswith("patch_"):
            return "single_channel_patch"
        return "unknown_thermal"

    return "unknown"


def expected_channels(modality, representation):
    if modality == "rgb":
        if representation == "rgb_texture":
            return 1
        return 3

    if modality == "thermal":
        return 1

    return None


def expected_size(representation):
    if representation.startswith("patch_"):
        return 112, 112
    return 224, 224


def load_image_info(path):
    path = Path(path)
    if not path.exists():
        return {
            "loaded": False,
            "reason": "file_missing",
            "height": None,
            "width": None,
            "channels": None,
            "dtype": None,
            "min_value": None,
            "max_value": None,
            "mean_value": None,
            "std_value": None
        }

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if img is None:
        return {
            "loaded": False,
            "reason": "cv2_read_failed",
            "height": None,
            "width": None,
            "channels": None,
            "dtype": None,
            "min_value": None,
            "max_value": None,
            "mean_value": None,
            "std_value": None
        }

    if len(img.shape) == 2:
        h, w = img.shape
        c = 1
    else:
        h, w, c = img.shape

    arr = img.astype(np.float32)

    return {
        "loaded": True,
        "reason": "",
        "height": int(h),
        "width": int(w),
        "channels": int(c),
        "dtype": str(img.dtype),
        "min_value": float(np.min(arr)),
        "max_value": float(np.max(arr)),
        "mean_value": float(np.mean(arr)),
        "std_value": float(np.std(arr))
    }


def main():
    if not INPUT_MANIFEST.exists():
        raise FileNotFoundError(f"Missing Stage 10B manifest: {INPUT_MANIFEST}")

    manifest = pd.read_csv(INPUT_MANIFEST)

    records = []

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Validating tensors"):
        participant_id = row["participant_id"]
        label = row["label"]
        modality = row["modality"]
        view = row["view"]
        representation = row["representation"]
        path = row["path"]
        status = row["status"]

        encoder_type = infer_encoder_input_type(modality, representation)
        exp_channels = expected_channels(modality, representation)
        exp_h, exp_w = expected_size(representation)

        info = load_image_info(path)

        shape_ok = (
            info["loaded"]
            and info["height"] == exp_h
            and info["width"] == exp_w
            and info["channels"] == exp_channels
        )

        intensity_ok = (
            info["loaded"]
            and info["min_value"] is not None
            and info["max_value"] is not None
            and info["min_value"] >= 0
            and info["max_value"] <= 255
        )

        dtype_ok = info["loaded"] and info["dtype"] in ["uint8", "uint16"]

        encoder_ready = (
            status == "saved"
            and info["loaded"]
            and shape_ok
            and intensity_ok
            and dtype_ok
            and not encoder_type.startswith("unknown")
        )

        if not info["loaded"]:
            failure_reason = info["reason"]
        elif not shape_ok:
            failure_reason = "shape_or_channel_mismatch"
        elif not intensity_ok:
            failure_reason = "intensity_range_invalid"
        elif not dtype_ok:
            failure_reason = "dtype_invalid"
        elif encoder_type.startswith("unknown"):
            failure_reason = "unknown_encoder_input_type"
        else:
            failure_reason = ""

        records.append({
            "participant_id": participant_id,
            "label": label,
            "modality": modality,
            "view": view,
            "representation": representation,
            "path": path,
            "source_status": status,
            "encoder_input_type": encoder_type,
            "expected_height": exp_h,
            "expected_width": exp_w,
            "expected_channels": exp_channels,
            "loaded": info["loaded"],
            "height": info["height"],
            "width": info["width"],
            "channels": info["channels"],
            "dtype": info["dtype"],
            "min_value": info["min_value"],
            "max_value": info["max_value"],
            "mean_value": info["mean_value"],
            "std_value": info["std_value"],
            "shape_ok": shape_ok,
            "intensity_ok": intensity_ok,
            "dtype_ok": dtype_ok,
            "encoder_ready": encoder_ready,
            "failure_reason": failure_reason
        })

    validation_df = pd.DataFrame(records)

    validation_df.to_csv(TABLES_OUT / "representation_tensor_validation_full.csv", index=False)

    encoder_ready_df = validation_df[validation_df["encoder_ready"]].copy()
    encoder_ready_df.to_csv(TABLES_OUT / "encoder_ready_representation_manifest.csv", index=False)

    failed_df = validation_df[~validation_df["encoder_ready"]].copy()
    failed_df.to_csv(TABLES_OUT / "failed_tensor_validation.csv", index=False)

    summary_by_type = (
        validation_df.groupby(["modality", "representation", "encoder_input_type"])
        .agg(
            total=("path", "count"),
            encoder_ready=("encoder_ready", "sum"),
            failed=("encoder_ready", lambda x: int((~x).sum())),
            mean_intensity=("mean_value", "mean"),
            std_intensity=("std_value", "mean")
        )
        .reset_index()
    )
    summary_by_type.to_csv(TABLES_OUT / "tensor_validation_summary_by_representation.csv", index=False)

    participant_completeness = (
        validation_df.groupby("participant_id")
        .agg(
            total_representations=("path", "count"),
            encoder_ready_representations=("encoder_ready", "sum"),
            failed_representations=("encoder_ready", lambda x: int((~x).sum()))
        )
        .reset_index()
    )

    participant_completeness["complete_for_encoder"] = (
        participant_completeness["total_representations"]
        == participant_completeness["encoder_ready_representations"]
    )

    participant_completeness.to_csv(TABLES_OUT / "participant_encoder_readiness.csv", index=False)

    modality_counts = (
        validation_df.groupby(["modality", "encoder_ready"])
        .size()
        .reset_index(name="count")
    )
    modality_counts.to_csv(TABLES_OUT / "encoder_readiness_by_modality.csv", index=False)

    summary = {
        "stage": "Stage10C",
        "title": "Representation Encoder Dataset and Tensor Validation",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_manifest": str(INPUT_MANIFEST),
        "total_representations_checked": int(len(validation_df)),
        "encoder_ready_representations": int(validation_df["encoder_ready"].sum()),
        "failed_representations": int((~validation_df["encoder_ready"]).sum()),
        "participants_checked": int(validation_df["participant_id"].nunique()),
        "participants_complete_for_encoder": int(participant_completeness["complete_for_encoder"].sum()),
        "participants_incomplete_for_encoder": int((~participant_completeness["complete_for_encoder"]).sum()),
        "outputs_saved_to": str(STAGE_OUT)
    }

    with open(STAGE_OUT / "Stage10C_Representation_Encoder_Dataset_Tensor_Validation_Summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    report = []
    report.append("# Stage 10C Representation Encoder Dataset and Tensor Validation\n")
    report.append(f"Generated at: {summary['created_at']}\n")

    report.append("## Purpose\n")
    report.append(
        "This stage validates all generated Stage 10B representations before they are used "
        "by CPMR-Net representation encoders. It checks file loading, tensor shape, channel count, "
        "dtype, intensity range, and encoder input type.\n"
    )

    report.append("## Summary\n")
    report.append(f"- Total representations checked: {summary['total_representations_checked']}")
    report.append(f"- Encoder-ready representations: {summary['encoder_ready_representations']}")
    report.append(f"- Failed representations: {summary['failed_representations']}")
    report.append(f"- Participants checked: {summary['participants_checked']}")
    report.append(f"- Participants complete for encoder: {summary['participants_complete_for_encoder']}")
    report.append(f"- Participants incomplete for encoder: {summary['participants_incomplete_for_encoder']}\n")

    report.append("## Output Files\n")
    report.append("- `representation_tensor_validation_full.csv`")
    report.append("- `encoder_ready_representation_manifest.csv`")
    report.append("- `failed_tensor_validation.csv`")
    report.append("- `tensor_validation_summary_by_representation.csv`")
    report.append("- `participant_encoder_readiness.csv`")
    report.append("- `encoder_readiness_by_modality.csv`")
    report.append("- `Stage10C_Representation_Encoder_Dataset_Tensor_Validation_Summary.json`\n")

    report.append("## Implementation Role\n")
    report.append(
        "The `encoder_ready_representation_manifest.csv` file is the official input for "
        "Stage 10D Representation Encoder implementation."
    )

    with open(REPORTS_OUT / "Stage10C_Representation_Encoder_Dataset_Tensor_Validation_Report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("=" * 80)
    print("STAGE 10C REPRESENTATION ENCODER DATASET AND TENSOR VALIDATION COMPLETED")
    print("=" * 80)
    print(f"Total representations checked: {summary['total_representations_checked']}")
    print(f"Encoder-ready representations: {summary['encoder_ready_representations']}")
    print(f"Failed representations: {summary['failed_representations']}")
    print(f"Participants checked: {summary['participants_checked']}")
    print(f"Participants complete for encoder: {summary['participants_complete_for_encoder']}")
    print(f"Participants incomplete for encoder: {summary['participants_incomplete_for_encoder']}")
    print(f"Outputs saved to: {STAGE_OUT}")
    print("=" * 80)


if __name__ == "__main__":
    main()