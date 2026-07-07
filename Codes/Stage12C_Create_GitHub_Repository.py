from pathlib import Path
import subprocess
import shutil
import sys
import json
import csv
from datetime import datetime

# ==========================================================
# Stage 12C: Publication Repository Builder
# CPMR-Net HandDx-200 Anemia Project
# ==========================================================

SOURCE_ROOT = Path(r"D:\47\472\New-Papers\Anemia_Paper\Experiments")

PUBLICATION_ROOT = Path(
    r"D:\47\472\New-Papers\Anemia_Paper\CPMR-Net-HandDx200-Anemia_PublicationRepo"
)

REPO_NAME = "CPMR-Net-HandDx200-Anemia"
DESCRIPTION = (
    "CPMR-Net for participant-level multimodal anemia diagnosis "
    "from HandDx-200 hand images"
)

VISIBILITY = "public"

GIT_USER_NAME = "Mahmoud Rokaya"
GIT_USER_EMAIL = "baselmah@yahoo.com"

MAX_FILE_MIB = 49.0

# ==========================================================
# What to copy
# ==========================================================

COPY_FULL_FOLDERS = [
    "Codes",
]

COPY_ROOT_FILES_IF_EXISTS = [
    "README.md",
    "LICENSE",
    "requirements.txt",
    "CITATION.cff",
]

ALLOWED_RESULT_EXTENSIONS = {
    ".csv",
    ".json",
    ".txt",
    ".md",
}

EXCLUDED_EXTENSIONS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".zip",
    ".rar",
    ".7z",
    ".pt",
    ".pth",
    ".ckpt",
    ".pyc",
    ".pyo",
    ".tmp",
    ".log",
}

EXCLUDED_DIR_NAMES = {
    ".git",
    "__pycache__",
    ".ipynb_checkpoints",
    "Publication_Repository",
    "CPMR-Net-HandDx200-Anemia_PublicationRepo",
}

# Keep only summary-like results, not all raw/generated outputs
IMPORTANT_RESULT_KEYWORDS = [
    "summary",
    "performance",
    "metrics",
    "benchmark",
    "comparison",
    "decision",
    "audit",
    "validation",
    "holdout",
    "cross",
    "cv",
    "confusion",
    "roc",
    "pr",
    "auc",
    "feature",
    "hyperparameter",
    "split",
    "report",
    "readme",
    "statistics",
    "inventory",
    "metadata",
]


# ==========================================================
# Utility functions
# ==========================================================

def run(cmd, cwd=None, check=True):
    print("\n>", " ".join(cmd))
    result = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=False,
    )
    print(result.stdout)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return result


def require_command(command):
    if shutil.which(command) is None:
        raise RuntimeError(
            f"{command} is not available in PATH. "
            f"Install it or add it to PATH before running this script."
        )


def safe_remove_dir(path):
    if path.exists():
        print(f"Removing existing folder: {path}")
        shutil.rmtree(path)


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def size_mib(path):
    return path.stat().st_size / (1024 * 1024)


def is_inside_excluded_dir(path):
    return any(part in EXCLUDED_DIR_NAMES for part in path.parts)


def is_safe_file(path):
    if not path.is_file():
        return False

    if is_inside_excluded_dir(path):
        return False

    if path.suffix.lower() in EXCLUDED_EXTENSIONS:
        return False

    if size_mib(path) >= MAX_FILE_MIB:
        return False

    return True


def copy_file(src, dst):
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def copy_tree_filtered(src_dir, dst_dir):
    copied = []
    skipped = []

    for src in src_dir.rglob("*"):
        if src.is_dir():
            continue

        rel = src.relative_to(src_dir)
        dst = dst_dir / rel

        if is_safe_file(src):
            copy_file(src, dst)
            copied.append(str(dst.relative_to(PUBLICATION_ROOT)))
        else:
            skipped.append(str(src.relative_to(SOURCE_ROOT)))

    return copied, skipped


def looks_like_important_result(path):
    name = path.name.lower()
    parent_text = " ".join(path.parts).lower()

    if path.suffix.lower() not in ALLOWED_RESULT_EXTENSIONS:
        return False

    if not is_safe_file(path):
        return False

    return any(k in name or k in parent_text for k in IMPORTANT_RESULT_KEYWORDS)


def write_text_file(path, content):
    ensure_dir(path.parent)
    path.write_text(content.strip() + "\n", encoding="utf-8")


# ==========================================================
# Repository file generators
# ==========================================================

def generate_gitignore():
    content = """
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.ipynb_checkpoints/

# OS / editor
.DS_Store
Thumbs.db
.vscode/
.idea/

# Large/generated research artifacts
*.pdf
*.png
*.jpg
*.jpeg
*.bmp
*.tif
*.tiff
*.webp
*.zip
*.rar
*.7z
*.pt
*.pth
*.ckpt

# Temporary files
*.tmp
*.log

# Local data folders
Data/
Outputs/
Raw/
raw/
dataset/
datasets/
"""
    write_text_file(PUBLICATION_ROOT / ".gitignore", content)


def generate_license():
    license_path = PUBLICATION_ROOT / "LICENSE"
    if license_path.exists():
        return

    year = datetime.now().year
    content = f"""
MIT License

Copyright (c) {year} Mahmoud Rokaya

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
    write_text_file(license_path, content)


def generate_requirements():
    req_path = PUBLICATION_ROOT / "requirements.txt"
    if req_path.exists():
        return

    content = """
numpy
pandas
scipy
scikit-learn
matplotlib
opencv-python
Pillow
torch
torchvision
tqdm
openpyxl
"""
    write_text_file(req_path, content)


def generate_citation():
    citation_path = PUBLICATION_ROOT / "CITATION.cff"
    if citation_path.exists():
        return

    content = """
cff-version: 1.2.0
message: "If you use this repository, please cite the associated CPMR-Net HandDx-200 manuscript."
title: "CPMR-Net for Participant-Level Multimodal Anemia Diagnosis from Hand Images"
authors:
  - family-names: "Rokaya"
    given-names: "Mahmoud"
repository-code: "https://github.com/mahmoudrokaya/CPMR-Net-HandDx200-Anemia"
license: "MIT"
"""
    write_text_file(citation_path, content)


def generate_readme():
    readme_path = PUBLICATION_ROOT / "README.md"

    if readme_path.exists():
        return

    content = f"""
# CPMR-Net HandDx-200 Anemia

This repository contains the publication-ready code and selected reproducibility artifacts for the project:

**Cooperative Physiological Multi-Representation Network (CPMR-Net) for participant-level multimodal anemia diagnosis from hand images.**

## Project Summary

The project investigates participant-level anemia diagnosis using the HandDx-200 multimodal hand imaging dataset.

The pipeline includes:

- dataset auditing and verification,
- RGB and thermal preprocessing,
- physiological multi-representation generation,
- handcrafted physiological feature extraction,
- statistical validation,
- nonredundant feature selection,
- classical machine-learning benchmarking,
- CPMR-Net architecture development,
- contrastive pretraining,
- progressive fine-tuning,
- conservative holdout evaluation,
- repeated participant-level cross-validation,
- final scientific model audit.

## Important Evaluation Principle

All experiments are performed at the participant level.

Images, thermal maps, physiological representations, handcrafted features, and learned embeddings from the same participant are never split across training, validation, test, or cross-validation folds.

## Repository Contents

```text
Codes/              Full staged implementation scripts
configs/            Selected configuration files
metadata/           Dataset and repository metadata summaries
example_results/    Selected lightweight result summaries
README.md           Project overview
requirements.txt    Python dependencies
LICENSE             License file
CITATION.cff        Citation metadata