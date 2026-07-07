from pathlib import Path
import subprocess
import shutil
import sys
from datetime import datetime

# ==========================================================
# Stage 12C: Publication Repository Builder
# Upload ONLY:
#   Codes/
#   README.md
#   LICENSE
#   requirements.txt
#   configs/
#   metadata/
#   example_results/
# ==========================================================

SOURCE_ROOT = Path(r"D:\47\472\New-Papers\Anemia_Paper\Experiments")

PUBLICATION_ROOT = Path(
    r"D:\47\472\New-Papers\Anemia_Paper\CPMR-Net-HandDx200-Anemia_PublicationRepo"
)

REPO_NAME = "CPMR-Net-HandDx200-Anemia"
REPO_OWNER = "mahmoudrokaya"
DESCRIPTION = "CPMR-Net for participant-level multimodal anemia diagnosis from HandDx-200 hand images"
VISIBILITY = "public"

GIT_USER_NAME = "Mahmoud Rokaya"
GIT_USER_EMAIL = "baselmah@yahoo.com"

# Only these items will be included and uploaded
WHITELIST_ITEMS = [
    "Codes",
    "README.md",
    "LICENSE",
    "requirements.txt",
    "configs",
    "metadata",
    "example_results",
]

# Files/folders excluded inside copied folders
EXCLUDED_DIRS = {
    ".git",
    "__pycache__",
    ".ipynb_checkpoints",
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

MAX_FILE_MIB = 49.0


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
        raise RuntimeError(f"{command} is not available in PATH.")


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def safe_delete_folder(path):
    if path.exists():
        print(f"Deleting old publication repository folder:\n{path}")
        shutil.rmtree(path)


def file_size_mib(path):
    return path.stat().st_size / (1024 * 1024)


def is_safe_file(path):
    if not path.is_file():
        return False

    if any(part in EXCLUDED_DIRS for part in path.parts):
        return False

    if path.suffix.lower() in EXCLUDED_EXTENSIONS:
        return False

    if file_size_mib(path) >= MAX_FILE_MIB:
        return False

    return True


def copy_file(src, dst):
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def copy_folder_filtered(src_dir, dst_dir):
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


def write_text(path, content):
    ensure_dir(path.parent)
    path.write_text(content.strip() + "\n", encoding="utf-8")


# ==========================================================
# Default files
# ==========================================================

def create_default_readme():
    readme = PUBLICATION_ROOT / "README.md"
    if readme.exists():
        return

    content = """
# CPMR-Net HandDx-200 Anemia

This repository contains the publication-ready implementation of the project:

**Cooperative Physiological Multi-Representation Network (CPMR-Net) for participant-level multimodal anemia diagnosis from hand images.**

## Repository Scope

This GitHub repository intentionally contains only the lightweight reproducibility package:

- `Codes/`
- `README.md`
- `LICENSE`
- `requirements.txt`
- `configs/`
- `metadata/`
- `example_results/`

Large generated files, raw image datasets, PDFs, archives, trained checkpoints, and image outputs are excluded.

## Key Principle

All experiments are participant-level. Images and derived representations from the same participant are never split across training, validation, test, or cross-validation folds.

## Main Components

- Dataset audit and verification
- Thermal decoding
- Physiological representation generation
- Handcrafted feature extraction
- Statistical validation
- Nonredundant feature selection
- Classical ML benchmarks
- CPMR-Net architecture
- Contrastive pretraining
- Progressive fine-tuning
- Holdout and repeated participant-level cross-validation
- Scientific decision audit

## Data

The HandDx-200 dataset must be obtained from its original public source. This repository does not redistribute the full image dataset.

## License

MIT License.
"""
    write_text(readme, content)


def create_default_license():
    license_file = PUBLICATION_ROOT / "LICENSE"
    if license_file.exists():
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
    write_text(license_file, content)


def create_default_requirements():
    req = PUBLICATION_ROOT / "requirements.txt"
    if req.exists():
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
    write_text(req, content)


def create_gitignore():
    content = """
# Excluded generated/heavy files
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

# Python cache
__pycache__/
*.pyc
*.pyo
.ipynb_checkpoints/

# OS/editor files
.DS_Store
Thumbs.db
.vscode/
.idea/

# Local/generated folders
Data/
Outputs/
Raw/
dataset/
datasets/
"""
    write_text(PUBLICATION_ROOT / ".gitignore", content)


# ==========================================================
# Build clean publication repository
# ==========================================================

def build_publication_repo():
    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(f"Source folder does not exist: {SOURCE_ROOT}")

    safe_delete_folder(PUBLICATION_ROOT)
    ensure_dir(PUBLICATION_ROOT)

    copied = []
    skipped = []

    for item in WHITELIST_ITEMS:
        src = SOURCE_ROOT / item
        dst = PUBLICATION_ROOT / item

        if not src.exists():
            print(f"Missing in source, will create if needed: {item}")
            continue

        if src.is_file():
            if is_safe_file(src):
                copy_file(src, dst)
                copied.append(item)
            else:
                skipped.append(item)

        elif src.is_dir():
            copied_items, skipped_items = copy_folder_filtered(src, dst)
            copied.extend(copied_items)
            skipped.extend(skipped_items)

    # Create required folders even if absent in source
    ensure_dir(PUBLICATION_ROOT / "configs")
    ensure_dir(PUBLICATION_ROOT / "metadata")
    ensure_dir(PUBLICATION_ROOT / "example_results")

    create_default_readme()
    create_default_license()
    create_default_requirements()
    create_gitignore()

    report = [
        "# Publication Repository Build Report",
        "",
        f"Generated at: {datetime.now().isoformat()}",
        f"Source root: `{SOURCE_ROOT}`",
        f"Publication root: `{PUBLICATION_ROOT}`",
        "",
        "## Whitelisted upload items",
        "",
    ]

    for item in WHITELIST_ITEMS:
        report.append(f"- `{item}`")

    report.extend([
        "",
        f"Included file count: {len(copied)}",
        f"Skipped file count: {len(skipped)}",
        "",
        "Large files, PDFs, images, archives, checkpoints, and cache files were excluded.",
    ])

    write_text(PUBLICATION_ROOT / "metadata" / "Repository_Build_Report.md", "\n".join(report))

    print("\nPublication repository created:")
    print(PUBLICATION_ROOT)
    print(f"Included files: {len(copied)}")
    print(f"Skipped files: {len(skipped)}")


# ==========================================================
# Verification
# ==========================================================

def verify_publication_repo():
    allowed_top = set(WHITELIST_ITEMS)
    allowed_top.add(".gitignore")

    violations = []

    for p in PUBLICATION_ROOT.rglob("*"):
        if not p.is_file():
            continue

        rel = p.relative_to(PUBLICATION_ROOT)
        parts = rel.parts

        if ".git" in parts:
            continue

        top = parts[0]

        if top not in allowed_top:
            violations.append(f"Not whitelisted: {rel}")
            continue

        if p.suffix.lower() in EXCLUDED_EXTENSIONS:
            violations.append(f"Excluded extension: {rel}")
            continue

        if file_size_mib(p) >= MAX_FILE_MIB:
            violations.append(f"Oversized file: {rel} ({file_size_mib(p):.2f} MiB)")
            continue

    if violations:
        print("\nRepository verification failed:")
        for v in violations[:100]:
            print(v)
        raise RuntimeError("Publication repository contains disallowed files.")

    print("\nRepository verification passed.")


# ==========================================================
# GitHub upload
# ==========================================================

def setup_git_and_github():
    require_command("git")
    require_command("gh")

    run(["gh", "auth", "status"])

    run(["git", "config", "--global", "user.name", GIT_USER_NAME])
    run(["git", "config", "--global", "user.email", GIT_USER_EMAIL])

    if not (PUBLICATION_ROOT / ".git").exists():
        run(["git", "init"], cwd=PUBLICATION_ROOT)

    run(["git", "branch", "-M", "main"], cwd=PUBLICATION_ROOT)

    remote_check = run(
        ["git", "remote", "get-url", "origin"],
        cwd=PUBLICATION_ROOT,
        check=False,
    )

    if remote_check.returncode != 0:
        repo_check = run(
            ["gh", "repo", "view", f"{REPO_OWNER}/{REPO_NAME}"],
            cwd=PUBLICATION_ROOT,
            check=False,
        )

        if repo_check.returncode != 0:
            run(
                [
                    "gh",
                    "repo",
                    "create",
                    REPO_NAME,
                    f"--{VISIBILITY}",
                    "--description",
                    DESCRIPTION,
                    "--source",
                    str(PUBLICATION_ROOT),
                    "--remote",
                    "origin",
                ],
                cwd=PUBLICATION_ROOT,
            )
        else:
            run(
                [
                    "git",
                    "remote",
                    "add",
                    "origin",
                    f"https://github.com/{REPO_OWNER}/{REPO_NAME}.git",
                ],
                cwd=PUBLICATION_ROOT,
            )


def stage_only_whitelist():
    run(["git", "rm", "-r", "--cached", "."], cwd=PUBLICATION_ROOT, check=False)

    for item in WHITELIST_ITEMS:
        p = PUBLICATION_ROOT / item
        if p.exists():
            run(["git", "add", item], cwd=PUBLICATION_ROOT)

    run(["git", "add", ".gitignore"], cwd=PUBLICATION_ROOT)

    tracked = run(["git", "ls-files"], cwd=PUBLICATION_ROOT).stdout.splitlines()

    violations = []

    for item in tracked:
        normalized = item.replace("\\", "/")
        top = normalized.split("/")[0]

        if normalized == ".gitignore":
            continue

        if top not in WHITELIST_ITEMS:
            violations.append(normalized)
            continue

        p = PUBLICATION_ROOT / item
        if p.exists() and p.is_file():
            if p.suffix.lower() in EXCLUDED_EXTENSIONS:
                violations.append(normalized)
            elif file_size_mib(p) >= MAX_FILE_MIB:
                violations.append(normalized)

    if violations:
        print("\nUnexpected tracked files:")
        for v in violations:
            print(v)
        raise RuntimeError("Git staging contains files outside whitelist.")

    print("\nGit staging verification passed.")


def commit_and_push():
    status = run(["git", "status", "--short"], cwd=PUBLICATION_ROOT)

    if status.stdout.strip():
        run(
            ["git", "commit", "-m", "Initial publication-ready CPMR-Net repository"],
            cwd=PUBLICATION_ROOT,
        )
    else:
        print("Nothing to commit.")

    run(["git", "push", "-u", "origin", "main"], cwd=PUBLICATION_ROOT)


# ==========================================================
# Main
# ==========================================================

def main():
    build_publication_repo()
    verify_publication_repo()
    setup_git_and_github()
    stage_only_whitelist()
    commit_and_push()

    print("\nCompleted successfully.")
    print(f"Repository folder: {PUBLICATION_ROOT}")
    print(f"GitHub repository: https://github.com/{REPO_OWNER}/{REPO_NAME}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nFAILED:")
        print(e)
        sys.exit(1)