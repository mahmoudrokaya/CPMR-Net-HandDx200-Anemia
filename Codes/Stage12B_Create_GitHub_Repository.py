from pathlib import Path
import subprocess
import shutil
import sys

PROJECT_DIR = Path(r"D:\47\472\New-Papers\Anemia_Paper\Experiments")
REPO_NAME = "CPMR-Net-HandDx200-Anemia"
DESCRIPTION = "CPMR-Net for participant-level multimodal anemia diagnosis from HandDx-200 hand images"
VISIBILITY = "public"

GIT_USER_NAME = "Mahmoud Rokaya"
GIT_USER_EMAIL = "baselmah@yahoo.com"

MAX_TRACKED_MIB = 49.0

EXCLUDED_EXTENSIONS = {
    ".pdf", ".zip", ".7z", ".rar",
    ".pt", ".pth", ".ckpt",
    ".pyc", ".pyo",
    ".tmp", ".log"
}

EXCLUDED_DIRS = {
    ".git",
    "__pycache__",
    ".ipynb_checkpoints"
}


def run(cmd, cwd=None, check=True):
    print("\n>", " ".join(cmd))
    result = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    print(result.stdout)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return result


def require_command(cmd):
    if shutil.which(cmd) is None:
        raise RuntimeError(f"{cmd} is not available in PATH.")


def is_excluded_path(path: Path):
    rel_parts = set(path.parts)
    if rel_parts.intersection(EXCLUDED_DIRS):
        return True
    if path.suffix.lower() in EXCLUDED_EXTENSIONS:
        return True
    if path.is_file():
        size_mib = path.stat().st_size / (1024 * 1024)
        if size_mib >= MAX_TRACKED_MIB:
            return True
    return False


def build_gitignore():
    oversized = []

    for p in PROJECT_DIR.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(PROJECT_DIR)
        if ".git" in rel.parts:
            continue
        size_mib = p.stat().st_size / (1024 * 1024)
        if size_mib >= MAX_TRACKED_MIB:
            oversized.append((rel, size_mib))

    lines = [
        "# Auto-generated GitHub-safe ignore file",
        "",
        "# Large/generated files",
        "*.pdf",
        "*.PDF",
        "*.zip",
        "*.ZIP",
        "*.7z",
        "*.rar",
        "*.pt",
        "*.pth",
        "*.ckpt",
        "",
        "# Python cache",
        "__pycache__/",
        "*.pyc",
        "*.pyo",
        "",
        "# Temporary files",
        "*.tmp",
        "*.log",
        ".ipynb_checkpoints/",
        ".DS_Store",
        "Thumbs.db",
        "",
        "# Oversized files detected automatically",
    ]

    for rel, size_mib in sorted(oversized, key=lambda x: str(x[0]).lower()):
        lines.append(f"{str(rel).replace(chr(92), '/')}  # {size_mib:.2f} MiB")

    (PROJECT_DIR / ".gitignore").write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = PROJECT_DIR / "GitHub_Excluded_Large_Files.txt"
    report.write_text(
        "GitHub excluded large files\n"
        "===========================\n\n"
        f"Policy: files >= {MAX_TRACKED_MIB} MiB are excluded.\n"
        "Also excluded: PDFs, archives, PyTorch checkpoints, Python cache.\n\n"
        + "\n".join(f"{size_mib:.2f} MiB | {rel}" for rel, size_mib in oversized),
        encoding="utf-8"
    )

    return oversized


def verify_no_large_tracked_files():
    tracked = run(["git", "ls-files"], cwd=PROJECT_DIR).stdout.splitlines()
    bad = []

    for rel in tracked:
        p = PROJECT_DIR / rel
        if p.exists() and p.is_file():
            size_mib = p.stat().st_size / (1024 * 1024)
            if size_mib >= MAX_TRACKED_MIB:
                bad.append((rel, size_mib))

    if bad:
        print("\nBlocked: oversized files are still tracked:")
        for rel, size_mib in bad:
            print(f"{size_mib:.2f} MiB | {rel}")
        raise RuntimeError("Oversized tracked files remain.")

    print("Large-file policy check passed.")


def main():
    if not PROJECT_DIR.exists():
        raise FileNotFoundError(PROJECT_DIR)

    require_command("git")
    require_command("gh")

    run(["gh", "auth", "status"])

    run(["git", "config", "--global", "user.name", GIT_USER_NAME])
    run(["git", "config", "--global", "user.email", GIT_USER_EMAIL])

    oversized = build_gitignore()
    print(f"\nExcluded oversized files: {len(oversized)}")

    if not (PROJECT_DIR / ".git").exists():
        run(["git", "init"], cwd=PROJECT_DIR)

    run(["git", "branch", "-M", "main"], cwd=PROJECT_DIR)

    remote_check = run(["git", "remote", "get-url", "origin"], cwd=PROJECT_DIR, check=False)

    if remote_check.returncode != 0:
        repo_check = run(["gh", "repo", "view", REPO_NAME], cwd=PROJECT_DIR, check=False)

        if repo_check.returncode != 0:
            run([
                "gh", "repo", "create", REPO_NAME,
                f"--{VISIBILITY}",
                "--description", DESCRIPTION,
                "--source", str(PROJECT_DIR),
                "--remote", "origin"
            ], cwd=PROJECT_DIR)
        else:
            run([
                "git", "remote", "add", "origin",
                f"https://github.com/{GIT_USER_NAME.replace(' ', '').lower()}/{REPO_NAME}.git"
            ], cwd=PROJECT_DIR)

    run(["git", "rm", "-r", "--cached", "."], cwd=PROJECT_DIR, check=False)
    run(["git", "add", "-A"], cwd=PROJECT_DIR)

    verify_no_large_tracked_files()

    status = run(["git", "status", "--short"], cwd=PROJECT_DIR)

    if status.stdout.strip():
        run(["git", "commit", "-m", "Initial CPMR-Net HandDx-200 project repository"], cwd=PROJECT_DIR)
    else:
        print("No changes to commit.")

    run(["git", "push", "-u", "origin", "main"], cwd=PROJECT_DIR)

    print("\nGitHub repository upload completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nFAILED:", e)
        sys.exit(1)