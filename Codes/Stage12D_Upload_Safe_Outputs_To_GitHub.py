from pathlib import Path
import subprocess
import shutil
import sys
import json
from datetime import datetime

# ==========================================================
# Stage 12D: Upload Safe Outputs to GitHub
#
# Source:
#   D:\47\472\New-Papers\Anemia_Paper\Experiments\Outputs
#
# Destination inside repository:
#   Outputs/
#
# Repository:
#   https://github.com/mahmoudrokaya/CPMR-Net-HandDx200-Anemia
#
# Policy:
#   - Skip files >= 49 MiB
#   - Skip PDFs, archives, checkpoints, caches, temporary files
#   - Continue even if individual files fail
# ==========================================================

SOURCE_OUTPUTS = Path(r"D:\47\472\New-Papers\Anemia_Paper\Experiments\Outputs")

REPO_DIR = Path(
    r"D:\47\472\New-Papers\Anemia_Paper\CPMR-Net-HandDx200-Anemia_PublicationRepo"
)

REPO_OWNER = "mahmoudrokaya"
REPO_NAME = "CPMR-Net-HandDx200-Anemia"
REMOTE_URL = f"https://github.com/{REPO_OWNER}/{REPO_NAME}.git"

DEST_OUTPUTS = REPO_DIR / "Outputs"

MAX_FILE_MIB = 49.0

GIT_USER_NAME = "Mahmoud Rokaya"
GIT_USER_EMAIL = "baselmah@yahoo.com"

SKIP_EXTENSIONS = {
    ".pdf",
    ".zip",
    ".rar",
    ".7z",
    ".tar",
    ".gz",
    ".pt",
    ".pth",
    ".ckpt",
    ".pyc",
    ".pyo",
    ".tmp",
    ".log",
}

SKIP_DIR_NAMES = {
    ".git",
    "__pycache__",
    ".ipynb_checkpoints",
}

# Keep this False if you want all safe outputs.
# Set True if you want only lightweight table/report files.
ONLY_LIGHTWEIGHT_RESULTS = False

LIGHTWEIGHT_EXTENSIONS = {
    ".csv",
    ".json",
    ".txt",
    ".md",
    ".xlsx",
}


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


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def size_mib(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def has_skipped_dir(path: Path) -> bool:
    return any(part in SKIP_DIR_NAMES for part in path.parts)


def should_skip_file(src: Path):
    if not src.is_file():
        return True, "not_a_file"

    if has_skipped_dir(src.relative_to(SOURCE_OUTPUTS)):
        return True, "excluded_directory"

    suffix = src.suffix.lower()

    if suffix in SKIP_EXTENSIONS:
        return True, f"excluded_extension_{suffix}"

    if ONLY_LIGHTWEIGHT_RESULTS and suffix not in LIGHTWEIGHT_EXTENSIONS:
        return True, f"not_lightweight_result_{suffix}"

    file_size = size_mib(src)

    if file_size >= MAX_FILE_MIB:
        return True, f"oversized_{file_size:.2f}_MiB"

    return False, "included"


def safe_copy_outputs():
    if not SOURCE_OUTPUTS.exists():
        raise FileNotFoundError(f"Outputs folder not found: {SOURCE_OUTPUTS}")

    ensure_dir(REPO_DIR)
    ensure_dir(DEST_OUTPUTS)

    copied = []
    skipped = []

    for src in SOURCE_OUTPUTS.rglob("*"):
        if src.is_dir():
            continue

        rel = src.relative_to(SOURCE_OUTPUTS)
        dst = DEST_OUTPUTS / rel

        skip, reason = should_skip_file(src)

        if skip:
            skipped.append({
                "relative_path": str(Path("Outputs") / rel).replace("\\", "/"),
                "reason": reason,
                "size_mib": round(size_mib(src), 4) if src.exists() else None,
            })
            continue

        try:
            ensure_dir(dst.parent)
            shutil.copy2(src, dst)
            copied.append({
                "relative_path": str(Path("Outputs") / rel).replace("\\", "/"),
                "size_mib": round(size_mib(dst), 4),
            })
        except Exception as exc:
            skipped.append({
                "relative_path": str(Path("Outputs") / rel).replace("\\", "/"),
                "reason": f"copy_failed_{exc}",
                "size_mib": round(size_mib(src), 4),
            })

    report_dir = REPO_DIR / "metadata"
    ensure_dir(report_dir)

    report = {
        "generated_at": datetime.now().isoformat(),
        "source_outputs": str(SOURCE_OUTPUTS),
        "destination_outputs": str(DEST_OUTPUTS),
        "max_file_mib": MAX_FILE_MIB,
        "only_lightweight_results": ONLY_LIGHTWEIGHT_RESULTS,
        "copied_count": len(copied),
        "skipped_count": len(skipped),
        "copied_files": copied,
        "skipped_files": skipped,
    }

    report_path = report_dir / "Outputs_Upload_Report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_lines = [
        "# Outputs Upload Report",
        "",
        f"Generated at: `{report['generated_at']}`",
        "",
        f"Source folder: `{SOURCE_OUTPUTS}`",
        f"Destination folder in repository: `Outputs/`",
        "",
        f"Copied files: **{len(copied)}**",
        f"Skipped files: **{len(skipped)}**",
        "",
        "Files were skipped when they exceeded the conservative file-size limit, used excluded extensions, or failed during copying.",
        "",
        "## Skipped files",
        "",
    ]

    for item in skipped[:500]:
        md_lines.append(
            f"- `{item['relative_path']}` — {item['reason']} — {item['size_mib']} MiB"
        )

    if len(skipped) > 500:
        md_lines.append(f"- ... {len(skipped) - 500} more skipped files listed in JSON report.")

    (report_dir / "Outputs_Upload_Report.md").write_text(
        "\n".join(md_lines) + "\n",
        encoding="utf-8",
    )

    print("\nCopy completed.")
    print(f"Copied files : {len(copied)}")
    print(f"Skipped files: {len(skipped)}")
    print(f"Report saved : {report_path}")

    return copied, skipped


def setup_repository():
    require_command("git")
    require_command("gh")

    run(["gh", "auth", "status"], check=True)

    run(["git", "config", "--global", "user.name", GIT_USER_NAME])
    run(["git", "config", "--global", "user.email", GIT_USER_EMAIL])

    ensure_dir(REPO_DIR)

    if not (REPO_DIR / ".git").exists():
        run(["git", "init"], cwd=REPO_DIR)

    run(["git", "branch", "-M", "main"], cwd=REPO_DIR)

    remote_check = run(
        ["git", "remote", "get-url", "origin"],
        cwd=REPO_DIR,
        check=False,
    )

    if remote_check.returncode != 0:
        run(["git", "remote", "add", "origin", REMOTE_URL], cwd=REPO_DIR)
    else:
        current_remote = remote_check.stdout.strip()
        if current_remote != REMOTE_URL:
            run(["git", "remote", "set-url", "origin", REMOTE_URL], cwd=REPO_DIR)


def update_gitignore():
    gitignore = REPO_DIR / ".gitignore"

    additions = [
        "",
        "# Stage12D safety exclusions",
        "*.pdf",
        "*.zip",
        "*.rar",
        "*.7z",
        "*.tar",
        "*.gz",
        "*.pt",
        "*.pth",
        "*.ckpt",
        "*.pyc",
        "*.pyo",
        "*.tmp",
        "*.log",
        "__pycache__/",
        ".ipynb_checkpoints/",
    ]

    existing = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""

    changed = False
    for line in additions:
        if line and line not in existing:
            existing += line + "\n"
            changed = True

    if changed or not gitignore.exists():
        gitignore.write_text(existing, encoding="utf-8")

    run(["git", "add", ".gitignore"], cwd=REPO_DIR, check=False)


def stage_outputs_safely():
    update_gitignore()

    files_to_add = []

    for path in DEST_OUTPUTS.rglob("*"):
        if not path.is_file():
            continue

        if ".git" in path.relative_to(REPO_DIR).parts:
            continue

        if size_mib(path) >= MAX_FILE_MIB:
            print(f"Skipping oversized staged candidate: {path}")
            continue

        if path.suffix.lower() in SKIP_EXTENSIONS:
            print(f"Skipping excluded staged candidate: {path}")
            continue

        files_to_add.append(path)

    # Add reports
    metadata_dir = REPO_DIR / "metadata"
    for report in [
        metadata_dir / "Outputs_Upload_Report.json",
        metadata_dir / "Outputs_Upload_Report.md",
    ]:
        if report.exists():
            files_to_add.append(report)

    added = 0
    failed = []

    for path in files_to_add:
        rel = str(path.relative_to(REPO_DIR)).replace("\\", "/")
        result = run(["git", "add", rel], cwd=REPO_DIR, check=False)

        if result.returncode == 0:
            added += 1
        else:
            failed.append(rel)

    print(f"\nFiles staged successfully: {added}")
    print(f"Files failed to stage   : {len(failed)}")

    if failed:
        fail_report = REPO_DIR / "metadata" / "Git_Add_Failures.txt"
        fail_report.write_text("\n".join(failed), encoding="utf-8")
        run(["git", "add", "metadata/Git_Add_Failures.txt"], cwd=REPO_DIR, check=False)


def verify_no_policy_violations():
    tracked = run(["git", "ls-files"], cwd=REPO_DIR).stdout.splitlines()

    violations = []

    for rel in tracked:
        path = REPO_DIR / rel

        if not path.exists() or not path.is_file():
            continue

        if path.suffix.lower() in SKIP_EXTENSIONS:
            violations.append(f"excluded_extension: {rel}")
            continue

        if size_mib(path) >= MAX_FILE_MIB:
            violations.append(f"oversized: {rel} ({size_mib(path):.2f} MiB)")
            continue

    if violations:
        print("\nPolicy violations found:")
        for item in violations[:200]:
            print(item)
        raise RuntimeError("Tracked files include GitHub-policy-risk files.")

    print("\nPolicy verification passed.")


def commit_and_push():
    status = run(["git", "status", "--short"], cwd=REPO_DIR)

    if not status.stdout.strip():
        print("\nNothing new to commit.")
        return

    commit_msg = "Add safe Outputs folder artifacts"
    run(["git", "commit", "-m", commit_msg], cwd=REPO_DIR)

    # Push with safer HTTP settings.
    run(["git", "config", "http.version", "HTTP/1.1"], cwd=REPO_DIR, check=False)
    run(["git", "config", "http.postBuffer", "524288000"], cwd=REPO_DIR, check=False)

    push_result = run(["git", "push", "-u", "origin", "main"], cwd=REPO_DIR, check=False)

    if push_result.returncode != 0:
        print("\nPush failed. The commit was created locally.")
        print("You can retry manually with:")
        print(f"cd {REPO_DIR}")
        print("git push -u origin main")
        raise RuntimeError("Git push failed, but unsafe files were skipped.")

    print("\nOutputs uploaded successfully.")


def main():
    setup_repository()
    safe_copy_outputs()
    stage_outputs_safely()
    verify_no_policy_violations()
    commit_and_push()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nFAILED:")
        print(exc)
        sys.exit(1)