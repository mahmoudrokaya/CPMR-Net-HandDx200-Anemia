# -*- coding: utf-8 -*-
r"""
Stage 0 - Project Folder Explorer
Explores the full Anemia_Paper project folder without modifying existing files.
"""

from pathlib import Path
import os
import json
import pandas as pd
from datetime import datetime

BASE_DIR = Path(r"D:\47\472\New-Papers\Anemia_Paper")
CODES_DIR = BASE_DIR / "Codes"
OUTPUTS_DIR = BASE_DIR / "Outputs"
STAGE_OUT = OUTPUTS_DIR / "Stage0_Project_Folder_Explorer"

STAGE_OUT.mkdir(parents=True, exist_ok=True)

def file_info(path: Path):
    stat = path.stat()
    return {
        "path": str(path),
        "relative_path": str(path.relative_to(BASE_DIR)),
        "name": path.name,
        "parent": str(path.parent.relative_to(BASE_DIR)),
        "suffix": path.suffix.lower(),
        "size_bytes": stat.st_size,
        "size_mb": round(stat.st_size / (1024 * 1024), 4),
        "modified_time": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
    }

all_files = []
all_dirs = []

for root, dirs, files in os.walk(BASE_DIR):
    root_path = Path(root)

    for d in dirs:
        dpath = root_path / d
        all_dirs.append({
            "relative_path": str(dpath.relative_to(BASE_DIR)),
            "name": d,
            "parent": str(root_path.relative_to(BASE_DIR)) if root_path != BASE_DIR else "."
        })

    for f in files:
        fpath = root_path / f
        try:
            all_files.append(file_info(fpath))
        except Exception as e:
            all_files.append({
                "path": str(fpath),
                "relative_path": str(fpath.relative_to(BASE_DIR)),
                "name": f,
                "error": str(e)
            })

files_df = pd.DataFrame(all_files)
dirs_df = pd.DataFrame(all_dirs)

# Summary by extension
ext_summary = (
    files_df.groupby("suffix", dropna=False)
    .agg(
        file_count=("name", "count"),
        total_size_mb=("size_mb", "sum")
    )
    .reset_index()
    .sort_values(["file_count", "total_size_mb"], ascending=False)
)

# Summary by top-level folder
files_df["top_level_folder"] = files_df["relative_path"].apply(lambda x: x.split("\\")[0] if "\\" in x else ".")
folder_summary = (
    files_df.groupby("top_level_folder")
    .agg(
        file_count=("name", "count"),
        total_size_mb=("size_mb", "sum")
    )
    .reset_index()
    .sort_values("total_size_mb", ascending=False)
)

# Important files
important_keywords = [
    "stage", "report", "summary", "benchmark", "feature", "metadata",
    "inventory", "statistical", "novelty", "paper", "idea", "result"
]

important_df = files_df[
    files_df["name"].str.lower().apply(
        lambda x: any(k in x for k in important_keywords)
    )
].copy()

important_df = important_df.sort_values("modified_time", ascending=False)

# Save outputs
files_df.to_csv(STAGE_OUT / "all_project_files.csv", index=False)
dirs_df.to_csv(STAGE_OUT / "all_project_directories.csv", index=False)
ext_summary.to_csv(STAGE_OUT / "file_extension_summary.csv", index=False)
folder_summary.to_csv(STAGE_OUT / "top_level_folder_summary.csv", index=False)
important_df.to_csv(STAGE_OUT / "important_project_files.csv", index=False)

summary = {
    "base_dir": str(BASE_DIR),
    "total_directories": len(dirs_df),
    "total_files": len(files_df),
    "total_size_mb": round(files_df["size_mb"].sum(), 3),
    "outputs_saved_to": str(STAGE_OUT),
    "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

with open(STAGE_OUT / "Stage0_Project_Folder_Explorer_Summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4, ensure_ascii=False)

# Markdown report
report = []
report.append("# Stage 0 Project Folder Explorer Report\n")
report.append(f"Generated at: {summary['generated_at']}\n")
report.append(f"Base folder: `{BASE_DIR}`\n")
report.append("## Overall Summary\n")
report.append(f"- Total directories: {summary['total_directories']}")
report.append(f"- Total files: {summary['total_files']}")
report.append(f"- Total size: {summary['total_size_mb']} MB\n")

report.append("## Output Files\n")
report.append("- `all_project_files.csv`")
report.append("- `all_project_directories.csv`")
report.append("- `file_extension_summary.csv`")
report.append("- `top_level_folder_summary.csv`")
report.append("- `important_project_files.csv`")
report.append("- `Stage0_Project_Folder_Explorer_Summary.json`\n")

report.append("## Purpose\n")
report.append(
    "This exploratory stage provides a complete inventory of the Anemia_Paper project folder "
    "before merging the conventional multimodal anemia-classification manuscript with the "
    "new cooperative physiological-intelligence framework."
)

with open(STAGE_OUT / "Stage0_Project_Folder_Explorer_Report.md", "w", encoding="utf-8") as f:
    f.write("\n".join(report))

print("=" * 80)
print("STAGE 0 PROJECT FOLDER EXPLORER COMPLETED")
print("=" * 80)
print(f"Base folder: {BASE_DIR}")
print(f"Total directories: {summary['total_directories']}")
print(f"Total files: {summary['total_files']}")
print(f"Total size: {summary['total_size_mb']} MB")
print(f"Outputs saved to: {STAGE_OUT}")
print("=" * 80)