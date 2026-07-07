"""
Stage 12 Utility
Folder Structure Inventory

Purpose:
    Scan the complete Experiments directory and generate a
    reproducible inventory for GitHub repository construction.

Author:
    Basel Mahmoud

"""

from pathlib import Path
import json
import csv
from collections import Counter

# ==========================================================
# Configuration
# ==========================================================

ROOT = Path(r"D:\47\472\New-Papers\Anemia_Paper\Experiments")

OUTPUT_DIR = ROOT.parent / "Repository_Inventory"
OUTPUT_DIR.mkdir(exist_ok=True)

TXT_FILE = OUTPUT_DIR / "Folder_Structure.txt"
CSV_FILE = OUTPUT_DIR / "Folder_Structure.csv"
JSON_FILE = OUTPUT_DIR / "Folder_Structure.json"
STAT_FILE = OUTPUT_DIR / "Folder_Statistics.txt"

# ==========================================================
# Build tree
# ==========================================================

records = []
extension_counter = Counter()

def build_tree(folder, prefix=""):
    entries = sorted(folder.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))

    lines = []

    for i, item in enumerate(entries):
        last = i == len(entries) - 1
        connector = "└── " if last else "├── "

        line = prefix + connector + item.name
        lines.append(line)

        rel = item.relative_to(ROOT)

        records.append({
            "relative_path": str(rel),
            "type": "Folder" if item.is_dir() else "File",
            "parent": str(rel.parent),
            "name": item.name,
            "extension": item.suffix.lower() if item.is_file() else "",
            "size_bytes": item.stat().st_size if item.is_file() else "",
        })

        if item.is_file():
            extension_counter[item.suffix.lower()] += 1

        if item.is_dir():
            extension = "    " if last else "│   "
            lines.extend(build_tree(item, prefix + extension))

    return lines

tree_lines = [ROOT.name]
tree_lines.extend(build_tree(ROOT))

# ==========================================================
# Save TXT tree
# ==========================================================

with open(TXT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(tree_lines))

# ==========================================================
# Save CSV
# ==========================================================

with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "relative_path",
            "type",
            "parent",
            "name",
            "extension",
            "size_bytes"
        ]
    )
    writer.writeheader()
    writer.writerows(records)

# ==========================================================
# Save JSON
# ==========================================================

with open(JSON_FILE, "w", encoding="utf-8") as f:
    json.dump(records, f, indent=4)

# ==========================================================
# Statistics
# ==========================================================

folders = sum(r["type"] == "Folder" for r in records)
files = sum(r["type"] == "File" for r in records)

with open(STAT_FILE, "w", encoding="utf-8") as f:

    f.write("Repository Inventory Summary\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Root Folder : {ROOT}\n")
    f.write(f"Folders     : {folders}\n")
    f.write(f"Files       : {files}\n\n")

    f.write("File Types\n")
    f.write("-" * 40 + "\n")

    for ext, count in sorted(extension_counter.items()):
        ext_name = ext if ext else "[No Extension]"
        f.write(f"{ext_name:12s} : {count}\n")

print("=" * 70)
print("Repository Inventory Generated")
print("=" * 70)
print("Tree       :", TXT_FILE)
print("CSV        :", CSV_FILE)
print("JSON       :", JSON_FILE)
print("Statistics :", STAT_FILE)
print("=" * 70)