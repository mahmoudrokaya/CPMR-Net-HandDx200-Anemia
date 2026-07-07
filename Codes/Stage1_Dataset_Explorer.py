import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from PIL import Image

# ============================================================
# PATHS
# ============================================================

DATA_DIR = r"D:\47\472\New-Papers\Anemia_Paper\Data"

OUTPUT_DIR = r"D:\47\472\New-Papers\Anemia_Paper\Outputs\Stage1_Dataset_Explorer"

FIGURES_DIR = os.path.join(OUTPUT_DIR, "Figures")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ============================================================
# INVENTORY ALL FILES
# ============================================================

all_files = []

for root, dirs, files in os.walk(DATA_DIR):
    for file in files:

        filepath = os.path.join(root, file)

        all_files.append({
            "file_name": file,
            "extension": Path(file).suffix,
            "size_mb": round(os.path.getsize(filepath)/(1024*1024),4),
            "folder": root
        })

inventory_df = pd.DataFrame(all_files)

inventory_df.to_csv(
    os.path.join(OUTPUT_DIR,
                 "dataset_file_inventory.csv"),
    index=False
)

# ============================================================
# FIND TABULAR FILES
# ============================================================

metadata_candidates = []

for ext in [".csv", ".xlsx", ".xls"]:
    metadata_candidates.extend(
        inventory_df[
            inventory_df["extension"].str.lower()==ext
        ]["file_name"].tolist()
    )

# ============================================================
# IMAGE AUDIT
# ============================================================

image_extensions = [
    ".png",".jpg",".jpeg",
    ".bmp",".tif",".tiff"
]

image_stats = []

for root, dirs, files in os.walk(DATA_DIR):

    for file in files:

        ext = Path(file).suffix.lower()

        if ext in image_extensions:

            try:

                image_path = os.path.join(root,file)

                img = Image.open(image_path)

                width,height = img.size

                image_stats.append({
                    "file":file,
                    "width":width,
                    "height":height,
                    "mode":img.mode
                })

            except:
                pass

image_df = pd.DataFrame(image_stats)

if len(image_df):

    image_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "image_resolution_statistics.csv"
        ),
        index=False
    )

# ============================================================
# IMAGE RESOLUTION HISTOGRAM
# ============================================================

if len(image_df):

    plt.figure(figsize=(8,6))

    sns.histplot(
        image_df["width"],
        bins=30
    )

    plt.title("Image Width Distribution")

    plt.savefig(
        os.path.join(
            FIGURES_DIR,
            "image_resolution_distribution.png"
        )
    )

    plt.close()

# ============================================================
# PROCESS METADATA FILES
# ============================================================

for file in metadata_candidates:

    print(f"Found metadata file: {file}")

# User will identify the main metadata file after execution