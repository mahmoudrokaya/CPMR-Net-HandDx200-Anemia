from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
import imageio.v3 as iio

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATA_DIR = Path(r"D:\47\472\New-Papers\Anemia_Paper\Data")
OUT_DIR = Path(r"D:\47\472\New-Papers\Anemia_Paper\Outputs\Stage2_Thermal_BMP_Decoder")
CONVERTED_DIR = OUT_DIR / "converted_thermal_png"
TABLES_DIR = OUT_DIR / "tables"
FIGURES_DIR = OUT_DIR / "figures"

for d in [OUT_DIR, CONVERTED_DIR, TABLES_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

thermal_files = list((DATA_DIR / "Anemia" / "Thermal").rglob("*.bmp")) + \
                list((DATA_DIR / "Normal" / "Thermal").rglob("*.bmp"))

records = []

def normalize_to_uint8(arr):
    arr = np.asarray(arr)

    if arr.ndim == 3:
        arr = arr[:, :, 0]

    arr = arr.astype(np.float32)

    finite = np.isfinite(arr)
    if finite.sum() == 0:
        return None

    lo, hi = np.percentile(arr[finite], [1, 99])

    if hi <= lo:
        hi = arr[finite].max()
        lo = arr[finite].min()

    if hi <= lo:
        return None

    arr = np.clip((arr - lo) / (hi - lo), 0, 1)
    return (arr * 255).astype(np.uint8)

def read_thermal(path):
    errors = []

    try:
        img = Image.open(path)
        arr = np.array(img)
        return arr, "PIL"
    except Exception as e:
        errors.append(f"PIL: {e}")

    try:
        arr = iio.imread(path)
        return arr, "imageio"
    except Exception as e:
        errors.append(f"imageio: {e}")

    return None, " | ".join(errors)

for path in thermal_files:
    label = path.parts[-3]   # Anemia or Normal

    arr, reader = read_thermal(path)

    if arr is None:
        records.append({
            "file_name": path.name,
            "label": label,
            "path": str(path),
            "decoded": False,
            "reader": reader,
            "width": None,
            "height": None,
            "dtype": None,
            "min_value": None,
            "max_value": None,
            "mean_value": None,
            "std_value": None,
            "converted_png": None
        })
        continue

    norm = normalize_to_uint8(arr)

    if norm is None:
        decoded = False
        converted_path = None
    else:
        decoded = True
        converted_path = CONVERTED_DIR / f"{path.stem}_thermal_norm.png"
        Image.fromarray(norm).save(converted_path)

    arr_np = np.asarray(arr)

    records.append({
        "file_name": path.name,
        "label": label,
        "path": str(path),
        "decoded": decoded,
        "reader": reader,
        "width": arr_np.shape[1] if arr_np.ndim >= 2 else None,
        "height": arr_np.shape[0] if arr_np.ndim >= 2 else None,
        "dtype": str(arr_np.dtype),
        "min_value": float(np.nanmin(arr_np)),
        "max_value": float(np.nanmax(arr_np)),
        "mean_value": float(np.nanmean(arr_np)),
        "std_value": float(np.nanstd(arr_np)),
        "converted_png": str(converted_path) if converted_path else None
    })

df = pd.DataFrame(records)
df.to_csv(TABLES_DIR / "thermal_bmp_decoding_audit.csv", index=False)

summary = df.groupby(["label", "decoded"]).size().reset_index(name="count")
summary.to_csv(TABLES_DIR / "thermal_decoding_summary.csv", index=False)

decoded_df = df[df["decoded"] == True]

if len(decoded_df) > 0:
    plt.figure(figsize=(7, 5))
    decoded_df["mean_value"].hist(bins=40)
    plt.title("Thermal Mean Value Distribution")
    plt.xlabel("Mean thermal intensity")
    plt.ylabel("Image count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "thermal_mean_value_distribution.png", dpi=300)
    plt.close()

    sample_paths = decoded_df["converted_png"].dropna().head(16).tolist()

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    axes = axes.flatten()

    for ax, img_path in zip(axes, sample_paths):
        img = Image.open(img_path)
        ax.imshow(img, cmap="gray")
        ax.axis("off")

    for ax in axes[len(sample_paths):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "thermal_sample_contact_sheet.png", dpi=300)
    plt.close()

with open(OUT_DIR / "Stage2_Thermal_BMP_Decoder_Report.md", "w", encoding="utf-8") as f:
    f.write("# Stage 2 Thermal BMP Decoder Report\n\n")
    f.write(f"Total thermal BMP files found: {len(df)}\n\n")
    f.write(f"Successfully decoded: {int(df['decoded'].sum())}\n\n")
    f.write(f"Failed decoding: {int((~df['decoded']).sum())}\n\n")
    f.write("The decoded thermal files were normalized to 8-bit grayscale PNG files for downstream machine learning preprocessing. Original BMP files were preserved.\n")

print("=" * 80)
print("STAGE 2 THERMAL BMP DECODING COMPLETED")
print("=" * 80)
print(f"Thermal files found: {len(df)}")
print(f"Decoded successfully: {int(df['decoded'].sum())}")
print(f"Failed: {int((~df['decoded']).sum())}")
print(f"Results saved to: {OUT_DIR}")
print("=" * 80)