# -*- coding: utf-8 -*-
"""
Habitat analysis
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from radiomics import featureextractor
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# Config
BASE_DIR = Path(".")
IMAGES_DIR = BASE_DIR / "images"
SEGM_DIR = BASE_DIR / "segmentation"
ENTROPY_DIR = BASE_DIR / "entropy"
HABITAT_SEGM_DIR = BASE_DIR / "habitat_mask"
CLUSTER_PDF = BASE_DIR / "clustering_analysis.pdf"
LABELS_COUNT_XLSX = BASE_DIR / "labels_count.xlsx"
SET_YAML = BASE_DIR / "set.yaml"

N_CLUSTERS = 4
RANDOM_STATE = 99

os.makedirs(ENTROPY_DIR, exist_ok=True)
os.makedirs(HABITAT_SEGM_DIR, exist_ok=True)

def _infer_id(seg_name: str) -> str:
    stem = Path(seg_name).stem
    return stem[:-6] if stem.endswith("_segm") else stem

def extract_entropy_for_all(segm_dir=SEGM_DIR, images_dir=IMAGES_DIR, entropy_dir=ENTROPY_DIR, yaml_path=SET_YAML):
    seg_files = sorted([f for f in os.listdir(segm_dir) if not f.startswith('.')])
    if not seg_files:
        raise FileNotFoundError(f"No segmentation files in {segm_dir}")
    extractor = featureextractor.RadiomicsFeatureExtractor(str(yaml_path))
    for seg_name in seg_files:
        img_id = _infer_id(seg_name)
        img_path = images_dir / img_id / f"{img_id}_imag.nrrd"
        mask_path = Path(segm_dir) / seg_name
        out_entropy = Path(entropy_dir) / f"{img_id}_entropy.nrrd"
        if not (img_path.exists() and mask_path.exists()):
            continue
        try:
            feats = extractor.execute(str(img_path), str(mask_path), voxelBased=True)
            key = next((k for k in feats.keys() if "entropy" in k.lower()), None)
            if key is None:
                continue
            ent_img = feats[key]
            if isinstance(ent_img, sitk.Image):
                sitk.WriteImage(ent_img, str(out_entropy))
        except Exception:
            continue

def build_couples(segm_dir=SEGM_DIR, images_dir=IMAGES_DIR, entropy_dir=ENTROPY_DIR):
    """Collect gray/entropy pairs inside mask==1 for all samples."""
    seg_files = sorted([f for f in os.listdir(segm_dir) if not f.startswith('.')])
    rows = []
    filecount = 0
    for seg_name in seg_files:
        img_id = _infer_id(seg_name)
        img_path = images_dir / img_id / f"{img_id}_imag.nrrd"
        mask_path = Path(segm_dir) / seg_name
        ent_path = Path(entropy_dir) / f"{img_id}_entropy.nrrd"
        if not (img_path.exists() and mask_path.exists() and ent_path.exists()):
            continue
        try:
            img_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(img_path)))
            mask_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path)))
            ent_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(ent_path)))
        except Exception:
            continue
        gray_vals = img_arr[mask_arr == 1]
        ent_vals = ent_arr[mask_arr == 1]
        finite = np.isfinite(ent_vals)
        gray_vals = gray_vals[finite]
        ent_vals = ent_vals[finite]
        if gray_vals.size == 0:
            continue
        filecount += 1
        df = pd.DataFrame({
            "gray": gray_vals,
            "entropy": ent_vals,
            "filecount": filecount,
            "SEG": seg_name
        })
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["gray", "entropy", "filecount", "SEG"])

def cluster_and_write_habitats(couples_df, n_clusters=N_CLUSTERS, out_dir=HABITAT_SEGM_DIR, segm_dir=SEGM_DIR):
    """Cluster (gray, entropy) and write habitat masks per sample."""
    if couples_df.empty:
        raise ValueError("No data for clustering.")
    scaler = MinMaxScaler()
    X = scaler.fit_transform(couples_df[["gray", "entropy"]])
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE)
    labels = kmeans.fit_predict(X) + 1  
    df = couples_df.copy()
    df["labels"] = labels

    seg_files = sorted([f for f in os.listdir(segm_dir) if not f.startswith('.')])
    for seg_name in seg_files:
        img_id = _infer_id(seg_name)
        mask_path = Path(segm_dir) / seg_name
        try:
            ori_img = sitk.ReadImage(str(mask_path))
            ori_arr = sitk.GetArrayFromImage(ori_img)
        except Exception:
            continue
        sel = df[df["SEG"] == seg_name]
        if sel.empty:
            continue
        lab_vals = sel["labels"].values
        mask_pos = (ori_arr == 1)
        total = int(mask_pos.sum())
        if total == 0:
            continue
        min_len = min(total, len(lab_vals))
        idx_flat = np.where(mask_pos.flatten())[0]
        flat_new = ori_arr.flatten().astype(np.int16)
        flat_new[idx_flat[:min_len]] = lab_vals[:min_len]
        new_arr = flat_new.reshape(ori_arr.shape)

        # full habitat image
        new_img = sitk.GetImageFromArray(new_arr.astype(np.int16))
        new_img.SetOrigin(ori_img.GetOrigin())
        new_img.SetDirection(ori_img.GetDirection())
        new_img.SetSpacing(ori_img.GetSpacing())
        sitk.WriteImage(new_img, str(Path(out_dir) / f"{img_id}_habitat.nrrd"))

        # per-cluster binary masks
        folder = Path(out_dir) / img_id
        folder.mkdir(parents=True, exist_ok=True)
        for lbl in range(1, n_clusters + 1):
            cluster_arr = (new_arr == lbl).astype(np.uint8)
            if cluster_arr.sum() == 0:
                continue
            cimg = sitk.GetImageFromArray(cluster_arr)
            cimg.SetOrigin(ori_img.GetOrigin())
            cimg.SetDirection(ori_img.GetDirection())
            cimg.SetSpacing(ori_img.GetSpacing())
            sitk.WriteImage(cimg, str(folder / f"{img_id}_reg{lbl}_habitat.nrrd"))
    return df

def export_counts_and_plot(couples_labeled_df, out_xlsx=LABELS_COUNT_XLSX, pdf_out=CLUSTER_PDF, n_clusters=N_CLUSTERS):
    """Save per-sample cluster counts and a before/after clustering scatter PDF."""
    if couples_labeled_df.empty:
        raise ValueError("Empty labeled data.")
    table = pd.crosstab(couples_labeled_df["SEG"], couples_labeled_df["labels"])
    table.to_excel(out_xlsx)

    scaler = MinMaxScaler()
    mm = scaler.fit_transform(couples_labeled_df[["gray", "entropy"]])
    labels0 = couples_labeled_df["labels"].values - 1
    colors = ['#F4A261', '#7209B7', '#E63946', '#2A9D8F'][:n_clusters]

    fig = plt.figure(figsize=(12, 5.5))
    gs = fig.add_gridspec(1, 2)
    ax0 = fig.add_subplot(gs[0])
    ax0.scatter(mm[:, 0], mm[:, 1], s=5, alpha=0.6, color='#457B9D')
    ax0.set_title("Before Clustering")
    ax0.set_xlabel("Normalized Gray")
    ax0.set_ylabel("Normalized Entropy")

    ax1 = fig.add_subplot(gs[1])
    for i, c in enumerate(colors):
        m = (labels0 == i)
        ax1.scatter(mm[m, 0], mm[m, 1], c=c, s=5, alpha=0.6, label=f"Cluster {i+1}")
    ax1.set_title("After Clustering")
    ax1.set_xlabel("Normalized Gray")
    ax1.legend(loc='lower right')

    plt.tight_layout()
    with PdfPages(pdf_out) as pdf:
        pdf.savefig(fig, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return table

def main():
    extract_entropy_for_all()
    couples = build_couples()
    if couples.empty:
        print("No data assembled. Exiting.")
        return
    labeled = cluster_and_write_habitats(couples)
    table = export_counts_and_plot(labeled)
    print("Saved counts to:", LABELS_COUNT_XLSX.resolve())
    print("Saved clustering PDF to:", Path(CLUSTER_PDF).resolve())

if __name__ == "__main__":
    main()