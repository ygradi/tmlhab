# -*- coding: utf-8 -*-
"""
Friedman-Nemenyi analysis & CD Diagram
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# config
output_dir = r"."
os.makedirs(output_dir, exist_ok=True)
pdf_path = os.path.join(output_dir, "Test_Set_CD_Diagram.pdf")
file_path = r"data.xlsx"

model_cols = ['Model1', 'Model2', 'Model3', 'Model4', 'Model5', 'Model6']
model_names = [f"Model {i}" for i in range(1, 7)]
true_col = "HER23"

n_bootstrap = 50
random_seed = 42

# metrics
def calculate_metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score, f1_score, precision_score
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, average="macro"),
        "Precision": precision_score(y_true, y_pred, average="macro")
    }

# bootstrap
def bootstrap_metrics(data_df, model_cols, true_col, n_bootstrap=50, random_state=42):
    np.random.seed(random_state)
    n_samples = len(data_df)
    metrics = ["Accuracy", "F1", "Precision"]
    out = {m: [] for m in metrics}
    for _ in range(n_bootstrap):
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        boot_df = data_df.iloc[idx]
        y_true = boot_df[true_col].values
        per_boot = {m: [] for m in metrics}
        for model in model_cols:
            y_pred = boot_df[model].values
            met = calculate_metrics(y_true, y_pred)
            for k in metrics:
                per_boot[k].append(met[k])
        for k in metrics:
            out[k].append(per_boot[k])
    for k in out:
        out[k] = np.array(out[k])  
    return out

# Friedman + Nemenyi
def friedman_nemenyi_test(data_matrix, model_names):
    N, k = data_matrix.shape
    ranks = np.zeros_like(data_matrix, dtype=float)
    for i in range(N):
        ranks[i, :] = stats.rankdata(-data_matrix[i, :]) 
    avg_ranks = ranks.mean(axis=0)
    try:
        stat, p_value = stats.friedmanchisquare(*[data_matrix[:, j] for j in range(k)])
    except Exception:
        stat, p_value = np.nan, np.nan
    q_alpha_dict = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
                    7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
    q_alpha = q_alpha_dict.get(k, 2.850)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * N))
    return {"avg_ranks": avg_ranks, "friedman_stat": stat, "p_value": p_value, "CD": cd, "model_names": model_names}

# plot stacked CD diagrams
def plot_all_metrics_cd(results_dict, figsize=(10, 12)):
    metrics = list(results_dict.keys())
    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=figsize)
    if n == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        res = results_dict[metric]
        avg_ranks = res["avg_ranks"]
        cd = res["CD"]
        names = res["model_names"]
        pval = res["p_value"]
        k = len(names)

        order = np.argsort(avg_ranks)
        sorted_ranks = avg_ranks[order]
        sorted_names = [names[i] for i in order]

        x_min = max(0.5, sorted_ranks.min() - 0.5)
        x_max = sorted_ranks.max() + 0.5

        ax.hlines(0.5, x_min, x_max, color="black", linewidth=1)
        ticks = np.arange(1, k + 1)
        for t in ticks:
            ax.vlines(t, 0.45, 0.55, color="black", linewidth=0.8)
            ax.text(t, 0.35, str(t), ha="center", va="top", fontsize=9, fontweight="bold")

        for idx, (r, nm) in enumerate(zip(sorted_ranks, sorted_names)):
            y = 0.62 if idx % 2 == 0 else 0.38
            ax.plot(r, y, marker="o", color="tab:blue")
            ax.text(r, y + (0.03 if idx % 2 == 0 else -0.05),
                    f"{nm} ({r:.2f})", ha="center",
                    va="bottom" if idx % 2 == 0 else "top", fontsize=9)

        cd_start = x_min + 0.1
        ax.hlines(0.85, cd_start, cd_start + cd, color="red", linewidth=3)
        ax.vlines([cd_start, cd_start + cd], [0.82, 0.82], [0.88, 0.88], color="red", linewidth=1.5)
        ax.text(cd_start + cd / 2, 0.92, f"CD={cd:.2f}", ha="center", va="bottom", fontsize=10, color="red", fontweight="bold")

        y_link = 0.56
        for i in range(len(sorted_ranks)):
            for j in range(i + 1, len(sorted_ranks)):
                if abs(sorted_ranks[j] - sorted_ranks[i]) < cd:
                    ax.hlines(y_link, sorted_ranks[i], sorted_ranks[j], colors="gray", linewidth=4, alpha=0.6)
                    y_link += 0.02

        sig_mark = "**" if (not np.isnan(pval) and pval < 0.01) else ("*" if (not np.isnan(pval) and pval < 0.05) else "ns")
        ax.set_title(f"{metric} (p={pval:.4f}) {sig_mark}", loc="left", fontsize=11, fontweight="bold")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, 1)
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.suptitle("Test Set - CD Diagrams", fontsize=12, fontweight="bold", y=0.995)
    return fig

# main
if __name__ == "__main__":
    df = pd.read_excel(file_path)
    test_df = df[df["tt"] == 0].copy()
    if test_df.empty:
        raise ValueError("No test-set rows (tt==0).")

    boot_results = bootstrap_metrics(test_df, model_cols, true_col=true_col, n_bootstrap=n_bootstrap, random_state=random_seed)

    metrics_to_analyze = ["Accuracy", "F1", "Precision"]
    all_results = {}
    for metric in metrics_to_analyze:
        arr = boot_results[metric]
        res = friedman_nemenyi_test(arr, model_names)
        all_results[metric] = res

    summary_rows = []
    for metric in metrics_to_analyze:
        res = all_results[metric]
        best_idx = int(np.argmin(res["avg_ranks"]))
        summary_rows.append({
            "Metric": metric,
            "Best Model": res["model_names"][best_idx],
            "Avg Rank": float(res["avg_ranks"][best_idx]),
            "p-value": float(res["p_value"]),
            "Significant": ("Yes" if (not np.isnan(res["p_value"]) and res["p_value"] < 0.05) else "No")
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_excel = os.path.join(output_dir, "Test_Set_Performance.xlsx")
    summary_df.to_excel(summary_excel, index=False)

    with PdfPages(pdf_path) as pdf:
        fig = plot_all_metrics_cd(all_results, figsize=(10, 12))
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)