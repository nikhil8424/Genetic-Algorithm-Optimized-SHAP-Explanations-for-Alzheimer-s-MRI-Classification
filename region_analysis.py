import os
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import config


def calculate_region_shap_scores(
    abs_shap_map: np.ndarray,
    grid_rows: int = config.GRID_ROWS,
    grid_cols: int = config.GRID_COLS,
    img_size: int = config.IMG_SIZE,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Divides 128x128 2D SHAP map into an 8x8 spatial grid (64 regions)."""
    abs_map = np.squeeze(abs_shap_map)
    region_h = img_size // grid_rows
    region_w = img_size // grid_cols

    region_scores_list = []
    records = []

    region_id = 0
    for r in range(grid_rows):
        for c in range(grid_cols):
            r_start, r_end = r * region_h, (r + 1) * region_h
            c_start, c_end = c * region_w, (c + 1) * region_w

            region_crop = abs_map[r_start:r_end, c_start:c_end]
            mean_score = float(np.mean(region_crop))
            region_scores_list.append(mean_score)

            records.append({
                "Region_ID": region_id,
                "Row": r,
                "Column": c,
                "SHAP_Score": mean_score,
            })
            region_id += 1

    region_scores = np.array(region_scores_list, dtype=np.float32)
    total_shap = float(np.sum(region_scores))

    if total_shap > 1e-9:
        norm_scores = region_scores / total_shap
    else:
        norm_scores = np.zeros_like(region_scores)

    for i, rec in enumerate(records):
        rec["Normalized_SHAP_Score"] = float(norm_scores[i])

    df_scores = pd.DataFrame(records)
    return region_scores, df_scores


def save_region_scores_csv(
    df_scores: pd.DataFrame,
    save_path: str = "results/region_scores.csv",
) -> None:
    """Saves region-level SHAP scores DataFrame to CSV."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_scores.to_csv(save_path, index=False)


def plot_region_grid(
    image: np.ndarray,
    grid_rows: int = config.GRID_ROWS,
    grid_cols: int = config.GRID_COLS,
    img_size: int = config.IMG_SIZE,
    save_path: str = "results/spatial_grid_overlay.png",
) -> None:
    """Displays the 8x8 spatial grid overlaid onto the 2D MRI slice."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img_2d = np.squeeze(image)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_2d, cmap="gray")

    region_h = img_size / grid_rows
    region_w = img_size / grid_cols

    for r in range(grid_rows + 1):
        ax.axhline(r * region_h - 0.5, color="#00ffcc", linestyle="--", linewidth=1.0, alpha=0.85)
    for c in range(grid_cols + 1):
        ax.axvline(c * region_w - 0.5, color="#00ffcc", linestyle="--", linewidth=1.0, alpha=0.85)

    for r in range(grid_rows):
        for c in range(grid_cols):
            idx = r * grid_cols + c
            ax.text(
                (c + 0.5) * region_w - 0.5,
                (r + 0.5) * region_h - 0.5,
                f"{idx}",
                color="#ffffff",
                fontsize=7,
                ha="center",
                va="center",
                weight="bold",
                bbox=dict(boxstyle="circle,pad=0.15", facecolor="#111111", alpha=0.6, edgecolor="none"),
            )

    ax.set_title("8x8 Spatial Grid Partition (64 Regions)", fontsize=12, fontweight="bold", pad=12)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_region_importance(
    region_scores: np.ndarray,
    grid_rows: int = config.GRID_ROWS,
    grid_cols: int = config.GRID_COLS,
    save_path: str = "results/region_importance_heatmap.png",
) -> None:
    """Renders 8x8 matrix of region importance scores."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    matrix = region_scores.reshape((grid_rows, grid_cols))

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        cbar=True,
        linewidths=0.5,
        linecolor="#333333",
    )
    plt.title("64-Region Aggregated SHAP Spatial Importance", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Spatial Column", fontsize=10)
    plt.ylabel("Spatial Row", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
