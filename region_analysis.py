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
    """
    Divides 128x128 2D SHAP map into an 8x8 spatial grid (64 regions).

    This function partitions the SHAP importance map into a grid of spatial regions.
    Each region (16x16 pixels) is assigned a mean SHAP score representing the
    average importance of that spatial area.

    This is a key step for the genetic algorithm, which operates on these 64 regions
    rather than individual pixels (reducing complexity from 2^16,384 to 2^64).

    Args:
        abs_shap_map: Absolute SHAP values (128, 128)
        grid_rows: Number of rows in the grid (default: 8)
        grid_cols: Number of columns in the grid (default: 8)
        img_size: Size of the image (default: 128)

    Returns:
        Tuple of (region_scores, df_scores):
        - region_scores: Array of 64 mean SHAP scores (one per region)
        - df_scores: DataFrame with Region_ID, Row, Column, SHAP_Score, Normalized_SHAP_Score
    """
    # Remove channel dimension if present
    abs_map = np.squeeze(abs_shap_map)
    # Calculate region dimensions (16x16 pixels for 128x128 image with 8x8 grid)
    region_h = img_size // grid_rows
    region_w = img_size // grid_cols

    region_scores_list = []
    records = []

    # Iterate through each region in the grid
    region_id = 0
    for r in range(grid_rows):
        for c in range(grid_cols):
            # Calculate region boundaries
            r_start, r_end = r * region_h, (r + 1) * region_h
            c_start, c_end = c * region_w, (c + 1) * region_w

            # Extract region from SHAP map
            region_crop = abs_map[r_start:r_end, c_start:c_end]
            # Compute mean SHAP score for this region
            mean_score = float(np.mean(region_crop))
            region_scores_list.append(mean_score)

            # Record region information
            records.append({
                "Region_ID": region_id,      # 0-63
                "Row": r,                    # 0-7
                "Column": c,                 # 0-7
                "SHAP_Score": mean_score,    # Mean absolute SHAP value
            })
            region_id += 1

    # Convert to numpy array
    region_scores = np.array(region_scores_list, dtype=np.float32)
    # Normalize scores to sum to 1 (represents percentage of total importance)
    total_shap = float(np.sum(region_scores))

    if total_shap > 1e-9:
        norm_scores = region_scores / total_shap
    else:
        norm_scores = np.zeros_like(region_scores)

    # Add normalized scores to records
    for i, rec in enumerate(records):
        rec["Normalized_SHAP_Score"] = float(norm_scores[i])

    df_scores = pd.DataFrame(records)
    return region_scores, df_scores


def save_region_scores_csv(
    df_scores: pd.DataFrame,
    save_path: str = "results/region_scores.csv",
) -> None:
    """
    Saves region-level SHAP scores DataFrame to CSV.

    This function saves the computed region scores to a CSV file for analysis
    and record-keeping. The CSV contains:
    - Region_ID: 0-63 identifier for each region
    - Row, Column: Grid coordinates
    - SHAP_Score: Mean absolute SHAP value
    - Normalized_SHAP_Score: Normalized to sum to 1

    Args:
        df_scores: DataFrame with region information
        save_path: Path to save the CSV file
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_scores.to_csv(save_path, index=False)


def plot_region_grid(
    image: np.ndarray,
    grid_rows: int = config.GRID_ROWS,
    grid_cols: int = config.GRID_COLS,
    img_size: int = config.IMG_SIZE,
    save_path: str = "results/spatial_grid_overlay.png",
) -> None:
    """
    Displays the 8x8 spatial grid overlaid onto the 2D MRI slice.

    This function visualizes how the image is partitioned into 64 regions.
    Each region is labeled with its ID (0-63) and bounded by grid lines.
    This helps understand the spatial structure used by the genetic algorithm.

    Args:
        image: Original MRI image (128, 128, 1)
        grid_rows: Number of rows in the grid (default: 8)
        grid_cols: Number of columns in the grid (default: 8)
        img_size: Size of the image (default: 128)
        save_path: Path to save the visualization
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img_2d = np.squeeze(image)

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_2d, cmap="gray")

    # Calculate region dimensions
    region_h = img_size / grid_rows
    region_w = img_size / grid_cols

    # Draw horizontal grid lines
    for r in range(grid_rows + 1):
        ax.axhline(r * region_h - 0.5, color="#00ffcc", linestyle="--", linewidth=1.0, alpha=0.85)
    # Draw vertical grid lines
    for c in range(grid_cols + 1):
        ax.axvline(c * region_w - 0.5, color="#00ffcc", linestyle="--", linewidth=1.0, alpha=0.85)

    # Label each region with its ID
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
    """
    Renders 8x8 matrix of region importance scores.

    This function creates a heatmap visualization showing the SHAP importance
    of each region in the 8x8 grid. Brighter colors indicate higher importance.
    This helps identify which spatial regions are most influential for the
    model's prediction.

    Args:
        region_scores: Array of 64 SHAP scores (one per region)
        grid_rows: Number of rows in the grid (default: 8)
        grid_cols: Number of columns in the grid (default: 8)
        save_path: Path to save the heatmap
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Reshape 1D array to 2D grid (8x8)
    matrix = region_scores.reshape((grid_rows, grid_cols))

    # Create heatmap
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        matrix,
        annot=True,          # Show numeric values in cells
        fmt=".3f",          # 3 decimal places
        cmap="YlOrRd",       # Yellow-Orange-Red colormap
        cbar=True,           # Show color bar
        linewidths=0.5,      # Grid line width
        linecolor="#333333", # Grid line color
    )
    plt.title("64-Region Aggregated SHAP Spatial Importance", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Spatial Column", fontsize=10)
    plt.ylabel("Spatial Row", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
