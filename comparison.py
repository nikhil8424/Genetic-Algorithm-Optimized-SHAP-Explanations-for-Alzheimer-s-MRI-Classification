import os
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import tensorflow as tf

import config
from genetic_algorithm import chromosome_to_mask, calculate_fitness, calculate_objectives
from xai_metrics import create_random_k_mask, evaluate_all_methods


def create_shap_top_k_mask(
    region_shap_scores: np.ndarray,
    k: int,
    grid_rows: int = config.GRID_ROWS,
    grid_cols: int = config.GRID_COLS,
    img_size: int = config.IMG_SIZE,
) -> Tuple[List[int], np.ndarray]:
    """
    Constructs 2D SHAP Top-K baseline chromosome and mask.

    This is a baseline method that selects the top K regions with the highest
    SHAP importance scores. It's a simple greedy approach: pick the regions
    that SHAP says are most important.

    This serves as a comparison baseline against the genetic algorithm, which
    considers multiple objectives simultaneously.

    Args:
        region_shap_scores: SHAP importance scores for each of the 64 regions
        k: Number of regions to select
        grid_rows: Number of rows in the grid (default: 8)
        grid_cols: Number of columns in the grid (default: 8)
        img_size: Size of the image (default: 128)

    Returns:
        Tuple of (chromosome, mask):
        - chromosome: Binary list of length 64 (1 = top-K region, 0 = other)
        - mask: Pixel-level binary mask (128, 128, 1)
    """
    total_regions = len(region_shap_scores)
    # Ensure k is valid (between 1 and total_regions)
    k = max(1, min(k, total_regions))
    # Get indices of top K regions (sorted by SHAP score descending)
    top_indices = set(np.argsort(region_shap_scores)[::-1][:k])
    # Create chromosome: 1 for top-K regions, 0 for others
    top_k_chromosome = [1 if i in top_indices else 0 for i in range(total_regions)]
    # Convert chromosome to pixel-level mask
    top_k_mask = chromosome_to_mask(top_k_chromosome, grid_rows, grid_cols, img_size)
    return top_k_chromosome, top_k_mask


def compare_all_methods(
    model: tf.keras.Model,
    image: np.ndarray,
    original_prob: float,
    ga_chromosome: List[int],
    region_shap_scores: np.ndarray,
    image_idx: int = 0,
    random_seed: int = config.RANDOM_SEED,
    save_path: str = "results/comparison.csv",
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Performs comparison between Random-K, SHAP Top-K, and GA-NSGA-II for 2D baseline.

    This function compares three methods for selecting important regions:
    1. Random-K: Select K regions randomly (baseline)
    2. SHAP Top-K: Select K regions with highest SHAP scores (greedy baseline)
    3. GA-NSGA-II: Use genetic algorithm to optimize multiple objectives (proposed method)

    All methods select the same number of regions (K) as the GA solution for fair comparison.

    Evaluation metrics:
    - Prediction Preservation: How well the model's prediction is maintained
    - SHAP Retention: How much SHAP importance is captured
    - Compactness: How sparse the selection is
    - Deletion AUC: How well prediction is maintained when regions are progressively removed
    - Insertion AUC: How quickly prediction recovers when regions are progressively added

    Args:
        model: Trained CNN model
        image: Original MRI image (128, 128, 1)
        original_prob: Model's prediction on full image
        ga_chromosome: Best chromosome from genetic algorithm
        region_shap_scores: SHAP scores for each region
        image_idx: Image index for labeling
        random_seed: Random seed for reproducibility
        save_path: Path to save comparison CSV

    Returns:
        Tuple of (comparison_df, results_dict):
        - comparison_df: DataFrame with comparison results
        - results_dict: Dictionary with detailed metrics for each method
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Use the comprehensive evaluation from xai_metrics
    results = evaluate_all_methods(
        model=model,
        image=image,
        original_prob=original_prob,
        region_shap_scores=region_shap_scores,
        ga_chromosome=ga_chromosome,
        random_seed=random_seed,
    )

    # Format results for CSV
    records = []
    for method_name, metrics in results.items():
        records.append({
            "Method": method_name,
            "Selected_Regions": metrics["k_selected"],
            "Original_Probability": original_prob,
            "Prediction_Preservation": metrics["prediction_preservation"],
            "SHAP_Retention": metrics["shap_retention"],
            "Compactness": metrics["compactness"],
            "Deletion_AUC": metrics["deletion_auc"],
            "Insertion_AUC": metrics["insertion_auc"],
        })

    comp_df = pd.DataFrame(records)
    comp_df.to_csv(save_path, index=False)
    return comp_df, results
