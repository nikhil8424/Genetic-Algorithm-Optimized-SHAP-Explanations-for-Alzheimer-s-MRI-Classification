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
    """Constructs 2D SHAP Top-K baseline chromosome and mask."""
    total_regions = len(region_shap_scores)
    k = max(1, min(k, total_regions))
    top_indices = set(np.argsort(region_shap_scores)[::-1][:k])
    top_k_chromosome = [1 if i in top_indices else 0 for i in range(total_regions)]
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
    """Performs comparison between Random-K, SHAP Top-K, and GA-NSGA-II for 2D baseline."""
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
