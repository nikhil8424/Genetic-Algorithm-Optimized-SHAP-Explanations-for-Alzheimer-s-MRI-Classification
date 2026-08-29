import os
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import tensorflow as tf

import config
from genetic_algorithm import chromosome_to_mask, calculate_fitness, calculate_objectives


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


def compare_ga_vs_shap(
    model: tf.keras.Model,
    image: np.ndarray,
    original_prob: float,
    ga_chromosome: List[int],
    region_shap_scores: np.ndarray,
    image_idx: int = 0,
    save_path: str = "results/comparison.csv",
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """Performs comparison between GA-Optimized mask and SHAP Top-K for 2D baseline."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    k_selected = sum(ga_chromosome)
    
    ga_fitness, ga_details = calculate_fitness(
        ga_chromosome, model=model, image=image, original_prob=original_prob, region_shap_scores=region_shap_scores
    )

    shap_chromosome, _ = create_shap_top_k_mask(region_shap_scores, k=k_selected)
    shap_fitness, shap_details = calculate_fitness(
        shap_chromosome, model=model, image=image, original_prob=original_prob, region_shap_scores=region_shap_scores
    )

    records = [
        {
            "Method": "SHAP Top-K Baseline",
            "Selected_Regions": k_selected,
            "Original_Probability": original_prob,
            "Masked_Probability": shap_details["masked_prob"],
            "Prediction_Preservation": shap_details["prediction_preservation"],
            "SHAP_Importance": shap_details["shap_importance"],
            "Sparsity": shap_details["sparsity_penalty"],
            "Fitness_Score": shap_fitness,
        },
        {
            "Method": "GA-Optimized Mask",
            "Selected_Regions": k_selected,
            "Original_Probability": original_prob,
            "Masked_Probability": ga_details["masked_prob"],
            "Prediction_Preservation": ga_details["prediction_preservation"],
            "SHAP_Importance": ga_details["shap_importance"],
            "Sparsity": ga_details["sparsity_penalty"],
            "Fitness_Score": ga_fitness,
        },
    ]

    comp_df = pd.DataFrame(records)
    comp_df.to_csv(save_path, index=False)
    return comp_df, {"SHAP_TopK": shap_details, "GA_Optimized": ga_details}
