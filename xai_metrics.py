import os
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import auc

import config
from genetic_algorithm import chromosome_to_mask, calculate_objectives


def create_random_k_mask(
    k: int,
    total_regions: int = config.NUM_REGIONS,
    grid_rows: int = config.GRID_ROWS,
    grid_cols: int = config.GRID_COLS,
    img_size: int = config.IMG_SIZE,
    random_seed: int = config.RANDOM_SEED,
) -> Tuple[List[int], np.ndarray]:
    """Constructs Random-K baseline chromosome and mask."""
    np.random.seed(random_seed)
    k = max(1, min(k, total_regions))
    random_indices = set(np.random.choice(total_regions, size=k, replace=False))
    random_k_chromosome = [1 if i in random_indices else 0 for i in range(total_regions)]
    random_k_mask = chromosome_to_mask(random_k_chromosome, grid_rows, grid_cols, img_size)
    return random_k_chromosome, random_k_mask


def compute_deletion_auc(
    model: tf.keras.Model,
    image: np.ndarray,
    selected_region_indices: List[int],
    original_prob: float,
    region_shap_scores: np.ndarray,
    grid_rows: int = config.GRID_ROWS,
    grid_cols: int = config.GRID_COLS,
    img_size: int = config.IMG_SIZE,
) -> float:
    """
    Computes Deletion AUC by progressively removing selected regions.
    Higher AUC indicates better preservation of prediction when important regions are removed.
    Optimized to use fewer evaluation steps for faster computation.
    """
    if not selected_region_indices:
        return 0.0

    img_3d = image if image.ndim == 3 else np.expand_dims(image, axis=-1)
    k = len(selected_region_indices)

    # Sort regions by SHAP importance (descending) for deletion
    region_importance = [(idx, region_shap_scores[idx]) for idx in selected_region_indices]
    region_importance.sort(key=lambda x: x[1], reverse=True)

    # Use fewer evaluation steps (max 16 instead of k) for speed
    max_steps = min(16, k)
    step_size = max(1, k // max_steps)

    probs = [original_prob]
    current_mask = np.ones((img_size, img_size, 1), dtype=np.float32)

    for step in range(max_steps):
        # Remove multiple regions at once
        start_idx = step * step_size
        end_idx = min((step + 1) * step_size, k)

        for region_idx, _ in region_importance[start_idx:end_idx]:
            row = region_idx // grid_cols
            col = region_idx % grid_cols
            region_h = img_size // grid_rows
            region_w = img_size // grid_cols
            current_mask[row*region_h:(row+1)*region_h, col*region_w:(col+1)*region_w, :] = 0.0

        masked_img = img_3d * current_mask
        batch_masked = np.expand_dims(masked_img, axis=0)
        masked_prob = float(model.predict(batch_masked, verbose=0)[0][0])
        probs.append(masked_prob)

    # Normalize AUC: compute area under curve and normalize by max possible
    x_points = np.linspace(0, 1, len(probs))
    deletion_auc_val = auc(x_points, probs)

    # Normalize by original probability (better explanation maintains higher probs)
    if original_prob > 0:
        normalized_auc = deletion_auc_val / original_prob
    else:
        normalized_auc = deletion_auc_val

    return float(np.clip(normalized_auc, 0.0, 1.0))


def compute_insertion_auc(
    model: tf.keras.Model,
    image: np.ndarray,
    selected_region_indices: List[int],
    original_prob: float,
    region_shap_scores: np.ndarray,
    grid_rows: int = config.GRID_ROWS,
    grid_cols: int = config.GRID_COLS,
    img_size: int = config.IMG_SIZE,
) -> float:
    """
    Computes Insertion AUC by progressively inserting selected regions.
    Higher AUC indicates faster recovery of prediction when important regions are added.
    Optimized to use fewer evaluation steps for faster computation.
    """
    if not selected_region_indices:
        return 0.0

    img_3d = image if image.ndim == 3 else np.expand_dims(image, axis=-1)
    k = len(selected_region_indices)

    # Sort regions by SHAP importance (descending) for insertion
    region_importance = [(idx, region_shap_scores[idx]) for idx in selected_region_indices]
    region_importance.sort(key=lambda x: x[1], reverse=True)

    # Use fewer evaluation steps (max 16 instead of k) for speed
    max_steps = min(16, k)
    step_size = max(1, k // max_steps)

    # Start with zero mask
    current_mask = np.zeros((img_size, img_size, 1), dtype=np.float32)
    masked_img = img_3d * current_mask
    batch_masked = np.expand_dims(masked_img, axis=0)
    start_prob = float(model.predict(batch_masked, verbose=0)[0][0])

    probs = [start_prob]

    for step in range(max_steps):
        # Add multiple regions at once
        start_idx = step * step_size
        end_idx = min((step + 1) * step_size, k)

        for region_idx, _ in region_importance[start_idx:end_idx]:
            row = region_idx // grid_cols
            col = region_idx % grid_cols
            region_h = img_size // grid_rows
            region_w = img_size // grid_cols
            current_mask[row*region_h:(row+1)*region_h, col*region_w:(col+1)*region_w, :] = 1.0

        masked_img = img_3d * current_mask
        batch_masked = np.expand_dims(masked_img, axis=0)
        masked_prob = float(model.predict(batch_masked, verbose=0)[0][0])
        probs.append(masked_prob)

    # Normalize AUC: compute area under curve
    x_points = np.linspace(0, 1, len(probs))
    insertion_auc_val = auc(x_points, probs)

    # Normalize by original probability (better explanation reaches original prob faster)
    if original_prob > 0:
        normalized_auc = insertion_auc_val / original_prob
    else:
        normalized_auc = insertion_auc_val

    return float(np.clip(normalized_auc, 0.0, 1.0))


def evaluate_all_methods(
    model: tf.keras.Model,
    image: np.ndarray,
    original_prob: float,
    region_shap_scores: np.ndarray,
    ga_chromosome: List[int],
    random_seed: int = config.RANDOM_SEED,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluates Random-K, SHAP Top-K, and GA-NSGA-II methods with all metrics.
    """
    k_selected = sum(ga_chromosome)
    selected_indices = [i for i, gene in enumerate(ga_chromosome) if gene == 1]

    # Random-K
    random_chromosome, _ = create_random_k_mask(k_selected, random_seed=random_seed)
    _, random_details = calculate_objectives(
        random_chromosome, model=model, image=image, original_prob=original_prob, region_shap_scores=region_shap_scores
    )
    random_indices = [i for i, gene in enumerate(random_chromosome) if gene == 1]
    random_deletion_auc = compute_deletion_auc(
        model, image, random_indices, original_prob, region_shap_scores
    )
    random_insertion_auc = compute_insertion_auc(
        model, image, random_indices, original_prob, region_shap_scores
    )

    # SHAP Top-K
    from comparison import create_shap_top_k_mask
    shap_chromosome, _ = create_shap_top_k_mask(region_shap_scores, k=k_selected)
    _, shap_details = calculate_objectives(
        shap_chromosome, model=model, image=image, original_prob=original_prob, region_shap_scores=region_shap_scores
    )
    shap_indices = [i for i, gene in enumerate(shap_chromosome) if gene == 1]
    shap_deletion_auc = compute_deletion_auc(
        model, image, shap_indices, original_prob, region_shap_scores
    )
    shap_insertion_auc = compute_insertion_auc(
        model, image, shap_indices, original_prob, region_shap_scores
    )

    # GA-NSGA-II
    _, ga_details = calculate_objectives(
        ga_chromosome, model=model, image=image, original_prob=original_prob, region_shap_scores=region_shap_scores
    )
    ga_deletion_auc = compute_deletion_auc(
        model, image, selected_indices, original_prob, region_shap_scores
    )
    ga_insertion_auc = compute_insertion_auc(
        model, image, selected_indices, original_prob, region_shap_scores
    )

    results = {
        "Random-K": {
            "prediction_preservation": random_details["f1_preservation"],
            "shap_retention": random_details["f2_shap_importance"],
            "compactness": random_details["f3_compactness"],
            "deletion_auc": random_deletion_auc,
            "insertion_auc": random_insertion_auc,
            "k_selected": k_selected,
        },
        "SHAP-TopK": {
            "prediction_preservation": shap_details["f1_preservation"],
            "shap_retention": shap_details["f2_shap_importance"],
            "compactness": shap_details["f3_compactness"],
            "deletion_auc": shap_deletion_auc,
            "insertion_auc": shap_insertion_auc,
            "k_selected": k_selected,
        },
        "GA-NSGA-II": {
            "prediction_preservation": ga_details["f1_preservation"],
            "shap_retention": ga_details["f2_shap_importance"],
            "compactness": ga_details["f3_compactness"],
            "deletion_auc": ga_deletion_auc,
            "insertion_auc": ga_insertion_auc,
            "k_selected": k_selected,
        },
    }

    return results
