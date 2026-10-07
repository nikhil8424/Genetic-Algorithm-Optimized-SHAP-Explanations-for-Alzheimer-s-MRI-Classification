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
    """
    Constructs Random-K baseline chromosome and mask.

    This is a simple baseline that selects K regions uniformly at random.
    It provides a lower bound for comparison - any intelligent method should
    outperform random selection.

    Args:
        k: Number of regions to select
        total_regions: Total number of regions (default: 64)
        grid_rows: Number of rows in the grid (default: 8)
        grid_cols: Number of columns in the grid (default: 8)
        img_size: Size of the image (default: 128)
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (chromosome, mask):
        - chromosome: Binary list of length 64 (1 = selected, 0 = not selected)
        - mask: Pixel-level binary mask (128, 128, 1)
    """
    np.random.seed(random_seed)
    k = max(1, min(k, total_regions))
    # Randomly select K unique region indices
    random_indices = set(np.random.choice(total_regions, size=k, replace=False))
    # Create chromosome: 1 for selected regions, 0 for others
    random_k_chromosome = [1 if i in random_indices else 0 for i in range(total_regions)]
    # Convert to pixel-level mask
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

    Deletion AUC measures how well the model's prediction is preserved when
    important regions are progressively removed from the image. A good explanation
    should maintain the prediction as long as possible when important regions are removed.

    Process:
    1. Start with full image (all regions present)
    2. Progressively remove regions in order of SHAP importance (most important first)
    3. Track model prediction after each removal step
    4. Compute area under the probability curve

    Higher Deletion AUC = better explanation (prediction maintained longer)

    Optimization: Uses at most 16 evaluation steps (not K steps) for speed,
    removing multiple regions per step.

    Args:
        model: Trained CNN model
        image: Original MRI image (128, 128, 1)
        selected_region_indices: Indices of selected regions
        original_prob: Model's prediction on full image
        region_shap_scores: SHAP scores for each region
        grid_rows: Number of rows in the grid (default: 8)
        grid_cols: Number of columns in the grid (default: 8)
        img_size: Size of the image (default: 128)

    Returns:
        Deletion AUC score between 0 and 1
    """
    if not selected_region_indices:
        return 0.0

    img_3d = image if image.ndim == 3 else np.expand_dims(image, axis=-1)
    k = len(selected_region_indices)

    # Sort regions by SHAP importance (descending) for deletion
    # Remove most important regions first
    region_importance = [(idx, region_shap_scores[idx]) for idx in selected_region_indices]
    region_importance.sort(key=lambda x: x[1], reverse=True)

    # Use fewer evaluation steps (max 16 instead of k) for speed
    max_steps = min(16, k)
    step_size = max(1, k // max_steps)

    probs = [original_prob]  # Start with original probability
    current_mask = np.ones((img_size, img_size, 1), dtype=np.float32)  # Initially, all regions present

    # Progressively remove regions
    for step in range(max_steps):
        # Remove multiple regions at once
        start_idx = step * step_size
        end_idx = min((step + 1) * step_size, k)

        for region_idx, _ in region_importance[start_idx:end_idx]:
            # Calculate region boundaries
            row = region_idx // grid_cols
            col = region_idx % grid_cols
            region_h = img_size // grid_rows
            region_w = img_size // grid_cols
            # Set region to 0 in mask (remove it)
            current_mask[row*region_h:(row+1)*region_h, col*region_w:(col+1)*region_w, :] = 0.0

        # Apply mask and get prediction
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

    Insertion AUC measures how quickly the model's prediction recovers when
    important regions are progressively added to a blank image. A good explanation
    should recover the prediction quickly when important regions are added.

    Process:
    1. Start with blank image (all regions masked out)
    2. Progressively add regions in order of SHAP importance (most important first)
    3. Track model prediction after each addition step
    4. Compute area under the probability curve

    Higher Insertion AUC = better explanation (prediction recovers faster)

    Optimization: Uses at most 16 evaluation steps (not K steps) for speed,
    adding multiple regions per step.

    Args:
        model: Trained CNN model
        image: Original MRI image (128, 128, 1)
        selected_region_indices: Indices of selected regions
        original_prob: Model's prediction on full image
        region_shap_scores: SHAP scores for each region
        grid_rows: Number of rows in the grid (default: 8)
        grid_cols: Number of columns in the grid (default: 8)
        img_size: Size of the image (default: 128)

    Returns:
        Insertion AUC score between 0 and 1
    """
    if not selected_region_indices:
        return 0.0

    img_3d = image if image.ndim == 3 else np.expand_dims(image, axis=-1)
    k = len(selected_region_indices)

    # Sort regions by SHAP importance (descending) for insertion
    # Add most important regions first
    region_importance = [(idx, region_shap_scores[idx]) for idx in selected_region_indices]
    region_importance.sort(key=lambda x: x[1], reverse=True)

    # Use fewer evaluation steps (max 16 instead of k) for speed
    max_steps = min(16, k)
    step_size = max(1, k // max_steps)

    # Start with zero mask (all regions masked out)
    current_mask = np.zeros((img_size, img_size, 1), dtype=np.float32)
    masked_img = img_3d * current_mask
    batch_masked = np.expand_dims(masked_img, axis=0)
    start_prob = float(model.predict(batch_masked, verbose=0)[0][0])

    probs = [start_prob]  # Start with probability on blank image

    # Progressively add regions
    for step in range(max_steps):
        # Add multiple regions at once
        start_idx = step * step_size
        end_idx = min((step + 1) * step_size, k)

        for region_idx, _ in region_importance[start_idx:end_idx]:
            # Calculate region boundaries
            row = region_idx // grid_cols
            col = region_idx % grid_cols
            region_h = img_size // grid_rows
            region_w = img_size // grid_cols
            # Set region to 1 in mask (add it)
            current_mask[row*region_h:(row+1)*region_h, col*region_w:(col+1)*region_w, :] = 1.0

        # Apply mask and get prediction
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

    This function performs a comprehensive comparison of three region selection methods:
    1. Random-K: Random region selection (naive baseline)
    2. SHAP Top-K: Select regions with highest SHAP scores (greedy baseline)
    3. GA-NSGA-II: Multi-objective genetic algorithm (proposed method)

    All methods select the same number of regions (K) as the GA solution for fair comparison.

    For each method, it computes:
    - Prediction Preservation (f1): How well the model's prediction is maintained
    - SHAP Retention (f2): How much SHAP importance is captured
    - Compactness (f3): How sparse the selection is
    - Deletion AUC: How well prediction is maintained when regions are removed
    - Insertion AUC: How quickly prediction recovers when regions are added

    Args:
        model: Trained CNN model
        image: Original MRI image (128, 128, 1)
        original_prob: Model's prediction on full image
        region_shap_scores: SHAP scores for each region
        ga_chromosome: Best chromosome from genetic algorithm
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary with method names as keys and metric dictionaries as values
    """
    # Get number of regions selected by GA
    k_selected = sum(ga_chromosome)
    selected_indices = [i for i, gene in enumerate(ga_chromosome) if gene == 1]

    # =======================
    # METHOD 1: RANDOM-K
    # =======================
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

    # =======================
    # METHOD 2: SHAP TOP-K
    # =======================
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

    # =======================
    # METHOD 3: GA-NSGA-II
    # =======================
    _, ga_details = calculate_objectives(
        ga_chromosome, model=model, image=image, original_prob=original_prob, region_shap_scores=region_shap_scores
    )
    ga_deletion_auc = compute_deletion_auc(
        model, image, selected_indices, original_prob, region_shap_scores
    )
    ga_insertion_auc = compute_insertion_auc(
        model, image, selected_indices, original_prob, region_shap_scores
    )

    # Compile results
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
