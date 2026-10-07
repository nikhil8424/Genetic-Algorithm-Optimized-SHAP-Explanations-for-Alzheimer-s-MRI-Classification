import os
import random
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from deap import base, creator, tools
import pandas as pd

import config


def chromosome_to_mask(
    chromosome: List[int],
    grid_rows: int = config.GRID_ROWS,
    grid_cols: int = config.GRID_COLS,
    img_size: int = config.IMG_SIZE,
) -> np.ndarray:
    """
    Converts a 64-element binary chromosome into a 128x128 pixel-level binary mask.

    A chromosome is a binary list of length 64 (one bit per region in the 8x8 grid).
    A value of 1 means the region is selected, 0 means it's not selected.
    This function upsamples the chromosome to pixel-level by expanding each region
    to its corresponding 16x16 pixel block.

    Example:
    - Chromosome: [1, 0, 1, 0, ...] (64 elements)
    - Reshaped to 8x8 grid: [[1, 0, ...], [1, 0, ...], ...]
    - Upsampled to 128x128: Each element becomes a 16x16 block

    Args:
        chromosome: Binary list of length 64 (1 = select region, 0 = don't select)
        grid_rows: Number of rows in the grid (default: 8)
        grid_cols: Number of columns in the grid (default: 8)
        img_size: Size of the output image (default: 128)

    Returns:
        Binary mask with shape (128, 128, 1) where 1 = keep pixel, 0 = mask out
    """
    # Reshape chromosome to 2D grid (8x8)
    chrom_arr = np.array(chromosome, dtype=np.float32).reshape((grid_rows, grid_cols))
    # Calculate region dimensions (16x16 pixels)
    region_h = img_size // grid_rows
    region_w = img_size // grid_cols
    # Upsample: repeat each row element 16 times, then repeat each row 16 times
    # This expands each grid cell to a 16x16 pixel block
    pixel_mask = np.repeat(np.repeat(chrom_arr, region_h, axis=0), region_w, axis=1)

    # Add channel dimension if needed
    if pixel_mask.ndim == 2:
        pixel_mask = np.expand_dims(pixel_mask, axis=-1)

    return pixel_mask


def calculate_objectives(
    chromosome: List[int],
    model: tf.keras.Model,
    image: np.ndarray,
    original_prob: float,
    region_shap_scores: np.ndarray,
) -> Tuple[Tuple[float, float, float], Dict[str, float]]:
    """
    Evaluates 2D chromosome on 3 objectives: Preservation, SHAP retained, Compactness.

    This function is the core fitness evaluation for the genetic algorithm. It computes
    three conflicting objectives that the GA tries to optimize simultaneously:

    1. f1 (Prediction Preservation): How well the model's prediction is preserved
       when only selected regions are kept. Higher is better.
       - Computed as: 1 - |original_prob - masked_prob|
       - If prediction doesn't change, f1 = 1.0 (perfect)

    2. f2 (SHAP Importance Retained): How much of the total SHAP importance is captured
       by the selected regions. Higher is better.
       - Computed as: sum(SHAP of selected regions) / sum(SHAP of all regions)
       - If all important regions are selected, f2 = 1.0 (perfect)

    3. f3 (Compactness): How sparse the selection is (fewer regions = more compact).
       Higher is better.
       - Computed as: 1 - (k_selected / total_regions)
       - If only 1 region is selected, f3 = 1 - 1/64 ≈ 0.98 (very compact)
       - If all 64 regions are selected, f3 = 0 (not compact)

    Args:
        chromosome: Binary list of length 64 (1 = select region, 0 = don't select)
        model: Trained CNN model
        image: Original MRI image (128, 128, 1)
        original_prob: Model's prediction probability on the full image
        region_shap_scores: SHAP importance scores for each of the 64 regions

    Returns:
        Tuple of (objectives, details):
        - objectives: Tuple of (f1, f2, f3) values (all in [0, 1])
        - details: Dictionary with detailed metrics
    """
    # Find which regions are selected (where chromosome has value 1)
    selected_indices = [i for i, gene in enumerate(chromosome) if gene == 1]
    k_selected = len(selected_indices)  # Number of selected regions

    # Handle edge case: no regions selected
    if k_selected == 0:
        empty_details = {
            "f1_preservation": 0.0,
            "f2_shap_importance": 0.0,
            "f3_compactness": 1.0,
            "masked_prob": 0.0,
            "k_selected": 0,
            "fitness": 0.0,
            "prediction_preservation": 0.0,
            "shap_importance": 0.0,
            "sparsity_penalty": 0.0,
        }
        return (0.0, 0.0, 1.0), empty_details

    # Convert chromosome to pixel-level mask
    mask_128 = chromosome_to_mask(chromosome)
    # Ensure image has 3 dimensions
    img_3d = image if image.ndim == 3 else np.expand_dims(image, axis=-1)
    # Apply mask: keep selected regions, zero out others
    masked_image = img_3d * mask_128
    # Get model prediction on masked image
    batch_masked = np.expand_dims(masked_image, axis=0)
    masked_prob = float(model.predict(batch_masked, verbose=0)[0][0])

    # f1: Prediction Preservation (should be close to original probability)
    pred_diff = abs(original_prob - masked_prob)
    f1 = float(np.clip(1.0 - pred_diff, 0.0, 1.0))

    # f2: SHAP Importance Retained (fraction of total importance captured)
    total_shap = float(np.sum(region_shap_scores))
    if total_shap > 1e-9:
        selected_shap = float(np.sum(region_shap_scores[selected_indices]))
        f2 = float(np.clip(selected_shap / total_shap, 0.0, 1.0))
    else:
        f2 = 0.0

    # f3: Compactness (inverse of sparsity - fewer regions is better)
    sparsity = float(k_selected / config.NUM_REGIONS)
    f3 = float(np.clip(1.0 - sparsity, 0.0, 1.0))

    # Compute composite scalar fitness (for reporting, not used by NSGA-II)
    details = {
        "f1_preservation": f1,
        "f2_shap_importance": f2,
        "f3_compactness": f3,
        "masked_prob": masked_prob,
        "k_selected": k_selected,
        "prediction_preservation": f1,
        "shap_importance": f2,
        "sparsity_penalty": sparsity,
        "fitness": float(config.FITNESS_ALPHA * f1 + config.FITNESS_BETA * f2 - config.FITNESS_GAMMA * sparsity),
    }
    return (f1, f2, f3), details


def calculate_fitness(
    chromosome: List[int],
    model: tf.keras.Model,
    image: np.ndarray,
    original_prob: float,
    region_shap_scores: np.ndarray,
    alpha: float = config.FITNESS_ALPHA,
    beta: float = config.FITNESS_BETA,
    gamma: float = config.FITNESS_GAMMA,
) -> Tuple[float, Dict[str, float]]:
    """
    Scalar composite wrapper around calculate_objectives().

    This function combines the three objectives into a single scalar fitness score
    using weighted sum. This is useful for:
    - Reporting purposes
    - Comparison with baseline methods
    - Single-objective optimization (if needed)

    Note: NSGA-II (the actual GA algorithm) uses the three objectives separately
    and does not use this scalar fitness. This is only for auxiliary purposes.

    Fitness formula: alpha * f1 + beta * f2 - gamma * sparsity
    - f1 and f2 are rewarded (positive contribution)
    - sparsity is penalized (negative contribution)

    Args:
        chromosome: Binary list of length 64
        model: Trained CNN model
        image: Original MRI image
        original_prob: Model's prediction on full image
        region_shap_scores: SHAP scores for each region
        alpha: Weight for f1 (prediction preservation)
        beta: Weight for f2 (SHAP importance)
        gamma: Weight for sparsity penalty

    Returns:
        Tuple of (scalar_fitness, details)
    """
    objectives, details = calculate_objectives(
        chromosome, model, image, original_prob, region_shap_scores
    )
    f1, f2, f3 = objectives
    sparsity = 1.0 - f3
    # Compute weighted sum
    scalar = float(alpha * f1 + beta * f2 - gamma * sparsity)
    scalar = max(0.0, scalar)  # Ensure non-negative
    details["fitness"] = scalar
    return scalar, details


def get_pareto_front(population) -> List:
    """
    Returns Pareto front rank 0 non-dominated solutions.

    In multi-objective optimization, a solution A dominates solution B if:
    - A is better than or equal to B in ALL objectives
    - A is strictly better than B in AT LEAST ONE objective

    The Pareto front consists of all non-dominated solutions - these are the
    best trade-offs between the objectives. No solution in the Pareto front
    can be improved in one objective without worsening another.

    Args:
        population: List of individuals (chromosomes) with fitness.values

    Returns:
        List of non-dominated individuals (Pareto front)
    """
    front = []
    for ind in population:
        dominated = False
        for other in population:
            if other is ind:
                continue
            other_vals = other.fitness.values
            ind_vals = ind.fitness.values
            # Check if 'other' dominates 'ind'
            if all(o >= i for o, i in zip(other_vals, ind_vals)) and any(o > i for o, i in zip(other_vals, ind_vals)):
                dominated = True
                break
        if not dominated:
            front.append(ind)
    return front


def select_knee_point(pareto_front: List) -> Tuple[List[int], Tuple[float, float, float]]:
    """
    Selects knee-point closest to utopia [1.0, 1.0, 1.0].

    The knee point is the solution on the Pareto front that represents the best
    compromise between all objectives. It's typically the point closest to the
    "utopia point" (1.0, 1.0, 1.0), which represents perfect scores on all
    objectives simultaneously (theoretically impossible but the ideal target).

    We use Euclidean distance to measure closeness to the utopia point.

    Args:
        pareto_front: List of non-dominated individuals

    Returns:
        Tuple of (best_chromosome, best_objectives)
        - best_chromosome: The selected knee point chromosome
        - best_objectives: The (f1, f2, f3) values of the knee point
    """
    utopia = np.array([1.0, 1.0, 1.0])  # Ideal point (perfect on all objectives)
    best_ind = None
    best_dist = float("inf")
    # Find individual closest to utopia point
    for ind in pareto_front:
        obj_vec = np.array(ind.fitness.values)
        dist = float(np.linalg.norm(obj_vec - utopia))
        if dist < best_dist:
            best_dist = dist
            best_ind = ind
    return list(best_ind), tuple(best_ind.fitness.values)


def run_genetic_algorithm(
    model: tf.keras.Model,
    image: np.ndarray,
    original_prob: float,
    region_shap_scores: np.ndarray,
    population_size: int = config.GA_POPULATION_SIZE,
    generations: int = config.GA_GENERATIONS,
    cxpb: float = config.GA_CROSSOVER_PROB,
    mutpb: float = config.GA_MUTATION_PROB,
    random_seed: int = config.RANDOM_SEED,
) -> Tuple[List[int], pd.DataFrame, Dict[str, float], List]:
    """
    Runs 2D NSGA-II on 64-bit spatial grid.

    NSGA-II (Non-dominated Sorting Genetic Algorithm II) is a multi-objective
    evolutionary algorithm that optimizes multiple conflicting objectives simultaneously.
    It maintains a diverse set of solutions (Pareto front) representing different
    trade-offs between the objectives.

    Algorithm steps:
    1. Initialize random population of chromosomes (binary lists of length 64)
    2. Evaluate each chromosome on 3 objectives (f1, f2, f3)
    3. For each generation:
       a. Select parents using tournament selection
       b. Create offspring via crossover (mix parents) and mutation (random bit flips)
       c. Evaluate offspring
       d. Combine parent and offspring populations
       e. Select best individuals using NSGA-II (non-dominated sorting + crowding distance)
    4. Return the knee point from the final Pareto front

    Optimization:
    - Uses evaluation cache to avoid redundant model predictions
    - Uses DEAP library for GA operations

    Args:
        model: Trained CNN model
        image: Original MRI image (128, 128, 1)
        original_prob: Model's prediction on full image
        region_shap_scores: SHAP scores for each of the 64 regions
        population_size: Number of individuals in population (default: 20)
        generations: Number of generations to evolve (default: 10)
        cxpb: Crossover probability (default: 0.7)
        mutpb: Mutation probability (default: 0.2)
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (best_chromosome, history_df, best_details, pareto_front):
        - best_chromosome: The selected knee point chromosome (list of 64 bits)
        - history_df: DataFrame with GA progress across generations
        - best_details: Dictionary with detailed metrics for best solution
        - pareto_front: List of all non-dominated solutions in final population
    """
    # Set random seeds for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Cache for chromosome evaluations to avoid redundant predictions
    # Since the same chromosome may appear multiple times, we cache results
    evaluation_cache = {}

    # Define fitness class for multi-objective optimization
    # weights=(1.0, 1.0, 1.0) means we want to maximize all 3 objectives
    fitness_cls_name = "FitnessMultiNSGA2_2D"
    individual_cls_name = "IndividualNSGA2_2D"

    if not hasattr(creator, fitness_cls_name):
        creator.create(fitness_cls_name, base.Fitness, weights=(1.0, 1.0, 1.0))
    if not hasattr(creator, individual_cls_name):
        creator.create(individual_cls_name, list, fitness=getattr(creator, fitness_cls_name))

    IndividualClass = getattr(creator, individual_cls_name)

    # Configure GA toolbox with operators
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)  # Random bit generator
    toolbox.register("individual", tools.initRepeat, IndividualClass, toolbox.attr_bool, n=config.NUM_REGIONS)  # Create individual with 64 bits
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)  # Create population

    # Define evaluation function with caching
    def evaluate(ind):
        # Convert chromosome to tuple for hashing (keys must be hashable)
        chrom_tuple = tuple(ind)
        if chrom_tuple in evaluation_cache:
            return evaluation_cache[chrom_tuple]

        # Compute objectives if not cached
        objectives, _ = calculate_objectives(
            ind, model=model, image=image, original_prob=original_prob, region_shap_scores=region_shap_scores
        )
        evaluation_cache[chrom_tuple] = objectives
        return objectives

    # Register GA operators
    toolbox.register("evaluate", evaluate)  # Fitness evaluation
    toolbox.register("mate", tools.cxTwoPoint)  # Two-point crossover
    toolbox.register("mutate", tools.mutFlipBit, indpb=config.GA_BIT_FLIP_PROB)  # Bit-flip mutation
    toolbox.register("select", tools.selNSGA2)  # NSGA-II selection

    # Initialize population
    pop = toolbox.population(n=population_size)
    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Apply NSGA-II selection to rank population
    pop = toolbox.select(pop, len(pop))
    history_records = []

    # Evolution loop
    for gen in range(1, generations + 1):
        # Select parents using tournament selection (dominance crowding distance)
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        # Apply crossover to pairs of offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values  # Fitness invalid after crossover
                del child2.fitness.values

        # Apply mutation to offspring
        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values  # Fitness invalid after mutation

        # Evaluate invalid (modified) individuals
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Combine parent and offspring populations
        combined = pop + offspring
        # Select best individuals for next generation using NSGA-II
        pop[:] = toolbox.select(combined, population_size)

        # Record statistics for this generation
        all_f1 = [ind.fitness.values[0] for ind in pop]
        all_f2 = [ind.fitness.values[1] for ind in pop]
        all_f3 = [ind.fitness.values[2] for ind in pop]
        front = get_pareto_front(pop)

        history_records.append({
            "Generation": gen,
            "Best_F1_Preservation": max(all_f1),
            "Best_F2_SHAP": max(all_f2),
            "Best_F3_Compactness": max(all_f3),
            "Avg_F1": sum(all_f1) / len(all_f1),
            "Avg_F2": sum(all_f2) / len(all_f2),
            "Avg_F3": sum(all_f3) / len(all_f3),
            "Pareto_Front_Size": len(front),
        })

    # Get final Pareto front and select knee point
    pareto_front = get_pareto_front(pop)
    best_chromosome, best_objectives = select_knee_point(pareto_front)
    # Get detailed metrics for best solution
    _, best_details = calculate_objectives(
        best_chromosome, model=model, image=image, original_prob=original_prob, region_shap_scores=region_shap_scores
    )

    return best_chromosome, pd.DataFrame(history_records), best_details, pareto_front


def save_ga_history_csv(
    history_df: pd.DataFrame,
    save_path: str = "results/ga_history.csv",
) -> None:
    """
    Saves 2D GA history to CSV.

    This function saves the progress of the genetic algorithm across generations,
    including best and average values for each objective and Pareto front size.

    Args:
        history_df: DataFrame with generation-wise statistics
        save_path: Path to save the CSV file
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    history_df.to_csv(save_path, index=False)


def plot_ga_fitness(
    history_df: pd.DataFrame,
    save_path: str = "results/ga_fitness.png",
) -> None:
    """
    Plots 2D GA multi-objective convergence.

    This function creates a two-panel visualization:
    1. Top panel: Convergence of the three objectives (f1, f2, f3) over generations
       Shows both best values and trends
    2. Bottom panel: Size of the Pareto front over generations
       Indicates how many diverse solutions are maintained

    Args:
        history_df: DataFrame with generation-wise statistics
        save_path: Path to save the plot
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    gens = history_df["Generation"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    # Plot objective convergence
    ax1.plot(gens, history_df["Best_F1_Preservation"], marker="o", color="#1f77b4", lw=2, label="f1 - Preservation (best)")
    ax1.plot(gens, history_df["Best_F2_SHAP"], marker="s", color="#ff7f0e", lw=2, label="f2 - SHAP (best)")
    ax1.plot(gens, history_df["Best_F3_Compactness"], marker="^", color="#2ca02c", lw=2, label="f3 - Compactness (best)")
    ax1.set_ylabel("Objective Value", fontsize=11)
    ax1.set_ylim(-0.05, 1.08)
    ax1.grid(True, linestyle=":", alpha=0.55)
    ax1.legend(loc="lower right", frameon=True, fontsize=9)

    # Plot Pareto front size
    ax2.bar(gens, history_df["Pareto_Front_Size"], color="#9467bd", alpha=0.75, label="Pareto Front Size")
    ax2.set_xlabel("Generation", fontsize=11)
    ax2.set_ylabel("Front Size", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.55)
    ax2.legend(loc="upper left", frameon=True, fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_pareto_front(
    pareto_front: List,
    sample_idx: int = 1,
    save_path: str = "results/pareto_front.png",
) -> None:
    """
    Plots the 2D Pareto front scatter.

    This function visualizes the Pareto front as a scatter plot. Since we have
    3 objectives, we plot two on the axes (f1 vs f3) and use color for the third (f2).

    - X-axis: f3 (Compactness)
    - Y-axis: f1 (Prediction Preservation)
    - Color: f2 (SHAP Importance Retained)

    Each point represents a non-dominated solution (different trade-off).

    Args:
        pareto_front: List of non-dominated individuals
        sample_idx: Sample index for title
        save_path: Path to save the plot
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not pareto_front:
        return

    # Extract objective values from Pareto front
    f1_vals = np.array([ind.fitness.values[0] for ind in pareto_front])
    f2_vals = np.array([ind.fitness.values[1] for ind in pareto_front])
    f3_vals = np.array([ind.fitness.values[2] for ind in pareto_front])

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(f3_vals, f1_vals, c=f2_vals, cmap="plasma", s=80, edgecolors="#333333", vmin=0.0, vmax=1.0)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("f2 -- SHAP Importance Retained", fontsize=10)
    ax.set_xlabel("f3 -- Compactness", fontsize=11)
    ax.set_ylabel("f1 -- Prediction Preservation", fontsize=11)
    ax.set_title(f"2D Baseline Pareto Front (Sample #{sample_idx})", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_pareto_front_csv(
    pareto_front: List,
    sample_idx: int = 1,
    save_path: str = "results/pareto_front.csv",
) -> None:
    """
    Saves Pareto front to CSV with only F1, F2, F3.

    This function saves all solutions in the Pareto front to a CSV file for
    further analysis. Each row represents one non-dominated solution.

    Args:
        pareto_front: List of non-dominated individuals
        sample_idx: Sample index (not used in CSV, kept for consistency)
        save_path: Path to save the CSV file
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not pareto_front:
        return

    # Extract information from each individual in Pareto front
    records = []
    for i, ind in enumerate(pareto_front):
        f1, f2, f3 = ind.fitness.values
        k_selected = sum(ind)  # Number of selected regions
        records.append({
            "Solution_ID": i,
            "K_Selected": k_selected,
            "F1_Preservation": f1,
            "F2_SHAP_Retained": f2,
            "F3_Compactness": f3,
        })

    df = pd.DataFrame(records)
    df.to_csv(save_path, index=False)
