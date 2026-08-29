import os
import random
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from deap import base, creator, tools

import config


def chromosome_to_mask(
    chromosome: List[int],
    grid_rows: int = config.GRID_ROWS,
    grid_cols: int = config.GRID_COLS,
    img_size: int = config.IMG_SIZE,
) -> np.ndarray:
    """Converts a 64-element binary chromosome into a 128x128 pixel-level binary mask."""
    chrom_arr = np.array(chromosome, dtype=np.float32).reshape((grid_rows, grid_cols))
    region_h = img_size // grid_rows
    region_w = img_size // grid_cols
    pixel_mask = np.repeat(np.repeat(chrom_arr, region_h, axis=0), region_w, axis=1)

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
    """Evaluates 2D chromosome on 3 objectives: Preservation, SHAP retained, Compactness."""
    selected_indices = [i for i, gene in enumerate(chromosome) if gene == 1]
    k_selected = len(selected_indices)

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

    mask_128 = chromosome_to_mask(chromosome)
    img_3d = image if image.ndim == 3 else np.expand_dims(image, axis=-1)
    masked_image = img_3d * mask_128
    batch_masked = np.expand_dims(masked_image, axis=0)
    masked_prob = float(model.predict(batch_masked, verbose=0)[0][0])
    pred_diff = abs(original_prob - masked_prob)
    f1 = float(np.clip(1.0 - pred_diff, 0.0, 1.0))

    total_shap = float(np.sum(region_shap_scores))
    if total_shap > 1e-9:
        selected_shap = float(np.sum(region_shap_scores[selected_indices]))
        f2 = float(np.clip(selected_shap / total_shap, 0.0, 1.0))
    else:
        f2 = 0.0

    sparsity = float(k_selected / config.NUM_REGIONS)
    f3 = float(np.clip(1.0 - sparsity, 0.0, 1.0))

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
    """Scalar composite wrapper around calculate_objectives()."""
    objectives, details = calculate_objectives(
        chromosome, model, image, original_prob, region_shap_scores
    )
    f1, f2, f3 = objectives
    sparsity = 1.0 - f3
    scalar = float(alpha * f1 + beta * f2 - gamma * sparsity)
    scalar = max(0.0, scalar)
    details["fitness"] = scalar
    return scalar, details


def get_pareto_front(population) -> List:
    """Returns Pareto front rank 0 non-dominated solutions."""
    front = []
    for ind in population:
        dominated = False
        for other in population:
            if other is ind:
                continue
            other_vals = other.fitness.values
            ind_vals = ind.fitness.values
            if all(o >= i for o, i in zip(other_vals, ind_vals)) and any(o > i for o, i in zip(other_vals, ind_vals)):
                dominated = True
                break
        if not dominated:
            front.append(ind)
    return front


def select_knee_point(pareto_front: List) -> Tuple[List[int], Tuple[float, float, float]]:
    """Selects knee-point closest to utopia [1.0, 1.0, 1.0]."""
    utopia = np.array([1.0, 1.0, 1.0])
    best_ind = None
    best_dist = float("inf")
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
    """Runs 2D NSGA-II on 64-bit spatial grid."""
    random.seed(random_seed)
    np.random.seed(random_seed)

    fitness_cls_name = "FitnessMultiNSGA2_2D"
    individual_cls_name = "IndividualNSGA2_2D"

    if not hasattr(creator, fitness_cls_name):
        creator.create(fitness_cls_name, base.Fitness, weights=(1.0, 1.0, 1.0))
    if not hasattr(creator, individual_cls_name):
        creator.create(individual_cls_name, list, fitness=getattr(creator, fitness_cls_name))

    IndividualClass = getattr(creator, individual_cls_name)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, IndividualClass, toolbox.attr_bool, n=config.NUM_REGIONS)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(ind):
        objectives, _ = calculate_objectives(
            ind, model=model, image=image, original_prob=original_prob, region_shap_scores=region_shap_scores
        )
        return objectives

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=config.GA_BIT_FLIP_PROB)
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=population_size)
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    pop = toolbox.select(pop, len(pop))
    history_records = []

    for gen in range(1, generations + 1):
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        combined = pop + offspring
        pop[:] = toolbox.select(combined, population_size)

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

    pareto_front = get_pareto_front(pop)
    best_chromosome, best_objectives = select_knee_point(pareto_front)
    _, best_details = calculate_objectives(
        best_chromosome, model=model, image=image, original_prob=original_prob, region_shap_scores=region_shap_scores
    )

    return best_chromosome, pd.DataFrame(history_records), best_details, pareto_front


def save_ga_history_csv(
    history_df: pd.DataFrame,
    save_path: str = "results/ga_history.csv",
) -> None:
    """Saves 2D GA history to CSV."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    history_df.to_csv(save_path, index=False)


def plot_ga_fitness(
    history_df: pd.DataFrame,
    save_path: str = "results/ga_fitness.png",
) -> None:
    """Plots 2D GA multi-objective convergence."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    gens = history_df["Generation"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(gens, history_df["Best_F1_Preservation"], marker="o", color="#1f77b4", lw=2, label="f1 - Preservation (best)")
    ax1.plot(gens, history_df["Best_F2_SHAP"], marker="s", color="#ff7f0e", lw=2, label="f2 - SHAP (best)")
    ax1.plot(gens, history_df["Best_F3_Compactness"], marker="^", color="#2ca02c", lw=2, label="f3 - Compactness (best)")
    ax1.set_ylabel("Objective Value", fontsize=11)
    ax1.set_ylim(-0.05, 1.08)
    ax1.grid(True, linestyle=":", alpha=0.55)
    ax1.legend(loc="lower right", frameon=True, fontsize=9)

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
    """Plots the 2D Pareto front scatter."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not pareto_front:
        return

    f1_vals = np.array([ind.fitness.values[0] for ind in pareto_front])
    f2_vals = np.array([ind.fitness.values[1] for ind in pareto_front])
    f3_vals = np.array([ind.fitness.values[2] for ind in pareto_front])

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
