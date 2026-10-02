import os
import sys
import argparse
import json
import time
import numpy as np
import pandas as pd
import tensorflow as tf

import config
from dataset import (
    load_dataset,
    split_dataset,
    print_dataset_statistics,
    select_stratified_evaluation_samples,
    generate_synthetic_mri_dataset,
)
from preprocessing import create_data_arrays
from model import build_cnn
from training import train_model, load_trained_model, save_model
from evaluation import evaluate_model
from shap_explainer import (
    create_shap_explainer,
    compute_shap_values,
    save_shap_visualizations,
)
from region_analysis import (
    calculate_region_shap_scores,
    save_region_scores_csv,
    plot_region_grid,
    plot_region_importance,
)
from genetic_algorithm import (
    run_genetic_algorithm,
    save_ga_history_csv,
    plot_ga_fitness,
    plot_pareto_front,
    save_pareto_front_csv,
    chromosome_to_mask,
)
from comparison import compare_all_methods
from visualization import (
    plot_training_history,
    plot_final_comparison,
)


def setup_directories() -> None:
    """Ensures all necessary output directories exist."""
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.SHAP_RESULTS_DIR, exist_ok=True)
    os.makedirs(config.FINAL_RESULTS_DIR, exist_ok=True)
    os.makedirs(config.COMPARISONS_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="2D GA-SHAP Alzheimer's MRI Classification Pipeline")
    parser.add_argument("--retrain", action="store_true", help="Force retrain 2D CNN model from scratch")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Number of training epochs")
    parser.add_argument("--pop-size", type=int, default=config.GA_POPULATION_SIZE, help="GA population size")
    parser.add_argument("--generations", type=int, default=config.GA_GENERATIONS, help="GA generations")
    parser.add_argument("--eval-samples", type=int, default=config.GA_EVAL_SAMPLES, help="Number of evaluation samples (default: 100)")
    parser.add_argument("--ga-seeds", type=int, nargs='+', default=config.GA_SEEDS, help="GA random seeds")
    return parser.parse_args()


def save_experiment_config(args, test_size, val_size, train_size, eval_size, save_path):
    """Saves experiment configuration to JSON."""
    config_dict = {
        "image_size": config.IMG_SIZE,
        "grid_size": f"{config.GRID_ROWS}x{config.GRID_COLS}",
        "dataset_split": {
            "train_size": train_size,
            "validation_size": val_size,
            "test_size": test_size,
            "train_pct": f"{train_size/(train_size+val_size+test_size)*100:.1f}%",
            "val_pct": f"{val_size/(train_size+val_size+test_size)*100:.1f}%",
            "test_pct": f"{test_size/(train_size+val_size+test_size)*100:.1f}%",
        },
        "evaluation_samples": eval_size,
        "ga_seeds": args.ga_seeds,
        "ga_hyperparameters": {
            "population_size": args.pop_size,
            "generations": args.generations,
            "crossover_probability": config.GA_CROSSOVER_PROB,
            "mutation_probability": config.GA_MUTATION_PROB,
            "bit_flip_probability": config.GA_BIT_FLIP_PROB,
        },
        "shap_settings": {
            "background_size": config.SHAP_BACKGROUND_SIZE,
        },
        "training_hyperparameters": {
            "batch_size": config.BATCH_SIZE,
            "epochs": args.epochs,
            "learning_rate": config.LEARNING_RATE,
            "early_stopping_patience": config.EARLY_STOPPING_PATIENCE,
        },
    }
    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=2)


def main():
    args = parse_args()
    setup_directories()
    np.random.seed(config.RANDOM_SEED)
    tf.random.set_seed(config.RANDOM_SEED)

    print("=" * 65)
    print("      2D GA-SHAP BASELINE: ALZHEIMER'S MRI CLASSIFICATION")
    print("=" * 65)

    # 1. Dataset Loading & Validation
    if not os.path.exists(config.DATASET_DIR):
        print(f"[Notice] Dataset directory '{config.DATASET_DIR}' not found.")
        print("Synthesizing demo MRI dataset for standalone execution...")
        generate_synthetic_mri_dataset(config.DATASET_DIR, samples_per_class=30)

    df = load_dataset(config.DATASET_DIR)
    print_dataset_statistics(df)
    train_df, val_df, test_df = split_dataset(df)

    print("Loading image data arrays into memory...")
    X_train, y_train = create_data_arrays(train_df)
    X_val, y_val = create_data_arrays(val_df)
    X_test, y_test = create_data_arrays(test_df)

    # 2. Model Training / Loading
    model_path = config.MODEL_SAVE_PATH
    if os.path.exists(model_path) and not args.retrain:
        print(f"\n[Model] Found existing pretrained model at '{model_path}'. Loading...")
        model = load_trained_model(model_path)
    else:
        print(f"\n[Model] Training 2D CNN from scratch for {args.epochs} epochs...")
        model = build_cnn(use_augmentation=True)
        history = train_model(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=args.epochs,
            batch_size=config.BATCH_SIZE,
            model_path=model_path,
        )
        plot_training_history(history, save_path=os.path.join(config.RESULTS_DIR, "training_curves.png"))

    # 3. Model Evaluation
    print("\n[Evaluation] Evaluating model performance on test set...")
    metrics = evaluate_model(model, X_test, y_test, results_dir=config.RESULTS_DIR)

    # 4. Select Stratified Evaluation Samples
    print(f"\n[Sampling] Selecting {args.eval_samples} stratified evaluation samples...")
    eval_df = select_stratified_evaluation_samples(test_df, n_samples=args.eval_samples)

    # Create mapping from test indices to eval indices
    eval_indices = eval_df.index.tolist()
    X_eval = X_test[eval_indices]
    y_eval = y_test[eval_indices]

    # 5. SHAP Explanation Setup
    print("\n[SHAP] Computing SHAP background and test explanations...")
    bg_size = min(config.SHAP_BACKGROUND_SIZE, len(X_train))
    bg_indices = np.random.choice(len(X_train), size=bg_size, replace=False)
    bg_data = X_train[bg_indices]
    explainer = create_shap_explainer(model, bg_data)

    shap_start_time = time.time()
    raw_shap_batch, abs_shap_batch = compute_shap_values(explainer, X_eval)
    shap_runtime = time.time() - shap_start_time
    print(f"[SHAP] SHAP computation completed in {shap_runtime:.2f} seconds")

    # 6. Multi-Seed GA Evaluation
    print(f"\n[GA] Running GA with {len(args.ga_seeds)} seeds: {args.ga_seeds}")
    all_results = []

    total_xai_start_time = time.time()

    for seed_idx, ga_seed in enumerate(args.ga_seeds):
        print(f"\n{'='*65}")
        print(f"GA SEED {seed_idx+1}/{len(args.ga_seeds)}: {ga_seed}")
        print(f"{'='*65}")

        seed_results = []
        ga_total_time = 0.0

        for i, idx in enumerate(eval_indices):
            sample_num = i + 1
            img = X_eval[i]
            raw_shap = raw_shap_batch[i]
            abs_shap = abs_shap_batch[i]

            true_label = eval_df.iloc[i]["binary_class_name"]
            orig_prob = float(model.predict(np.expand_dims(img, axis=0), verbose=0)[0][0])
            pred_label = "Demented" if orig_prob >= 0.5 else "Normal"

            print(f"\n--- Sample #{sample_num:02d}/{len(eval_indices)} (True: {true_label}, Pred: {pred_label}, Prob: {orig_prob:.3f}) ---")

            # Save SHAP visualization only for first seed, first few samples to avoid clutter
            if seed_idx == 0 and i < 5:
                save_shap_visualizations(
                    image=img,
                    raw_shap=raw_shap,
                    abs_shap=abs_shap,
                    image_index=i,
                    true_label=true_label,
                    pred_label=pred_label,
                    pred_prob=orig_prob,
                    save_dir=config.SHAP_RESULTS_DIR,
                )

            # Region Analysis (8x8 Grid = 64 Regions)
            reg_scores, df_scores = calculate_region_shap_scores(abs_shap)
            if seed_idx == 0 and i == 0:
                save_region_scores_csv(df_scores, os.path.join(config.RESULTS_DIR, "region_scores.csv"))
                plot_region_grid(img, save_path=os.path.join(config.RESULTS_DIR, "spatial_grid_overlay.png"))
                plot_region_importance(reg_scores, save_path=os.path.join(config.RESULTS_DIR, "region_importance_heatmap.png"))

            # Run NSGA-II Genetic Algorithm
            ga_start_time = time.time()
            best_chrom, hist_df, ga_details, p_front = run_genetic_algorithm(
                model=model,
                image=img,
                original_prob=orig_prob,
                region_shap_scores=reg_scores,
                population_size=args.pop_size,
                generations=args.generations,
                random_seed=ga_seed,
            )
            ga_time = time.time() - ga_start_time
            ga_total_time += ga_time

            # Save GA artifacts (only for first seed, representative samples)
            if seed_idx == 0 and i < 3:
                save_ga_history_csv(hist_df, os.path.join(config.RESULTS_DIR, f"ga_history_sample_{sample_num:02d}.csv"))
                plot_ga_fitness(hist_df, os.path.join(config.RESULTS_DIR, f"ga_fitness_sample_{sample_num:02d}.png"))
                save_pareto_front_csv(p_front, sample_idx=sample_num, save_path=os.path.join(config.RESULTS_DIR, f"pareto_front_sample_{sample_num:02d}.csv"))
                plot_pareto_front(p_front, sample_idx=sample_num, save_path=os.path.join(config.RESULTS_DIR, f"pareto_front_sample_{sample_num:02d}.png"))

            # Comparison with all baselines
            comp_df, details_dict = compare_all_methods(
                model=model,
                image=img,
                original_prob=orig_prob,
                ga_chromosome=best_chrom,
                region_shap_scores=reg_scores,
                image_idx=i,
                random_seed=ga_seed,
                save_path=os.path.join(config.COMPARISONS_DIR, f"comparison_seed{ga_seed}_sample{sample_num:02d}.csv"),
            )

            # Generate visualization (only for first seed, first sample)
            if seed_idx == 0 and i == 0:
                k_selected = sum(best_chrom)
                from comparison import create_shap_top_k_mask
                _, shap_mask = create_shap_top_k_mask(reg_scores, k=k_selected)
                ga_mask = chromosome_to_mask(best_chrom)
                ga_masked_img = img * ga_mask

                sample_info = {
                    "index": sample_num,
                    "true_class": true_label,
                    "pred_class": pred_label,
                    "pred_prob": orig_prob,
                }
                plot_final_comparison(
                    original_img=img,
                    abs_shap_map=abs_shap,
                    shap_topk_mask=shap_mask,
                    ga_mask=ga_mask,
                    ga_masked_img=ga_masked_img,
                    sample_info=sample_info,
                    save_path=os.path.join(config.FINAL_RESULTS_DIR, f"sample_{sample_num:02d}_final_comparison.png"),
                )

            # Record results
            for method_name, method_metrics in details_dict.items():
                seed_results.append({
                    "Sample": sample_num,
                    "True_Class": true_label,
                    "Pred_Class": pred_label,
                    "Original_Prob": orig_prob,
                    "GA_Seed": ga_seed,
                    "Method": method_name,
                    "Prediction_Preservation": method_metrics["prediction_preservation"],
                    "SHAP_Retention": method_metrics["shap_retention"],
                    "Compactness": method_metrics["compactness"],
                    "Deletion_AUC": method_metrics["deletion_auc"],
                    "Insertion_AUC": method_metrics["insertion_auc"],
                    "K_Selected": method_metrics["k_selected"],
                })

        all_results.extend(seed_results)
        print(f"\n[GA Seed {ga_seed}] Completed {len(eval_indices)} samples in {ga_total_time:.2f} seconds")

    total_xai_time = time.time() - total_xai_start_time

    # 7. Aggregate Results
    print("\n[Aggregation] Computing mean ± std across samples and seeds...")
    results_df = pd.DataFrame(all_results)

    # Aggregate by method
    metrics_to_aggregate = ["Prediction_Preservation", "SHAP_Retention", "Compactness", "Deletion_AUC", "Insertion_AUC"]
    aggregated = []

    for method in ["Random-K", "SHAP-TopK", "GA-NSGA-II"]:
        method_df = results_df[results_df["Method"] == method]
        if len(method_df) == 0:
            continue

        row = {"Method": method}
        for metric in metrics_to_aggregate:
            values = method_df[metric].values
            row[f"{metric}_Mean"] = float(np.mean(values))
            row[f"{metric}_Std"] = float(np.std(values))
        aggregated.append(row)

    agg_df = pd.DataFrame(aggregated)
    agg_path = os.path.join(config.COMPARISONS_DIR, "aggregate_results.csv")
    agg_df.to_csv(agg_path, index=False)
    print(f"[Aggregation] Saved aggregated results to {agg_path}")

    # 8. Save per-sample detailed results
    detailed_path = os.path.join(config.RESULTS_DIR, "detailed_results.csv")
    results_df.to_csv(detailed_path, index=False)

    # 9. Save experiment configuration
    config_path = os.path.join(config.RESULTS_DIR, "experiment_config.json")
    save_experiment_config(
        args,
        test_size=len(test_df),
        val_size=len(val_df),
        train_size=len(train_df),
        eval_size=len(eval_df),
        save_path=config_path,
    )
    print(f"[Config] Saved experiment configuration to {config_path}")

    # 10. Create final summary
    summary_path = os.path.join(config.RESULTS_DIR, "final_summary.csv")
    summary_data = {
        "Metric": [
            "Dataset_Total_Size",
            "Train_Size",
            "Validation_Size",
            "Test_Size",
            "Evaluation_Samples",
            "GA_Seeds",
            "Accuracy",
            "Precision",
            "Recall",
            "F1_Score",
            "ROC_AUC",
            "Specificity",
            "SHAP_Runtime_s",
            "Total_XAI_Runtime_s",
        ],
        "Value": [
            len(df),
            len(train_df),
            len(val_df),
            len(test_df),
            len(eval_df),
            str(args.ga_seeds),
            f"{metrics['Accuracy']:.4f}",
            f"{metrics['Precision']:.4f}",
            f"{metrics['Recall']:.4f}",
            f"{metrics['F1 Score']:.4f}",
            f"{metrics['ROC-AUC']:.4f}",
            f"{metrics['Specificity']:.4f}",
            f"{shap_runtime:.2f}",
            f"{total_xai_time:.2f}",
        ],
    }
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_path, index=False)
    print(f"[Summary] Saved final summary to {summary_path}")

    print("\n" + "=" * 65)
    print("           PIPELINE EXECUTION COMPLETE!")
    print("=" * 65)
    print(f"Results have been saved to '{config.RESULTS_DIR}/':")
    print(f"  - Classification metrics:  {config.RESULTS_DIR}/confusion_matrix.png, roc_curve.png")
    print(f"  - Spatial analysis:       {config.RESULTS_DIR}/spatial_grid_overlay.png, region_importance_heatmap.png")
    print(f"  - GA artifacts:           {config.RESULTS_DIR}/ga_fitness_*.png, pareto_front_*.png")
    print(f"  - Comparisons:           {config.COMPARISONS_DIR}/aggregate_results.csv")
    print(f"  - Detailed results:       {config.RESULTS_DIR}/detailed_results.csv")
    print(f"  - Configuration:          {config.RESULTS_DIR}/experiment_config.json")
    print(f"  - Final summary:          {config.RESULTS_DIR}/final_summary.csv")
    print(f"\nRuntimes:")
    print(f"  - SHAP computation:       {shap_runtime:.2f} seconds")
    print(f"  - Total XAI evaluation:   {total_xai_time:.2f} seconds")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
