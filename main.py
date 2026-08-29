import os
import sys
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

import config
from dataset import (
    load_dataset,
    split_dataset,
    print_dataset_statistics,
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
    chromosome_to_mask,
)
from comparison import compare_ga_vs_shap, create_shap_top_k_mask
from visualization import (
    plot_training_history,
    plot_final_comparison,
)


def setup_directories() -> None:
    """Ensures all necessary output directories exist."""
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.SHAP_RESULTS_DIR, exist_ok=True)
    os.makedirs(config.FINAL_RESULTS_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="2D GA-SHAP Alzheimer's MRI Classification Pipeline")
    parser.add_argument("--retrain", action="store_true", help="Force retrain 2D CNN model from scratch")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Number of training epochs")
    parser.add_argument("--pop-size", type=int, default=config.GA_POPULATION_SIZE, help="GA population size")
    parser.add_argument("--generations", type=int, default=config.GA_GENERATIONS, help="GA generations")
    parser.add_argument("--demo-samples", type=int, default=config.GA_DEMO_IMAGES, help="Number of test demo samples")
    return parser.parse_args()


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

    # 4. SHAP Explanation Setup
    print("\n[SHAP] Computing SHAP background and test explanations...")
    bg_size = min(config.SHAP_BACKGROUND_SIZE, len(X_train))
    bg_indices = np.random.choice(len(X_train), size=bg_size, replace=False)
    bg_data = X_train[bg_indices]
    explainer = create_shap_explainer(model, bg_data)

    num_samples = min(args.demo_samples, len(X_test))
    demo_indices = list(range(num_samples))
    demo_images = X_test[demo_indices]
    raw_shap_batch, abs_shap_batch = compute_shap_values(explainer, demo_images)

    summary_records = []

    # 5. Multi-Objective Genetic Algorithm & Comparison
    for i, idx in enumerate(demo_indices):
        sample_num = i + 1
        img = demo_images[i]
        raw_shap = raw_shap_batch[i]
        abs_shap = abs_shap_batch[i]
        
        true_label = test_df.iloc[idx]["binary_class_name"]
        orig_prob = float(model.predict(np.expand_dims(img, axis=0), verbose=0)[0][0])
        pred_label = "Demented" if orig_prob >= 0.5 else "Normal"

        print(f"\n--- Processing Sample #{sample_num:02d} (True: {true_label}, Pred: {pred_label}, Prob: {orig_prob:.3f}) ---")

        # Save 4-panel SHAP visualization
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
        if i == 0:
            save_region_scores_csv(df_scores, os.path.join(config.RESULTS_DIR, "region_scores.csv"))
            plot_region_grid(img, save_path=os.path.join(config.RESULTS_DIR, "spatial_grid_overlay.png"))
            plot_region_importance(reg_scores, save_path=os.path.join(config.RESULTS_DIR, "region_importance_heatmap.png"))

        # Run NSGA-II Genetic Algorithm
        print(f"Running NSGA-II GA (Pop: {args.pop_size}, Gen: {args.generations})...")
        best_chrom, hist_df, ga_details, p_front = run_genetic_algorithm(
            model=model,
            image=img,
            original_prob=orig_prob,
            region_shap_scores=reg_scores,
            population_size=args.pop_size,
            generations=args.generations,
        )

        # Save GA artifacts
        save_ga_history_csv(hist_df, os.path.join(config.RESULTS_DIR, f"ga_history_sample_{sample_num:02d}.csv"))
        plot_ga_fitness(hist_df, os.path.join(config.RESULTS_DIR, f"ga_fitness_sample_{sample_num:02d}.png"))
        plot_pareto_front(p_front, sample_idx=sample_num, save_path=os.path.join(config.RESULTS_DIR, f"pareto_front_sample_{sample_num:02d}.png"))

        # Comparison with Top-K Baseline
        comp_df, details_dict = compare_ga_vs_shap(
            model=model,
            image=img,
            original_prob=orig_prob,
            ga_chromosome=best_chrom,
            region_shap_scores=reg_scores,
            image_idx=i,
            save_path=os.path.join(config.RESULTS_DIR, f"comparison_sample_{sample_num:02d}.csv"),
        )

        # Generate 5-panel Final Visual Comparison
        k_selected = sum(best_chrom)
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

        summary_records.append({
            "Sample": sample_num,
            "True_Class": true_label,
            "Pred_Class": pred_label,
            "Original_Prob": orig_prob,
            "Regions_Selected": k_selected,
            "GA_Preservation": ga_details["f1_preservation"],
            "GA_SHAP_Retained": ga_details["f2_shap_importance"],
            "GA_Compactness": ga_details["f3_compactness"],
            "TopK_Preservation": details_dict["SHAP_TopK"]["prediction_preservation"],
            "TopK_SHAP_Retained": details_dict["SHAP_TopK"]["shap_importance"],
        })

    # Save summary dataframe
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(os.path.join(config.RESULTS_DIR, "final_results.csv"), index=False)

    print("\n" + "=" * 65)
    print("           PIPELINE EXECUTION COMPLETE!")
    print("=" * 65)
    print(f"Results have been saved to '{config.RESULTS_DIR}/':")
    print(f"  - Confusion Matrix & ROC:    {config.RESULTS_DIR}/confusion_matrix.png, roc_curve.png")
    print(f"  - Spatial Grid & Heatmap:    {config.RESULTS_DIR}/spatial_grid_overlay.png, region_importance_heatmap.png")
    print(f"  - SHAP Visualizations:       {config.SHAP_RESULTS_DIR}/")
    print(f"  - GA Fitness & Pareto Fronts: {config.RESULTS_DIR}/ga_fitness_*.png, pareto_front_*.png")
    print(f"  - Final 5-Panel Comparisons: {config.FINAL_RESULTS_DIR}/")
    print(f"  - Metrics & Summary CSVs:    {config.RESULTS_DIR}/final_results.csv, comparison_*.csv")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
