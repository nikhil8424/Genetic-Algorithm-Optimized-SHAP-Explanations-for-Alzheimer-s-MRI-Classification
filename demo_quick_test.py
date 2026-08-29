import os
import sys
import numpy as np
import tensorflow as tf

import config
from dataset import load_dataset, split_dataset
from preprocessing import preprocess_image, load_image
from training import load_trained_model
from shap_explainer import create_shap_explainer, compute_shap_values
from region_analysis import calculate_region_shap_scores
from genetic_algorithm import run_genetic_algorithm, calculate_objectives


def run_quick_verification():
    print("=" * 60)
    print("      2D GA-SHAP STANDALONE QUICK VERIFICATION TEST")
    print("=" * 60)

    # 1. Check Python & TensorFlow
    print(f"[1/6] Python Version: {sys.version.split()[0]}")
    print(f"[1/6] TensorFlow Version: {tf.__version__}")

    # 2. Check Dataset
    print("\n[2/6] Verifying Dataset...")
    if not os.path.exists(config.DATASET_DIR):
        print(f"Error: Dataset not found at '{config.DATASET_DIR}'.")
        return False
    df = load_dataset(config.DATASET_DIR)
    print(f"      Successfully indexed {len(df)} images across 4 classes.")

    # 3. Check Pretrained Model
    print("\n[3/6] Verifying Pretrained Model...")
    model_path = config.MODEL_SAVE_PATH
    if not os.path.exists(model_path):
        print(f"Error: Model not found at '{model_path}'.")
        return False
    model = load_trained_model(model_path)
    print(f"      Model loaded. Total params: {model.count_params():,}")

    # 4. Check Single Image Preprocessing & Inference
    print("\n[4/6] Verifying Single Image Preprocessing & Model Prediction...")
    sample_row = df.iloc[0]
    img = load_image(sample_row["filepath"])
    prob = float(model.predict(np.expand_dims(img, axis=0), verbose=0)[0][0])
    print(f"      Sample Class: {sample_row['original_class']} | Pred Prob: {prob:.4f}")

    # 5. Check SHAP Explainer
    print("\n[5/6] Verifying SHAP Explainer...")
    bg_data = np.zeros((3, config.IMG_SIZE, config.IMG_SIZE, config.CHANNELS), dtype=np.float32)
    explainer = create_shap_explainer(model, bg_data)
    test_batch = np.expand_dims(img, axis=0)
    raw_s, abs_s = compute_shap_values(explainer, test_batch)
    print(f"      SHAP values calculated. Shape: {abs_s.shape}")

    # 6. Check Mini Genetic Algorithm Run
    print("\n[6/6] Verifying Multi-Objective NSGA-II Genetic Algorithm...")
    reg_scores, df_scores = calculate_region_shap_scores(abs_s[0])
    best_chrom, hist_df, details, p_front = run_genetic_algorithm(
        model=model,
        image=img,
        original_prob=prob,
        region_shap_scores=reg_scores,
        population_size=10,
        generations=2,
    )
    print(f"      GA completed 2 generations. Best chromosome selected {sum(best_chrom)}/64 regions.")
    print(f"      Pareto front size: {len(p_front)} solutions.")
    print(f"      Prediction Preservation (f1): {details['f1_preservation']:.4f}")
    print(f"      SHAP Retained (f2): {details['f2_shap_importance']:.4f}")
    print(f"      Compactness (f3): {details['f3_compactness']:.4f}")

    print("\n" + "=" * 60)
    print("  ALL 6 VERIFICATION CHECKS PASSED SUCCESSFULLY! ")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = run_quick_verification()
    sys.exit(0 if success else 1)
