import os
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.callbacks import History

import config


def plot_training_history(
    history: History,
    save_path: str = "results/training_curves.png",
) -> None:
    """Plots and saves training and validation Loss and Accuracy trajectories."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    hist = history.history
    epochs_range = range(1, len(hist["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    ax1.plot(epochs_range, hist["loss"], color="#d62728", lw=2, label="Train Loss")
    if "val_loss" in hist:
        ax1.plot(epochs_range, hist["val_loss"], color="#ff7f0e", lw=2, linestyle="--", label="Val Loss")
    ax1.set_title("2D Baseline Loss", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Binary Crossentropy", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(frameon=True)

    acc_key = "accuracy" if "accuracy" in hist else "binary_accuracy"
    val_acc_key = f"val_{acc_key}"

    if acc_key in hist:
        ax2.plot(epochs_range, hist[acc_key], color="#1f77b4", lw=2, label="Train Accuracy")
    if val_acc_key in hist:
        ax2.plot(epochs_range, hist[val_acc_key], color="#2ca02c", lw=2, linestyle="--", label="Val Accuracy")
    ax2.set_title("2D Baseline Accuracy", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Accuracy", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(frameon=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_final_comparison(
    original_img: np.ndarray,
    abs_shap_map: np.ndarray,
    shap_topk_mask: np.ndarray,
    ga_mask: np.ndarray,
    ga_masked_img: np.ndarray,
    sample_info: Dict,
    save_path: str,
) -> None:
    """Generates 2D 5-panel visual comparison."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    img_2d = np.squeeze(original_img)
    shap_2d = np.squeeze(abs_shap_map)
    shap_topk_2d = np.squeeze(shap_topk_mask)
    ga_mask_2d = np.squeeze(ga_mask)
    ga_masked_2d = np.squeeze(ga_masked_img)

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.8))

    axes[0].imshow(img_2d, cmap="gray")
    axes[0].set_title(f"1. Original MRI\nTrue: {sample_info.get('true_class', 'N/A')}", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    vmax = max(np.percentile(shap_2d, 99), 1e-5)
    im_shap = axes[1].imshow(shap_2d, cmap="magma", vmin=0, vmax=vmax)
    axes[1].set_title("2. 2D SHAP Importance", fontsize=10, fontweight="bold")
    axes[1].axis("off")
    plt.colorbar(im_shap, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(shap_topk_2d, cmap="Blues", vmin=0, vmax=1)
    axes[2].set_title("3. SHAP Top-K Mask", fontsize=10, fontweight="bold")
    axes[2].axis("off")

    axes[3].imshow(ga_mask_2d, cmap="Greens", vmin=0, vmax=1)
    axes[3].set_title("4. GA Optimized Mask", fontsize=10, fontweight="bold")
    axes[3].axis("off")

    axes[4].imshow(ga_masked_2d, cmap="gray")
    axes[4].set_title("5. GA Masked MRI", fontsize=10, fontweight="bold")
    axes[4].axis("off")

    plt.suptitle(
        f"2D Explanation Comparison - Sample #{sample_info.get('index', 1)} "
        f"(True: {sample_info.get('true_class', 'N/A')}, Pred: {sample_info.get('pred_class', 'N/A')}, p={sample_info.get('pred_prob', 0.0):.3f})",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
