import os
from typing import Dict, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
import tensorflow as tf

import config
from preprocessing import preprocess_image


def calculate_specificity(y_true: np.ndarray, y_pred_binary: np.ndarray) -> float:
    """Calculates Specificity: TN / (TN + FP)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0


def generate_confusion_matrix(
    y_true: np.ndarray,
    y_pred_binary: np.ndarray,
    save_path: str = "results/confusion_matrix.png",
) -> np.ndarray:
    """Generates and saves 2D Confusion Matrix heatmap."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred_binary)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal (0)", "Demented (1)"],
        yticklabels=["Normal (0)", "Demented (1)"],
        cbar=True,
    )
    plt.title("Confusion Matrix - 2D Baseline", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Class", fontsize=11)
    plt.ylabel("True Class", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    return cm


def generate_roc_curve(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    auc_score: float,
    save_path: str = "results/roc_curve.png",
) -> None:
    """Generates and saves 2D ROC curve."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#1f77b4", lw=2.5, label=f"2D Baseline (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], color="#7f7f7f", lw=1.5, linestyle="--", label="Random Chance")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.title("ROC Curve - 2D Baseline", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def evaluate_model(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    results_dir: str = config.RESULTS_DIR,
) -> Dict[str, float]:
    """Computes all classification metrics for the 2D baseline."""
    os.makedirs(results_dir, exist_ok=True)

    probs = model.predict(X_test, verbose=0).flatten()
    preds_binary = (probs >= 0.5).astype(int)

    acc = accuracy_score(y_test, preds_binary)
    prec = precision_score(y_test, preds_binary, zero_division=0)
    rec = recall_score(y_test, preds_binary, zero_division=0)
    f1 = f1_score(y_test, preds_binary, zero_division=0)
    
    try:
        auc = roc_auc_score(y_test, probs)
    except Exception:
        auc = 0.5

    spec = calculate_specificity(y_test, preds_binary)

    metrics_dict = {
        "Accuracy": float(acc),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1 Score": float(f1),
        "ROC-AUC": float(auc),
        "Specificity": float(spec),
    }

    generate_confusion_matrix(y_test, preds_binary, os.path.join(results_dir, "confusion_matrix.png"))
    generate_roc_curve(y_test, probs, auc, os.path.join(results_dir, "roc_curve.png"))

    print("\n" + "=" * 45)
    print("         2D MODEL EVALUATION RESULTS")
    print("=" * 45)
    for k, v in metrics_dict.items():
        print(f"  {k:<20}: {v:.4f}")
    print("=" * 45 + "\n")

    return metrics_dict
