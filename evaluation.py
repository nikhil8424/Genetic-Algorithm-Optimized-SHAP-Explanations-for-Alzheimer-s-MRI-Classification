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
    """
    Calculates Specificity: TN / (TN + FP).

    Specificity measures the proportion of actual negatives that are correctly identified.
    It's also known as the True Negative Rate. For medical diagnosis, specificity is
    important to avoid false positives (incorrectly diagnosing healthy people as sick).

    Args:
        y_true: True binary labels (0 or 1)
        y_pred_binary: Predicted binary labels (0 or 1)

    Returns:
        Specificity score between 0 and 1
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0


def generate_confusion_matrix(
    y_true: np.ndarray,
    y_pred_binary: np.ndarray,
    save_path: str = "results/confusion_matrix.png",
) -> np.ndarray:
    """
    Generates and saves 2D Confusion Matrix heatmap.

    A confusion matrix shows the performance of a classification model:
    - True Negatives (TN): Correctly predicted as Normal
    - False Positives (FP): Incorrectly predicted as Demented (actually Normal)
    - False Negatives (FN): Incorrectly predicted as Normal (actually Demented)
    - True Positives (TP): Correctly predicted as Demented

    The heatmap visualizes these counts with darker colors indicating higher values.

    Args:
        y_true: True binary labels (0 or 1)
        y_pred_binary: Predicted binary labels (0 or 1)
        save_path: Path to save the confusion matrix plot

    Returns:
        Confusion matrix as numpy array [[TN, FP], [FN, TP]]
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred_binary)

    # Create heatmap visualization
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,      # Show numbers in cells
        fmt="d",         # Integer format
        cmap="Blues",    # Blue color scheme
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
    """
    Generates and saves 2D ROC curve.

    ROC (Receiver Operating Characteristic) curve shows the trade-off between:
    - True Positive Rate (TPR = Recall) on Y-axis
    - False Positive Rate (FPR = 1 - Specificity) on X-axis

    A perfect classifier has AUC = 1.0 (curve goes to top-left corner)
    A random classifier has AUC = 0.5 (diagonal line)
    Higher AUC indicates better model performance

    Args:
        y_true: True binary labels (0 or 1)
        y_pred_probs: Predicted probabilities (continuous values between 0 and 1)
        auc_score: Area Under the Curve score
        save_path: Path to save the ROC curve plot
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Calculate FPR and TPR at various threshold values
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)

    # Create ROC curve plot
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#1f77b4", lw=2.5, label=f"2D Baseline (AUC = {auc_score:.4f})")
    # Diagonal line represents random classifier
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
    """
    Computes all classification metrics for the 2D baseline.

    This function evaluates the trained model on the test set and computes:
    - Accuracy: Overall correctness
    - Precision: How many predicted positives are actually positive
    - Recall: How many actual positives were correctly predicted
    - F1 Score: Harmonic mean of precision and recall
    - ROC-AUC: Area under the ROC curve (ranking quality)
    - Specificity: How many actual negatives were correctly predicted

    It also generates visualization plots (confusion matrix, ROC curve).

    Args:
        model: Trained Keras model to evaluate
        X_test: Test images (n_samples, 128, 128, 1)
        y_test: Test labels (n_samples,)
        results_dir: Directory to save evaluation plots

    Returns:
        Dictionary containing all computed metrics
    """
    os.makedirs(results_dir, exist_ok=True)

    # Get model predictions (probabilities)
    probs = model.predict(X_test, verbose=0).flatten()
    # Convert probabilities to binary predictions (threshold = 0.5)
    preds_binary = (probs >= 0.5).astype(int)

    # Compute classification metrics
    acc = accuracy_score(y_test, preds_binary)
    prec = precision_score(y_test, preds_binary, zero_division=0)
    rec = recall_score(y_test, preds_binary, zero_division=0)
    f1 = f1_score(y_test, preds_binary, zero_division=0)

    # Compute ROC-AUC (handle edge cases)
    try:
        auc = roc_auc_score(y_test, probs)
    except Exception:
        auc = 0.5

    # Compute specificity
    spec = calculate_specificity(y_test, preds_binary)

    # Store metrics in dictionary
    metrics_dict = {
        "Accuracy": float(acc),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1 Score": float(f1),
        "ROC-AUC": float(auc),
        "Specificity": float(spec),
    }

    # Generate visualization plots
    generate_confusion_matrix(y_test, preds_binary, os.path.join(results_dir, "confusion_matrix.png"))
    generate_roc_curve(y_test, probs, auc, os.path.join(results_dir, "roc_curve.png"))

    # Print results
    print("\n" + "=" * 45)
    print("         2D MODEL EVALUATION RESULTS")
    print("=" * 45)
    for k, v in metrics_dict.items():
        print(f"  {k:<20}: {v:.4f}")
    print("=" * 45 + "\n")

    return metrics_dict
