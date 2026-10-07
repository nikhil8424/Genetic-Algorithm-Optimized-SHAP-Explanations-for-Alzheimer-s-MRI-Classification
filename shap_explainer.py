import os
from typing import List, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import shap

import config


def create_shap_explainer(
    model: tf.keras.Model,
    background_data: np.ndarray,
) -> Union[shap.GradientExplainer, shap.DeepExplainer, object]:
    """
    Initializes a 2D SHAP explainer with robust fallbacks.

    SHAP (SHapley Additive exPlanations) explains model predictions by allocating
    contribution scores to each input feature (pixel in our case). It uses a
    reference dataset (background_data) to compute these values.

    This function tries multiple SHAP explainer implementations in order:
    1. GradientExplainer (recommended for deep learning)
    2. DeepExplainer (older implementation)
    3. Custom gradient-based fallback (if both fail)

    Args:
        model: Trained Keras model to explain
        background_data: Reference dataset for SHAP computation (n_samples, 128, 128, 1)

    Returns:
        SHAP explainer object with shap_values() method
    """
    # Try GradientExplainer first (recommended for deep learning)
    try:
        explainer = shap.GradientExplainer(model, background_data)
        return explainer
    except Exception as e_grad:
        print(f"[SHAP Notice] GradientExplainer fallback: {str(e_grad)}")
        # Fallback to DeepExplainer (older implementation)
        try:
            explainer = shap.DeepExplainer(model, background_data)
            return explainer
        except Exception as e_deep:
            print(f"[SHAP Notice] DeepExplainer fallback: {str(e_deep)}")

            # Final fallback: custom gradient-based explainer
            class GradientAttributionFallback:
                """
                Custom gradient-based explainer using TensorFlow gradients.

                This computes a simple saliency map by:
                1. Computing gradient of output w.r.t input
                2. Multiplying by difference from background
                """
                def __init__(self, target_model, bg):
                    self.model = target_model
                    self.bg = bg

                def shap_values(self, X):
                    """Compute gradient-based attribution for each image."""
                    val_list = []
                    for img in X:
                        img_expanded = np.expand_dims(img, axis=0)
                        diff = img_expanded - self.bg
                        with tf.GradientTape() as tape:
                            x_tensor = tf.convert_to_tensor(img_expanded, dtype=tf.float32)
                            tape.watch(x_tensor)
                            pred = self.model(x_tensor)
                        grads = tape.gradient(pred, x_tensor).numpy()
                        saliency = grads * diff.mean(axis=0, keepdims=True)
                        val_list.append(saliency[0])
                    return [np.array(val_list)]

            return GradientAttributionFallback(model, background_data)


def compute_shap_values(
    explainer: Union[shap.GradientExplainer, shap.DeepExplainer, object],
    test_images: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes raw and absolute SHAP values for a batch of 2D test images.

    This function uses the explainer to compute SHAP values for the test images.
    SHAP values can be positive (pushes prediction toward Demented) or negative
    (pushes prediction toward Normal). The absolute values represent the magnitude
    of importance regardless of direction.

    Args:
        explainer: SHAP explainer object
        test_images: Batch of test images (n_samples, 128, 128, 1)

    Returns:
        Tuple of (raw_shap_values, abs_shap_values)
        - raw_shap_values: Can be positive or negative (n_samples, 128, 128)
        - abs_shap_values: Absolute importance (n_samples, 128, 128)
    """
    # Compute SHAP values using the explainer
    shap_vals = explainer.shap_values(test_images)

    # Handle different SHAP output formats
    if isinstance(shap_vals, list):
        # For binary classification, SHAP returns [values_for_class_0, values_for_class_1]
        # We want values for the positive class (Demented = 1)
        if len(shap_vals) == 2:
            raw_vals = shap_vals[1]
        else:
            raw_vals = shap_vals[0]
    else:
        raw_vals = shap_vals

    # Convert to numpy array and ensure float32
    raw_vals = np.asarray(raw_vals, dtype=np.float32)
    # Remove channel dimension if present (128, 128, 1) -> (128, 128)
    if raw_vals.ndim == 4 and raw_vals.shape[-1] == 1:
        raw_vals = np.squeeze(raw_vals, axis=-1)

    # Compute absolute values (magnitude of importance)
    abs_vals = np.abs(raw_vals)
    return raw_vals, abs_vals


def save_shap_visualizations(
    image: np.ndarray,
    raw_shap: np.ndarray,
    abs_shap: np.ndarray,
    image_index: int,
    true_label: str,
    pred_label: str,
    pred_prob: float,
    save_dir: str = config.SHAP_RESULTS_DIR,
) -> None:
    """
    Generates and saves the 4-panel 2D SHAP explanation plots.

    This function creates a comprehensive visualization with 4 panels:
    1. Original MRI: The input brain scan
    2. Signed SHAP Map: Shows direction of influence (red = pushes toward Demented, blue = pushes toward Normal)
    3. Absolute SHAP Map: Shows magnitude of importance (brighter = more important)
    4. SHAP Overlay: Overlays importance heatmap on the original MRI

    Args:
        image: Original MRI image (128, 128, 1)
        raw_shap: Raw SHAP values (can be positive or negative) (128, 128)
        abs_shap: Absolute SHAP values (magnitude) (128, 128)
        image_index: Index of the sample (for filename)
        true_label: True class label (Normal or Demented)
        pred_label: Predicted class label (Normal or Demented)
        pred_prob: Predicted probability for Demented class
        save_dir: Directory to save the visualization
    """
    os.makedirs(save_dir, exist_ok=True)

    # Remove channel dimension for visualization
    img_2d = np.squeeze(image)
    raw_2d = np.squeeze(raw_shap)
    abs_2d = np.squeeze(abs_shap)

    # Create 4-panel figure
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    # Panel 1: Original MRI
    axes[0].imshow(img_2d, cmap="gray")
    axes[0].set_title(f"Original MRI\nTrue: {true_label}", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: Signed SHAP Map (red-blue colormap)
    vmax = max(np.percentile(abs_2d, 99), 1e-5)  # Use 99th percentile as max for better contrast
    im1 = axes[1].imshow(raw_2d, cmap="seismic", vmin=-vmax, vmax=vmax)
    axes[1].set_title("Signed SHAP Map\n(Red: +Demented, Blue: -)", fontsize=11, fontweight="bold")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: Absolute SHAP Map (magma colormap)
    im2 = axes[2].imshow(abs_2d, cmap="magma", vmin=0, vmax=vmax)
    axes[2].set_title("Absolute SHAP Map\n(Magnitude of Impact)", fontsize=11, fontweight="bold")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # Panel 4: SHAP Overlay on original MRI
    axes[3].imshow(img_2d, cmap="gray")
    axes[3].imshow(abs_2d, cmap="hot", alpha=0.55, vmin=0, vmax=vmax)
    axes[3].set_title(
        f"SHAP Overlay\nPred: {pred_label} (p={pred_prob:.3f})",
        fontsize=11,
        fontweight="bold",
    )
    axes[3].axis("off")

    # Add overall title
    plt.suptitle(
        f"2D SHAP Analysis - Sample #{image_index + 1}",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    # Save figure
    out_file = os.path.join(save_dir, f"sample_{image_index + 1:02d}_shap_analysis.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
