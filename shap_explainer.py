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
    """Initializes a 2D SHAP explainer with robust fallbacks."""
    try:
        explainer = shap.GradientExplainer(model, background_data)
        return explainer
    except Exception as e_grad:
        print(f"[SHAP Notice] GradientExplainer fallback: {str(e_grad)}")
        try:
            explainer = shap.DeepExplainer(model, background_data)
            return explainer
        except Exception as e_deep:
            print(f"[SHAP Notice] DeepExplainer fallback: {str(e_deep)}")
            
            class IntegratedGradientWrapper:
                def __init__(self, target_model, bg):
                    self.model = target_model
                    self.bg = bg

                def shap_values(self, X):
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

            return IntegratedGradientWrapper(model, background_data)


def compute_shap_values(
    explainer: Union[shap.GradientExplainer, shap.DeepExplainer, object],
    test_images: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes raw and absolute SHAP values for a batch of 2D test images."""
    shap_vals = explainer.shap_values(test_images)

    if isinstance(shap_vals, list):
        if len(shap_vals) == 2:
            raw_vals = shap_vals[1]
        else:
            raw_vals = shap_vals[0]
    else:
        raw_vals = shap_vals

    raw_vals = np.asarray(raw_vals, dtype=np.float32)
    if raw_vals.ndim == 4 and raw_vals.shape[-1] == 1:
        raw_vals = np.squeeze(raw_vals, axis=-1)

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
    """Generates and saves the 4-panel 2D SHAP explanation plots."""
    os.makedirs(save_dir, exist_ok=True)

    img_2d = np.squeeze(image)
    raw_2d = np.squeeze(raw_shap)
    abs_2d = np.squeeze(abs_shap)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    axes[0].imshow(img_2d, cmap="gray")
    axes[0].set_title(f"Original MRI\nTrue: {true_label}", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    vmax = max(np.percentile(abs_2d, 99), 1e-5)
    im1 = axes[1].imshow(raw_2d, cmap="seismic", vmin=-vmax, vmax=vmax)
    axes[1].set_title("Signed SHAP Map\n(Red: +Demented, Blue: -)", fontsize=11, fontweight="bold")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(abs_2d, cmap="magma", vmin=0, vmax=vmax)
    axes[2].set_title("Absolute SHAP Map\n(Magnitude of Impact)", fontsize=11, fontweight="bold")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].imshow(img_2d, cmap="gray")
    axes[3].imshow(abs_2d, cmap="hot", alpha=0.55, vmin=0, vmax=vmax)
    axes[3].set_title(
        f"SHAP Overlay\nPred: {pred_label} (p={pred_prob:.3f})",
        fontsize=11,
        fontweight="bold",
    )
    axes[3].axis("off")

    plt.suptitle(
        f"2D SHAP Analysis - Sample #{image_index + 1}",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    out_file = os.path.join(save_dir, f"sample_{image_index + 1:02d}_shap_analysis.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
