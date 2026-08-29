import os
from typing import Tuple, Union
import numpy as np
import pandas as pd
from PIL import Image

import config


def load_image(
    filepath: str,
    target_size: Tuple[int, int] = (config.IMG_SIZE, config.IMG_SIZE),
) -> np.ndarray:
    """Loads a 2D image file, converts to grayscale, and resizes to target_size."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Image file not found: '{filepath}'")

    try:
        with Image.open(filepath) as img:
            img_gray = img.convert("L")
            img_resized = img_gray.resize(target_size, resample=Image.Resampling.BILINEAR)
            img_array = np.asarray(img_resized, dtype=np.float32)
            img_normalized = img_array / 255.0
            img_tensor = np.expand_dims(img_normalized, axis=-1)
            return img_tensor
    except Exception as e:
        raise RuntimeError(f"Failed to load image at '{filepath}': {str(e)}")


def preprocess_image(
    image_input: Union[str, Image.Image, np.ndarray],
    target_size: Tuple[int, int] = (config.IMG_SIZE, config.IMG_SIZE),
) -> np.ndarray:
    """General 2D preprocessor returning a normalized (128, 128, 1) float32 array."""
    if isinstance(image_input, str):
        return load_image(image_input, target_size=target_size)

    if isinstance(image_input, Image.Image):
        img_gray = image_input.convert("L")
        img_resized = img_gray.resize(target_size, resample=Image.Resampling.BILINEAR)
        img_array = np.asarray(img_resized, dtype=np.float32) / 255.0
        return np.expand_dims(img_array, axis=-1)

    if isinstance(image_input, np.ndarray):
        arr = image_input.astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=-1)
        elif arr.ndim == 3 and arr.shape[-1] > 1:
            arr = np.mean(arr, axis=-1, keepdims=True)
        return arr

    raise TypeError(f"Unsupported image input type: {type(image_input)}")


def create_data_arrays(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Loads all 2D images referenced in the DataFrame."""
    if "filepath" not in df.columns or "binary_label" not in df.columns:
        raise KeyError("DataFrame must contain 'filepath' and 'binary_label' columns.")

    num_samples = len(df)
    X = np.zeros((num_samples, config.IMG_SIZE, config.IMG_SIZE, config.CHANNELS), dtype=np.float32)
    y = np.zeros(num_samples, dtype=np.float32)

    for idx, row in df.iterrows():
        filepath = row["filepath"]
        label = row["binary_label"]
        X[idx] = load_image(filepath)
        y[idx] = float(label)

    return X, y
