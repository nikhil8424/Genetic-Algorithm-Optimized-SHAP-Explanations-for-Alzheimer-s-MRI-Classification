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
    """
    Loads a 2D image file, converts to grayscale, and resizes to target_size.

    This function performs the following preprocessing steps:
    1. Opens the image file
    2. Converts to grayscale (single channel)
    3. Resizes to target dimensions (default: 128x128)
    4. Normalizes pixel values to [0, 1] range
    5. Adds channel dimension for CNN input (H, W, 1)

    Args:
        filepath: Path to the image file
        target_size: Target dimensions (height, width) for resizing

    Returns:
        Preprocessed image as numpy array with shape (height, width, 1)

    Raises:
        FileNotFoundError: If image file doesn't exist
        RuntimeError: If image loading fails
    """
    # Check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Image file not found: '{filepath}'")

    try:
        # Open image and convert to grayscale
        with Image.open(filepath) as img:
            img_gray = img.convert("L")  # 'L' mode = grayscale
            # Resize to target dimensions using bilinear interpolation
            img_resized = img_gray.resize(target_size, resample=Image.Resampling.BILINEAR)
            # Convert to numpy array (float32)
            img_array = np.asarray(img_resized, dtype=np.float32)
            # Normalize pixel values from [0, 255] to [0, 1]
            img_normalized = img_array / 255.0
            # Add channel dimension: (H, W) -> (H, W, 1) for CNN
            img_tensor = np.expand_dims(img_normalized, axis=-1)
            return img_tensor
    except Exception as e:
        raise RuntimeError(f"Failed to load image at '{filepath}': {str(e)}")


def preprocess_image(
    image_input: Union[str, Image.Image, np.ndarray],
    target_size: Tuple[int, int] = (config.IMG_SIZE, config.IMG_SIZE),
) -> np.ndarray:
    """
    General 2D preprocessor returning a normalized (128, 128, 1) float32 array.

    This function handles multiple input types and converts them to a standardized
    format suitable for the CNN model. It can process:
    - File paths (strings)
    - PIL Image objects
    - NumPy arrays

    Args:
        image_input: Input image (file path, PIL Image, or numpy array)
        target_size: Target dimensions (height, width) for resizing

    Returns:
        Preprocessed image as numpy array with shape (height, width, 1)

    Raises:
        TypeError: If input type is not supported
    """
    # Case 1: Input is a file path (string)
    if isinstance(image_input, str):
        return load_image(image_input, target_size=target_size)

    # Case 2: Input is a PIL Image object
    if isinstance(image_input, Image.Image):
        img_gray = image_input.convert("L")  # Convert to grayscale
        img_resized = img_gray.resize(target_size, resample=Image.Resampling.BILINEAR)
        img_array = np.asarray(img_resized, dtype=np.float32) / 255.0  # Normalize
        return np.expand_dims(img_array, axis=-1)  # Add channel dimension

    # Case 3: Input is a numpy array
    if isinstance(image_input, np.ndarray):
        arr = image_input.astype(np.float32)
        # Normalize if values are in [0, 255] range
        if arr.max() > 1.0:
            arr = arr / 255.0
        # Add channel dimension if 2D array
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=-1)
        # Convert RGB to grayscale if 3D array with multiple channels
        elif arr.ndim == 3 and arr.shape[-1] > 1:
            arr = np.mean(arr, axis=-1, keepdims=True)
        return arr

    # Unsupported input type
    raise TypeError(f"Unsupported image input type: {type(image_input)}")


def create_data_arrays(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads all 2D images referenced in the DataFrame.

    This function iterates through the DataFrame and loads all images into memory.
    It creates two arrays:
    - X: Image data with shape (num_samples, height, width, channels)
    - y: Binary labels with shape (num_samples,)

    This is used to prepare data for training/evaluation where all images need
    to be in memory at once.

    Args:
        df: DataFrame with 'filepath' and 'binary_label' columns

    Returns:
        Tuple of (X, y) where:
        - X: numpy array of images with shape (n, 128, 128, 1)
        - y: numpy array of labels with shape (n,)

    Raises:
        KeyError: If DataFrame doesn't have required columns
    """
    # Validate DataFrame has required columns
    if "filepath" not in df.columns or "binary_label" not in df.columns:
        raise KeyError("DataFrame must contain 'filepath' and 'binary_label' columns.")

    # Pre-allocate arrays for efficiency
    num_samples = len(df)
    X = np.zeros((num_samples, config.IMG_SIZE, config.IMG_SIZE, config.CHANNELS), dtype=np.float32)
    y = np.zeros(num_samples, dtype=np.float32)

    # Load each image and label
    for idx, row in df.iterrows():
        filepath = row["filepath"]
        label = row["binary_label"]
        X[idx] = load_image(filepath)  # Load and preprocess image
        y[idx] = float(label)         # Store label as float

    return X, y
