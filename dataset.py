import os
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np

import config


def verify_dataset_structure(dataset_dir: str = config.DATASET_DIR) -> None:
    """
    Verifies that the dataset directory and all required class subfolders exist.

    This function checks if the main dataset directory exists and if all four
    required class subdirectories (NonDemented, VeryMildDemented, MildDemented,
    ModerateDemented) are present. Raises an error if any are missing.

    Args:
        dataset_dir: Path to the main dataset directory

    Raises:
        FileNotFoundError: If dataset directory or any class subfolder is missing
    """
    # Check if main dataset directory exists
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(
            f"Dataset directory not found at '{dataset_dir}'. "
            f"Please place your dataset in '{dataset_dir}' or update config.DATASET_DIR."
        )

    # Check if all required class subdirectories exist
    missing_folders = []
    for class_name in config.ORIGINAL_CLASSES:
        class_folder = os.path.join(dataset_dir, class_name)
        if not os.path.exists(class_folder) or not os.path.isdir(class_folder):
            missing_folders.append(class_folder)

    # Raise error if any folders are missing
    if missing_folders:
        raise FileNotFoundError(
            f"The following required class folders are missing:\n"
            + "\n".join(f" - {f}" for f in missing_folders)
            + f"\nExpected all four subdirectories: {', '.join(config.ORIGINAL_CLASSES)}"
        )


def create_dataframe(dataset_dir: str = config.DATASET_DIR) -> pd.DataFrame:
    """
    Scans the dataset directory, collects valid image paths, and creates a DataFrame.

    This function iterates through each class subdirectory, collects all valid image
    files (jpg, jpeg, png), and creates a DataFrame with columns for filepath,
    original class, binary label, and binary class name.

    Args:
        dataset_dir: Path to the main dataset directory

    Returns:
        DataFrame with columns: filepath, original_class, binary_label, binary_class_name

    Raises:
        ValueError: If no valid image files are found
    """
    # First verify the dataset structure
    verify_dataset_structure(dataset_dir)

    # Collect all valid image files
    records = []
    for class_name in config.ORIGINAL_CLASSES:
        class_dir = os.path.join(dataset_dir, class_name)
        file_list = sorted(os.listdir(class_dir))

        # Process each file in the class directory
        for fname in file_list:
            # Check if file has valid image extension
            if fname.lower().endswith(config.VALID_IMAGE_EXTENSIONS):
                full_path = os.path.join(class_dir, fname)
                records.append({
                    "filepath": full_path,                                    # Full path to image file
                    "original_class": class_name,                            # Original 4-class label
                    "binary_label": config.CLASS_MAPPING[class_name],        # Binary label (0 or 1)
                    "binary_class_name": config.BINARY_CLASS_NAMES[config.CLASS_MAPPING[class_name]]  # Human-readable binary label
                })

    # Create DataFrame from collected records
    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError(
            f"No valid image files ({', '.join(config.VALID_IMAGE_EXTENSIONS)}) "
            f"were found in the subfolders of '{dataset_dir}'."
        )
    return df


def create_binary_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps original 4-class categorical labels to binary labels (0 = Normal, 1 = Demented).

    This function converts the 4-class classification problem (NonDemented,
    VeryMildDemented, MildDemented, ModerateDemented) into a binary classification
    problem (Normal vs Demented) using the mapping defined in config.

    Args:
        df: DataFrame with original_class column

    Returns:
        DataFrame with added binary_label and binary_class_name columns
    """
    df_copy = df.copy()
    # Map original class to binary label (0 or 1)
    if "binary_label" not in df_copy.columns:
        df_copy["binary_label"] = df_copy["original_class"].map(config.CLASS_MAPPING)
    # Map binary label to human-readable name
    if "binary_class_name" not in df_copy.columns:
        df_copy["binary_class_name"] = df_copy["binary_label"].map(config.BINARY_CLASS_NAMES)
    return df_copy


def load_dataset(dataset_dir: str = config.DATASET_DIR) -> pd.DataFrame:
    """
    Loads dataset, creates DataFrame, and maps binary labels.

    This is the main entry point for loading the dataset. It:
    1. Scans the dataset directory for image files
    2. Creates a DataFrame with file paths and labels
    3. Maps 4-class labels to binary labels

    Args:
        dataset_dir: Path to the main dataset directory

    Returns:
        DataFrame with columns: filepath, original_class, binary_label, binary_class_name
    """
    df = create_dataframe(dataset_dir)
    df = create_binary_labels(df)
    return df


def print_dataset_statistics(df: pd.DataFrame) -> None:
    """
    Computes and prints dynamic dataset statistics from the loaded DataFrame.

    This function displays:
    - Count of samples in each of the 4 original classes
    - Count of samples in each of the 2 binary classes
    - Total number of samples

    Args:
        df: DataFrame with dataset information
    """
    print("\n" + "=" * 45)
    print("           DATASET STATISTICS (2D BASELINE)")
    print("=" * 45)

    # Print counts for each original class
    orig_counts = df["original_class"].value_counts()
    for class_name in config.ORIGINAL_CLASSES:
        count = orig_counts.get(class_name, 0)
        print(f"  {class_name:<20}: {count:>6}")

    print("-" * 45)
    # Print counts for binary classes
    binary_counts = df["binary_label"].value_counts()
    normal_count = binary_counts.get(0, 0)
    demented_count = binary_counts.get(1, 0)
    total_count = len(df)

    print(f"  Normal (0)          : {normal_count:>6}")
    print(f"  Demented (1)        : {demented_count:>6}")
    print(f"  Total Samples       : {total_count:>6}")
    print("=" * 45 + "\n")


def split_dataset(
    df: pd.DataFrame,
    test_size: float = config.TEST_SIZE,
    val_size: float = config.VALIDATION_SIZE,
    random_state: int = config.RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs stratified 68% Training / 16% Validation / 16% Test split.

    This function splits the dataset into three parts while maintaining the
    class distribution (stratified sampling). This ensures each split has a
    similar proportion of Normal and Demented samples.

    Args:
        df: DataFrame containing the full dataset
        test_size: Fraction of data to use for testing (default: 0.16 = 16%)
        val_size: Fraction of data to use for validation (default: 0.16 = 16%)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_df, val_df, test_df) DataFrames
    """
    # First split: separate test set from the rest
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,                    # 16% for testing
        stratify=df["binary_label"],            # Maintain class balance
        random_state=random_state,
        shuffle=True,
    )

    # Second split: separate validation set from training set
    # Adjust ratio because we're splitting from the remaining 84%
    adjusted_val_ratio = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=adjusted_val_ratio,          # 16% of total for validation
        stratify=train_val_df["binary_label"],  # Maintain class balance
        random_state=random_state,
        shuffle=True,
    )

    # Reset indices for clean DataFrames
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Print split summary
    print(f"Split Summary (Stratified):")
    print(f"  Training Set   : {len(train_df):>5} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Validation Set : {len(val_df):>5} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test Set       : {len(test_df):>5} samples ({len(test_df)/len(df)*100:.1f}%)")
    print()

    return train_df, val_df, test_df


def select_stratified_evaluation_samples(
    test_df: pd.DataFrame,
    n_samples: int = 100,
    random_state: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """
    Selects a stratified subset of test samples for XAI evaluation.

    This function selects a balanced subset of test samples for explainable AI
    evaluation. It attempts to select equal numbers of Normal and Demented samples
    (50/50 split) to ensure fair evaluation across both classes.

    Args:
        test_df: DataFrame containing the test set
        n_samples: Total number of samples to select (default: 100)
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with stratified evaluation samples
    """
    # Target: half from each class
    n_per_class = n_samples // 2

    # Separate by class
    normal_df = test_df[test_df["binary_label"] == 0]
    demented_df = test_df[test_df["binary_label"] == 1]

    # Check available samples
    available_normal = len(normal_df)
    available_demented = len(demented_df)

    # Adjust if insufficient samples available
    actual_normal = min(n_per_class, available_normal)
    actual_demented = min(n_per_class, available_demented)

    # Set random seed for reproducibility
    np.random.seed(random_state)

    # Randomly sample from each class
    selected_normal = normal_df.sample(n=actual_normal, random_state=random_state) if actual_normal > 0 else pd.DataFrame()
    selected_demented = demented_df.sample(n=actual_demented, random_state=random_state) if actual_demented > 0 else pd.DataFrame()

    # Combine and shuffle
    eval_df = pd.concat([selected_normal, selected_demented], ignore_index=True)
    eval_df = eval_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Print selection summary
    print(f"Evaluation Sample Selection (Stratified):")
    print(f"  Target samples per class: {n_per_class}")
    print(f"  Normal selected         : {actual_normal}")
    print(f"  Demented selected       : {actual_demented}")
    print(f"  Total evaluation samples : {len(eval_df)}")
    print()

    return eval_df


def generate_synthetic_mri_dataset(
    dataset_dir: str = config.DATASET_DIR,
    samples_per_class: int = 25,
) -> None:
    """
    Utility to synthesize 2D brain MRI-like image slices for baseline testing.

    This function creates synthetic MRI images when real data is not available.
    It generates images with brain-like features including:
    - Skull outline (elliptical)
    - Brain tissue (with noise)
    - Ventricles (fluid-filled spaces that enlarge in dementia)
    - Hippocampus (brain region that shrinks in dementia)

    The severity of dementia is simulated by:
    - Enlarging ventricles (more severe = larger ventricles)
    - Shrinking hippocampus (more severe = smaller hippocampus)

    Args:
        dataset_dir: Directory where synthetic dataset will be created
        samples_per_class: Number of synthetic images to generate per class
    """
    # Create main dataset directory
    os.makedirs(dataset_dir, exist_ok=True)

    # Generate images for each class
    for class_name in config.ORIGINAL_CLASSES:
        class_folder = os.path.join(dataset_dir, class_name)
        os.makedirs(class_folder, exist_ok=True)

        # Skip if enough images already exist
        existing = [f for f in os.listdir(class_folder) if f.lower().endswith(config.VALID_IMAGE_EXTENSIONS)]
        if len(existing) >= samples_per_class:
            continue

        # Set random seed for this class (different seed per class)
        np.random.seed(config.RANDOM_SEED + hash(class_name) % 1000)
        size = config.IMG_SIZE

        # Generate synthetic images
        for i in range(samples_per_class):
            # Create coordinate grid
            y, x = np.ogrid[:size, :size]
            cy, cx = size / 2, size / 2  # Center of image

            # Create skull outline (ellipse)
            mask_skull = ((x - cx)**2 / (0.42 * size)**2 + (y - cy)**2 / (0.46 * size)**2) <= 1.0
            # Create brain tissue (slightly smaller ellipse)
            mask_brain = ((x - cx)**2 / (0.38 * size)**2 + (y - cy)**2 / (0.42 * size)**2) <= 1.0

            # Add noise to simulate brain tissue
            noise = np.random.normal(0.55, 0.08, (size, size))
            img_arr = np.zeros((size, size), dtype=np.float32)
            img_arr[mask_brain] = noise[mask_brain]

            # Add skull rim (brighter tissue)
            skull_rim = mask_skull & (~mask_brain)
            img_arr[skull_rim] = np.random.normal(0.85, 0.05, (size, size))[skull_rim]

            # Ventricles (fluid-filled spaces) - enlarge with dementia severity
            vent_size_multiplier = 1.0
            if class_name == "VeryMildDemented":
                vent_size_multiplier = 1.35  # 35% larger
            elif class_name == "MildDemented":
                vent_size_multiplier = 1.85  # 85% larger
            elif class_name == "ModerateDemented":
                vent_size_multiplier = 2.40  # 140% larger

            # Create left and right ventricles (ellipses)
            v_left = ((x - (cx - 10))**2 / (4 * vent_size_multiplier)**2 + (y - (cy - 4))**2 / (14 * vent_size_multiplier)**2) <= 1.0
            v_right = ((x - (cx + 10))**2 / (4 * vent_size_multiplier)**2 + (y - (cy - 4))**2 / (14 * vent_size_multiplier)**2) <= 1.0
            # Ventricles appear dark (cerebrospinal fluid)
            img_arr[v_left | v_right] = 0.08 + np.random.normal(0.0, 0.02, (size, size))[v_left | v_right]

            # Hippocampus (memory center) - shrinks with dementia severity
            if class_name in ["MildDemented", "ModerateDemented"]:
                hip_left = ((x - (cx - 24))**2 / 5**2 + (y - (cy + 18))**2 / 8**2) <= 1.0
                hip_right = ((x - (cx + 24))**2 / 5**2 + (y - (cy + 18))**2 / 8**2) <= 1.0
                # Hippocampus appears darker when shrunk (tissue loss)
                img_arr[hip_left | hip_right] *= 0.45

            # Clip values to valid range [0, 1]
            img_arr = np.clip(img_arr, 0.0, 1.0)
            # Convert to uint8 for saving
            img_uint8 = (img_arr * 255.0).astype(np.uint8)

            # Save as JPEG image
            pil_img = Image.fromarray(img_uint8, mode="L")
            save_path = os.path.join(class_folder, f"mri_sample_{i+1:03d}.jpg")
            pil_img.save(save_path, "JPEG", quality=95)
