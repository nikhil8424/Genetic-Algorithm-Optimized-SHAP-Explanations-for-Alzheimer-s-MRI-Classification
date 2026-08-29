import os
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np

import config


def verify_dataset_structure(dataset_dir: str = config.DATASET_DIR) -> None:
    """Verifies that the dataset directory and all required class subfolders exist."""
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(
            f"Dataset directory not found at '{dataset_dir}'. "
            f"Please place your dataset in '{dataset_dir}' or update config.DATASET_DIR."
        )

    missing_folders = []
    for class_name in config.ORIGINAL_CLASSES:
        class_folder = os.path.join(dataset_dir, class_name)
        if not os.path.exists(class_folder) or not os.path.isdir(class_folder):
            missing_folders.append(class_folder)

    if missing_folders:
        raise FileNotFoundError(
            f"The following required class folders are missing:\n"
            + "\n".join(f" - {f}" for f in missing_folders)
            + f"\nExpected all four subdirectories: {', '.join(config.ORIGINAL_CLASSES)}"
        )


def create_dataframe(dataset_dir: str = config.DATASET_DIR) -> pd.DataFrame:
    """Scans the dataset directory, collects valid image paths, and creates a DataFrame."""
    verify_dataset_structure(dataset_dir)

    records = []
    for class_name in config.ORIGINAL_CLASSES:
        class_dir = os.path.join(dataset_dir, class_name)
        file_list = sorted(os.listdir(class_dir))

        for fname in file_list:
            if fname.lower().endswith(config.VALID_IMAGE_EXTENSIONS):
                full_path = os.path.join(class_dir, fname)
                records.append({
                    "filepath": full_path,
                    "original_class": class_name,
                    "binary_label": config.CLASS_MAPPING[class_name],
                    "binary_class_name": config.BINARY_CLASS_NAMES[config.CLASS_MAPPING[class_name]]
                })

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError(
            f"No valid image files ({', '.join(config.VALID_IMAGE_EXTENSIONS)}) "
            f"were found in the subfolders of '{dataset_dir}'."
        )
    return df


def create_binary_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Maps original 4-class categorical labels to binary labels (0 = Normal, 1 = Demented)."""
    df_copy = df.copy()
    if "binary_label" not in df_copy.columns:
        df_copy["binary_label"] = df_copy["original_class"].map(config.CLASS_MAPPING)
    if "binary_class_name" not in df_copy.columns:
        df_copy["binary_class_name"] = df_copy["binary_label"].map(config.BINARY_CLASS_NAMES)
    return df_copy


def load_dataset(dataset_dir: str = config.DATASET_DIR) -> pd.DataFrame:
    """Loads dataset, creates DataFrame, and maps binary labels."""
    df = create_dataframe(dataset_dir)
    df = create_binary_labels(df)
    return df


def print_dataset_statistics(df: pd.DataFrame) -> None:
    """Computes and prints dynamic dataset statistics from the loaded DataFrame."""
    print("\n" + "=" * 45)
    print("           DATASET STATISTICS (2D BASELINE)")
    print("=" * 45)
    
    orig_counts = df["original_class"].value_counts()
    for class_name in config.ORIGINAL_CLASSES:
        count = orig_counts.get(class_name, 0)
        print(f"  {class_name:<20}: {count:>6}")
        
    print("-" * 45)
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
    """Performs stratified 70% Training / 15% Validation / 15% Test split."""
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["binary_label"],
        random_state=random_state,
        shuffle=True,
    )

    adjusted_val_ratio = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=adjusted_val_ratio,
        stratify=train_val_df["binary_label"],
        random_state=random_state,
        shuffle=True,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"Split Summary (Stratified):")
    print(f"  Training Set   : {len(train_df):>5} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Validation Set : {len(val_df):>5} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test Set       : {len(test_df):>5} samples ({len(test_df)/len(df)*100:.1f}%)")
    print()

    return train_df, val_df, test_df


def generate_synthetic_mri_dataset(
    dataset_dir: str = config.DATASET_DIR,
    samples_per_class: int = 25,
) -> None:
    """Utility to synthesize 2D brain MRI-like image slices for baseline testing."""
    os.makedirs(dataset_dir, exist_ok=True)
    for class_name in config.ORIGINAL_CLASSES:
        class_folder = os.path.join(dataset_dir, class_name)
        os.makedirs(class_folder, exist_ok=True)
        
        existing = [f for f in os.listdir(class_folder) if f.lower().endswith(config.VALID_IMAGE_EXTENSIONS)]
        if len(existing) >= samples_per_class:
            continue

        np.random.seed(config.RANDOM_SEED + hash(class_name) % 1000)
        size = config.IMG_SIZE

        for i in range(samples_per_class):
            y, x = np.ogrid[:size, :size]
            cy, cx = size / 2, size / 2

            mask_skull = ((x - cx)**2 / (0.42 * size)**2 + (y - cy)**2 / (0.46 * size)**2) <= 1.0
            mask_brain = ((x - cx)**2 / (0.38 * size)**2 + (y - cy)**2 / (0.42 * size)**2) <= 1.0
            
            noise = np.random.normal(0.55, 0.08, (size, size))
            img_arr = np.zeros((size, size), dtype=np.float32)
            img_arr[mask_brain] = noise[mask_brain]
            
            skull_rim = mask_skull & (~mask_brain)
            img_arr[skull_rim] = np.random.normal(0.85, 0.05, (size, size))[skull_rim]
            
            vent_size_multiplier = 1.0
            if class_name == "VeryMildDemented":
                vent_size_multiplier = 1.35
            elif class_name == "MildDemented":
                vent_size_multiplier = 1.85
            elif class_name == "ModerateDemented":
                vent_size_multiplier = 2.40

            v_left = ((x - (cx - 10))**2 / (4 * vent_size_multiplier)**2 + (y - (cy - 4))**2 / (14 * vent_size_multiplier)**2) <= 1.0
            v_right = ((x - (cx + 10))**2 / (4 * vent_size_multiplier)**2 + (y - (cy - 4))**2 / (14 * vent_size_multiplier)**2) <= 1.0
            img_arr[v_left | v_right] = 0.08 + np.random.normal(0.0, 0.02, (size, size))[v_left | v_right]
            
            if class_name in ["MildDemented", "ModerateDemented"]:
                hip_left = ((x - (cx - 24))**2 / 5**2 + (y - (cy + 18))**2 / 8**2) <= 1.0
                hip_right = ((x - (cx + 24))**2 / 5**2 + (y - (cy + 18))**2 / 8**2) <= 1.0
                img_arr[hip_left | hip_right] *= 0.45

            img_arr = np.clip(img_arr, 0.0, 1.0)
            img_uint8 = (img_arr * 255.0).astype(np.uint8)
            
            pil_img = Image.fromarray(img_uint8, mode="L")
            save_path = os.path.join(class_folder, f"mri_sample_{i+1:03d}.jpg")
            pil_img.save(save_path, "JPEG", quality=95)
