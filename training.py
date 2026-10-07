import os
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    History,
)
import numpy as np

import config


def train_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    model_path: str = config.MODEL_SAVE_PATH,
) -> History:
    """
    Trains 2D CNN with EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint callbacks.

    This function trains the CNN model using three important callbacks:
    1. EarlyStopping: Stops training if validation loss doesn't improve for N epochs
    2. ReduceLROnPlateau: Reduces learning rate if validation loss plateaus
    3. ModelCheckpoint: Saves the best model (based on validation loss) to disk

    Training process:
    - Model trains on X_train, y_train
    - Validates on X_val, y_val after each epoch
    - Callbacks monitor validation loss and take action as needed
    - Best model weights are saved to model_path

    Args:
        model: Compiled Keras model to train
        X_train: Training images (n_samples, 128, 128, 1)
        y_train: Training labels (n_samples,)
        X_val: Validation images (n_samples, 128, 128, 1)
        y_val: Validation labels (n_samples,)
        epochs: Maximum number of training epochs
        batch_size: Number of samples per gradient update
        model_path: Path to save the best model

    Returns:
        History object containing training metrics (loss, accuracy, etc.) per epoch
    """
    # Ensure directory for saving model exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Configure training callbacks
    callbacks = [
        # EarlyStopping: Stop training if validation loss doesn't improve for 7 epochs
        # restore_best_weights=True ensures we keep the best model, not the last one
        EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        # ReduceLROnPlateau: If validation loss plateaus for 3 epochs, reduce learning rate by 50%
        # This helps the model fine-tune when it's close to optimum
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,           # Reduce LR by half
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=1e-6,          # Don't reduce below 1e-6
            verbose=1,
        ),
        # ModelCheckpoint: Save the model whenever validation loss improves
        # save_best_only=True ensures we only keep the best version
        ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # Start training
    print(f"\nStarting 2D CNN Model Training ({epochs} epochs max, batch size {batch_size})...")
    history = model.fit(
        X_train,              # Training images
        y_train,              # Training labels
        validation_data=(X_val, y_val),  # Validation data for monitoring
        epochs=epochs,        # Maximum epochs (may stop early due to EarlyStopping)
        batch_size=batch_size,  # Batch size for gradient updates
        callbacks=callbacks,  # Apply the configured callbacks
        verbose=1,            # Print progress bar and metrics each epoch
    )

    return history


def save_model(model: tf.keras.Model, filepath: str = config.MODEL_SAVE_PATH) -> None:
    """
    Explicitly saves Keras 2D model to disk.

    This function saves the entire model (architecture, weights, optimizer state)
    to a .keras file. The saved model can be loaded later for inference or
    continued training.

    Args:
        model: Trained Keras model to save
        filepath: Path where the model will be saved
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    model.save(filepath)
    print(f"2D Baseline model saved to '{filepath}'")


def load_trained_model(filepath: str = config.MODEL_SAVE_PATH) -> tf.keras.Model:
    """
    Loads a saved 2D Keras model from disk.

    This function loads a previously saved model (architecture, weights, optimizer state)
    from a .keras file. The loaded model is ready for inference or continued training.

    Args:
        filepath: Path to the saved model file

    Returns:
        Loaded Keras model

    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found at '{filepath}'.")

    model = tf.keras.models.load_model(filepath)
    print(f"Successfully loaded 2D model from '{filepath}'")
    return model
