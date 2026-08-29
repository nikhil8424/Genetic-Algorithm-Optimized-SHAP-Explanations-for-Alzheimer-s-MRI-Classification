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
    """Trains 2D CNN with EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint callbacks."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=1e-6,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    print(f"\nStarting 2D CNN Model Training ({epochs} epochs max, batch size {batch_size})...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    return history


def save_model(model: tf.keras.Model, filepath: str = config.MODEL_SAVE_PATH) -> None:
    """Explicitly saves Keras 2D model to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    model.save(filepath)
    print(f"2D Baseline model saved to '{filepath}'")


def load_trained_model(filepath: str = config.MODEL_SAVE_PATH) -> tf.keras.Model:
    """Loads a saved 2D Keras model from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found at '{filepath}'.")

    model = tf.keras.models.load_model(filepath)
    print(f"Successfully loaded 2D model from '{filepath}'")
    return model
