import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, metrics
import config


def create_data_augmentation() -> tf.keras.Sequential:
    """
    Creates a Keras sequential model for data augmentation.

    Data augmentation increases the effective size of the training dataset by
    applying random transformations to images during training. This helps prevent
    overfitting and improves model generalization.

    Augmentations applied:
    - Random horizontal flip (mirror left-right)
    - Random rotation (±5% of 360° = ±18°)
    - Random zoom (±5%)

    All augmentations use a fixed seed for reproducibility.

    Returns:
        Keras Sequential model containing augmentation layers
    """
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal", seed=config.RANDOM_SEED),  # Randomly flip left-right
            layers.RandomRotation(0.05, seed=config.RANDOM_SEED),     # Randomly rotate ±18°
            layers.RandomZoom(0.05, seed=config.RANDOM_SEED),          # Randomly zoom in/out ±5%
        ],
        name="data_augmentation",
    )


def build_cnn(
    input_shape=(config.IMG_SIZE, config.IMG_SIZE, config.CHANNELS),
    learning_rate: float = config.LEARNING_RATE,
    use_augmentation: bool = True,
) -> tf.keras.Model:
    """
    Constructs and compiles the 2D CNN architecture for Alzheimer's classification.

    Architecture:
    - Input: 128x128x1 grayscale MRI image
    - Optional: Data augmentation (flip, rotate, zoom)
    - Conv Block 1: 32 filters, 3x3 kernel -> BatchNorm -> ReLU -> MaxPool (2x2)
    - Conv Block 2: 64 filters, 3x3 kernel -> BatchNorm -> ReLU -> MaxPool (2x2)
    - Conv Block 3: 128 filters, 3x3 kernel -> BatchNorm -> ReLU -> MaxPool (2x2)
    - Global Average Pooling: Reduces spatial dimensions to 1D
    - Dense: 128 units with ReLU activation
    - Dropout: 40% dropout to prevent overfitting
    - Output: 1 unit with sigmoid activation (binary classification)

    The network progressively increases filter count (32 -> 64 -> 128) to capture
    increasingly complex features. MaxPooling reduces spatial dimensions (128 -> 64 -> 32 -> 16).

    Args:
        input_shape: Shape of input images (default: 128x128x1)
        learning_rate: Learning rate for Adam optimizer
        use_augmentation: Whether to include data augmentation

    Returns:
        Compiled Keras Model ready for training
    """
    # Input layer
    inputs = layers.Input(shape=input_shape, name="mri_input")
    x = inputs

    # Optional data augmentation (only during training)
    if use_augmentation:
        augmentation_layer = create_data_augmentation()
        x = augmentation_layer(x)

    # =======================
    # CONVOLUTIONAL BLOCK 1
    # =======================
    # 32 filters, 3x3 kernel, 'same' padding preserves spatial dimensions
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1")(x)
    # Batch normalization stabilizes training by normalizing layer inputs
    x = layers.BatchNormalization(name="bn1")(x)
    # ReLU activation introduces non-linearity
    x = layers.ReLU(name="relu1")(x)
    # Max pooling reduces spatial dimensions by half (128 -> 64)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)

    # =======================
    # CONVOLUTIONAL BLOCK 2
    # =======================
    # 64 filters (double previous) to capture more complex features
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.ReLU(name="relu2")(x)
    # Max pooling reduces spatial dimensions by half (64 -> 32)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)

    # =======================
    # CONVOLUTIONAL BLOCK 3
    # =======================
    # 128 filters (double previous) for high-level features
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.ReLU(name="relu3")(x)
    # Max pooling reduces spatial dimensions by half (32 -> 16)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)

    # =======================
    # CLASSIFICATION HEAD
    # =======================
    # Global Average Pooling: averages over spatial dimensions (16x16 -> 1)
    # More parameter-efficient than Flatten + Dense
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    # Dense layer with 128 units for feature combination
    x = layers.Dense(128, activation="relu", name="dense_features")(x)
    # Dropout: randomly set 40% of units to 0 during training to prevent overfitting
    x = layers.Dropout(0.4, seed=config.RANDOM_SEED, name="dropout")(x)
    # Output layer: single unit with sigmoid for binary classification (0-1 probability)
    outputs = layers.Dense(1, activation="sigmoid", name="binary_output")(x)

    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs, name="GA_SHAP_Alzheimer_CNN_2D")

    # Compile the model with optimizer, loss function, and metrics
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),  # Adam optimizer
        loss="binary_crossentropy",  # Binary cross-entropy loss for classification
        metrics=[
            metrics.BinaryAccuracy(name="accuracy"),  # Classification accuracy
            metrics.Precision(name="precision"),       # Precision (TP / (TP + FP))
            metrics.Recall(name="recall"),             # Recall (TP / (TP + FN))
            metrics.AUC(name="auc"),                   # Area under ROC curve
        ],
    )

    return model
