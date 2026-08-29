import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, metrics
import config


def create_data_augmentation() -> tf.keras.Sequential:
    """Creates a Keras sequential model for data augmentation."""
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal", seed=config.RANDOM_SEED),
            layers.RandomRotation(0.05, seed=config.RANDOM_SEED),
            layers.RandomZoom(0.05, seed=config.RANDOM_SEED),
        ],
        name="data_augmentation",
    )


def build_cnn(
    input_shape=(config.IMG_SIZE, config.IMG_SIZE, config.CHANNELS),
    learning_rate: float = config.LEARNING_RATE,
    use_augmentation: bool = True,
) -> tf.keras.Model:
    """Constructs and compiles the 2D CNN architecture."""
    inputs = layers.Input(shape=input_shape, name="mri_input")
    x = inputs

    if use_augmentation:
        augmentation_layer = create_data_augmentation()
        x = augmentation_layer(x)

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.ReLU(name="relu1")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.ReLU(name="relu2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.ReLU(name="relu3")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)

    # Global Average Pooling & Dense Classification Head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, activation="relu", name="dense_features")(x)
    x = layers.Dropout(0.4, seed=config.RANDOM_SEED, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="binary_output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="GA_SHAP_Alzheimer_CNN_2D")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            metrics.BinaryAccuracy(name="accuracy"),
            metrics.Precision(name="precision"),
            metrics.Recall(name="recall"),
            metrics.AUC(name="auc"),
        ],
    )

    return model
