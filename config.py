# =============================================================================
# 2D GA-SHAP BASELINE CONFIGURATION: Hyperparameters, Paths & GA Settings
# =============================================================================

import os

# Dataset & Directory Configuration
DATASET_DIR = "data/OriginalDataset"
RESULTS_DIR = "results"
SHAP_RESULTS_DIR = "results/shap"
FINAL_RESULTS_DIR = "results/final"
MODELS_DIR = "models"
MODEL_SAVE_PATH = "models/baseline_2d_model.keras"

# 2D Image & Preprocessing Settings
IMG_SIZE = 128
CHANNELS = 1
VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# Dataset Splits & Reproducibility
TEST_SIZE = 0.16
VALIDATION_SIZE = 0.16
RANDOM_SEED = 42

# 2D CNN Training Hyperparameters
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 7
REDUCE_LR_PATIENCE = 3

# 2D Spatial Grid Configuration (8x8 = 64 spatial regions, 16x16 pixels each)
GRID_ROWS = 8
GRID_COLS = 8
NUM_REGIONS = GRID_ROWS * GRID_COLS
REGION_PIXEL_SIZE = IMG_SIZE // GRID_ROWS

# Multi-Objective NSGA-II Genetic Algorithm Hyperparameters
GA_POPULATION_SIZE = 40
GA_GENERATIONS = 15
GA_CROSSOVER_PROB = 0.7
GA_MUTATION_PROB = 0.2
GA_BIT_FLIP_PROB = 0.08
GA_DEMO_IMAGES = 3

# Composite Scalar Weights for Comparative Reporting
FITNESS_ALPHA = 1.0 / 3.0  # Weight for prediction preservation (f1)
FITNESS_BETA  = 1.0 / 3.0  # Weight for SHAP importance retained (f2)
FITNESS_GAMMA = 1.0 / 3.0  # Sparsity penalty (f3)

# SHAP Explainer Settings
SHAP_BACKGROUND_SIZE = 25

# Class Definitions & Mappings
ORIGINAL_CLASSES = [
    "NonDemented",
    "VeryMildDemented",
    "MildDemented",
    "ModerateDemented",
]

CLASS_MAPPING = {
    "NonDemented": 0,        # Normal
    "VeryMildDemented": 1,   # Demented
    "MildDemented": 1,       # Demented
    "ModerateDemented": 1,   # Demented
}

BINARY_CLASS_NAMES = {
    0: "Normal",
    1: "Demented",
}
