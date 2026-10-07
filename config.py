# =============================================================================
# 2D GA-SHAP BASELINE CONFIGURATION: Hyperparameters, Paths & GA Settings
# =============================================================================
# This file contains all configuration parameters for the pipeline including:
# - Directory paths for data, models, and results
# - Image preprocessing settings
# - Dataset split ratios
# - CNN training hyperparameters
# - Genetic algorithm parameters
# - SHAP explainer settings
# - Class label mappings

import os

# =============================================================================
# DATASET & DIRECTORY CONFIGURATION
# =============================================================================
# These define where to find/save data, models, and results

DATASET_DIR = "data/OriginalDataset"      # Path to input MRI dataset folder
RESULTS_DIR = "results"                    # Main directory for all output results
SHAP_RESULTS_DIR = "results/shap"          # Directory for SHAP explanation visualizations
FINAL_RESULTS_DIR = "results/final"        # Directory for final comparison plots
COMPARISONS_DIR = "results/comparisons"    # Directory for method comparison CSVs
MODELS_DIR = "models"                      # Directory for saving trained models
MODEL_SAVE_PATH = "models/baseline_2d_model.keras"  # Path to save/load the trained CNN model

# =============================================================================
# 2D IMAGE & PREPROCESSING SETTINGS
# =============================================================================
# These control how images are loaded and preprocessed

IMG_SIZE = 128                              # Resize all MRI images to 128x128 pixels
CHANNELS = 1                                # Use grayscale (1 channel) for MRI images
VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")  # Acceptable image file formats

# =============================================================================
# DATASET SPLITS & REPRODUCIBILITY
# =============================================================================
# These control how the dataset is split and ensure reproducible results

TEST_SIZE = 0.16                            # 16% of data reserved for testing
VALIDATION_SIZE = 0.16                      # 16% of data reserved for validation
                                            # Remaining 68% used for training
RANDOM_SEED = 42                            # Fixed seed for reproducible random operations

# =============================================================================
# 2D CNN TRAINING HYPERPARAMETERS
# =============================================================================
# These control the training process of the convolutional neural network

BATCH_SIZE = 32                             # Number of samples per gradient update
EPOCHS = 30                                 # Maximum number of training epochs
LEARNING_RATE = 0.001                       # Learning rate for Adam optimizer
EARLY_STOPPING_PATIENCE = 7                # Stop training if validation loss doesn't improve for 7 epochs
REDUCE_LR_PATIENCE = 3                      # Reduce learning rate if validation loss doesn't improve for 3 epochs

# =============================================================================
# 2D SPATIAL GRID CONFIGURATION
# =============================================================================
# The image is divided into regions for the genetic algorithm to work with
# 8x8 grid = 64 regions total, each region is 16x16 pixels (128/8 = 16)

GRID_ROWS = 8                               # Number of rows in the spatial grid
GRID_COLS = 8                               # Number of columns in the spatial grid
NUM_REGIONS = GRID_ROWS * GRID_COLS        # Total number of spatial regions (64)
REGION_PIXEL_SIZE = IMG_SIZE // GRID_ROWS   # Pixel size of each region (16x16)

# =============================================================================
# MULTI-OBJECTIVE NSGA-II GENETIC ALGORITHM HYPERPARAMETERS
# =============================================================================
# These control the evolutionary optimization process

GA_POPULATION_SIZE = 20                     # Number of individuals (solutions) in each generation
GA_GENERATIONS = 10                         # Number of generations to evolve
GA_CROSSOVER_PROB = 0.7                     # Probability of crossover between two parents (70%)
GA_MUTATION_PROB = 0.2                      # Probability of mutation in offspring (20%)
GA_BIT_FLIP_PROB = 0.08                     # Probability of flipping each bit during mutation (8%)
GA_EVAL_SAMPLES = 20                        # Number of test samples to evaluate with GA
GA_SEEDS = [42]                             # Random seeds for GA runs (can add more for multiple runs)

# =============================================================================
# COMPOSITE SCALAR WEIGHTS FOR COMPARATIVE REPORTING
# =============================================================================
# These weights combine the three objectives into a single fitness score
# Used for reporting and comparison purposes (not for NSGA-II itself)

FITNESS_ALPHA = 1.0 / 3.0  # Weight for prediction preservation (f1) - 33.3%
FITNESS_BETA  = 1.0 / 3.0  # Weight for SHAP importance retained (f2) - 33.3%
FITNESS_GAMMA = 1.0 / 3.0  # Sparsity penalty (f3) - 33.3% (penalized, not rewarded)

# =============================================================================
# SHAP EXPLAINER SETTINGS
# =============================================================================
# These control the SHAP (SHapley Additive exPlanations) computation

SHAP_BACKGROUND_SIZE = 25                   # Number of background samples for SHAP explainer
                                            # Used as reference for computing SHAP values

# =============================================================================
# CLASS DEFINITIONS & MAPPINGS
# =============================================================================
# Original dataset has 4 classes, which we map to 2 binary classes
# Binary classification: Normal (0) vs Demented (1)

ORIGINAL_CLASSES = [
    "NonDemented",          # No signs of Alzheimer's
    "VeryMildDemented",     # Very early stage
    "MildDemented",         # Mild stage
    "ModerateDemented",     # Moderate stage
]

# Map 4 original classes to 2 binary classes
CLASS_MAPPING = {
    "NonDemented": 0,        # Normal (no dementia)
    "VeryMildDemented": 1,   # Demented (early stage)
    "MildDemented": 1,       # Demented (mild stage)
    "ModerateDemented": 1,   # Demented (moderate stage)
}

# Human-readable names for binary classes
BINARY_CLASS_NAMES = {
    0: "Normal",
    1: "Demented",
}
