# GA-SHAP Pipeline Flow: Complete Explanation

## Overview

This pipeline implements a novel approach for explainable AI (XAI) in medical image classification, specifically for Alzheimer's disease detection from MRI scans. It combines:

1. **Deep Learning**: 2D CNN for binary classification (Normal vs Demented)
2. **SHAP**: SHapley Additive exPlanations for feature importance
3. **Genetic Algorithm**: NSGA-II multi-objective optimization for region selection

The goal is to identify the most important brain regions that drive the model's prediction while maintaining prediction accuracy and using as few regions as possible.

---

## High-Level Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    1. DATASET LOADING                           │
│  - Load MRI images from 4 classes → map to 2 binary classes      │
│  - Split: 68% train / 16% validation / 16% test                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                 2. MODEL TRAINING / LOADING                      │
│  - Build 2D CNN (3 conv blocks, data augmentation)             │
│  - Train with early stopping, learning rate reduction           │
│  - Save best model based on validation loss                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   3. MODEL EVALUATION                            │
│  - Predict on test set                                          │
│  - Compute metrics: Accuracy, Precision, Recall, F1, ROC-AUC   │
│  - Generate confusion matrix and ROC curve                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│             4. SELECT EVALUATION SAMPLES                         │
│  - Choose stratified subset from test set                       │
│  - Balance: 50% Normal, 50% Demented                           │
│  - Default: 20 samples (configurable)                          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    5. SHAP EXPLANATION                           │
│  - Initialize SHAP explainer with background samples            │
│  - Compute SHAP values for each evaluation sample               │
│  - Generate importance heatmaps                                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  6. REGION ANALYSIS                              │
│  - Divide 128x128 SHAP map into 8x8 grid (64 regions)          │
│  - Compute mean SHAP score per region                           │
│  - This reduces complexity from 2^16,384 to 2^64 possibilities  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│           7. GENETIC ALGORITHM (PER SAMPLE)                      │
│  For each evaluation sample:                                   │
│  - Initialize population of 64-bit chromosomes                  │
│  - Each bit = one region (1=select, 0=don't select)            │
│  - Optimize 3 objectives:                                       │
│    • f1: Prediction Preservation (maintain model output)        │
│    • f2: SHAP Importance Retained (capture important regions)   │
│    • f3: Compactness (use few regions)                          │
│  - Run NSGA-II for 10 generations                               │
│  - Select knee point from Pareto front                          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              8. METHOD COMPARISON                               │
│  Compare 3 methods (all select same K regions):                │
│  - Random-K: Random region selection (baseline)                │
│  - SHAP Top-K: Select regions with highest SHAP scores         │
│  - GA-NSGA-II: Multi-objective optimization (proposed)         │
│  - Evaluate with: Prediction Preservation, SHAP Retention,       │
│    Compactness, Deletion AUC, Insertion AUC                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│               9. AGGREGATE & SAVE RESULTS                       │
│  - Compute mean ± std across samples and seeds                  │
│  - Save detailed per-sample results                             │
│  - Save experiment configuration                               │
│  - Generate final summary CSV                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Detailed Step-by-Step Explanation

### Step 1: Dataset Loading (`dataset.py`)

**Purpose**: Load and prepare the MRI dataset for training and evaluation.

**Process**:
1. Scan the dataset directory for image files
2. Organize by original 4 classes:
   - NonDemented
   - VeryMildDemented
   - MildDemented
   - ModerateDemented
3. Map to 2 binary classes:
   - NonDemented → Normal (0)
   - VeryMildDemented, MildDemented, ModerateDemented → Demented (1)
4. Split dataset:
   - 68% Training
   - 16% Validation
   - 16% Test
   - Stratified: maintains class balance in each split

**Key Functions**:
- `load_dataset()`: Main entry point
- `split_dataset()`: Performs stratified split
- `select_stratified_evaluation_samples()`: Selects balanced subset for XAI

**Output**: 
- `train_df`, `val_df`, `test_df`: DataFrames with file paths and labels
- Later converted to numpy arrays: `X_train`, `y_train`, etc.

---

### Step 2: Model Training (`model.py`, `training.py`)

**Purpose**: Train a CNN to classify MRI images as Normal or Demented.

**Architecture** (2D CNN):
```
Input (128x128x1)
    ↓
Data Augmentation (optional)
    ↓
Conv Block 1: Conv2D(32) → BatchNorm → ReLU → MaxPool(2x2)
    ↓ (64x64x32)
Conv Block 2: Conv2D(64) → BatchNorm → ReLU → MaxPool(2x2)
    ↓ (32x64x64)
Conv Block 3: Conv2D(128) → BatchNorm → ReLU → MaxPool(2x2)
    ↓ (16x16x128)
Global Average Pooling
    ↓ (128)
Dense(128) → ReLU → Dropout(0.4)
    ↓ (128)
Dense(1) → Sigmoid
    ↓ (1) - Probability of Demented
```

**Training Process**:
1. Build model with data augmentation (flip, rotate, zoom)
2. Compile with Adam optimizer (lr=0.001) and binary crossentropy loss
3. Train with callbacks:
   - **EarlyStopping**: Stop if validation loss doesn't improve for 7 epochs
   - **ReduceLROnPlateau**: Reduce learning rate by 50% if validation loss plateaus for 3 epochs
   - **ModelCheckpoint**: Save best model based on validation loss
4. Save trained model to `models/baseline_2d_model.keras`

**Key Functions**:
- `build_cnn()`: Constructs the CNN architecture
- `train_model()`: Trains the model with callbacks
- `load_trained_model()`: Loads a saved model

**Output**: Trained CNN model ready for inference

---

### Step 3: Model Evaluation (`evaluation.py`)

**Purpose**: Assess the trained model's performance on the test set.

**Metrics Computed**:
- **Accuracy**: Overall correctness (TP + TN) / Total
- **Precision**: TP / (TP + FP) - How many predicted positives are actually positive
- **Recall**: TP / (TP + FN) - How many actual positives were correctly predicted
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve - Ranking quality
- **Specificity**: TN / (TN + FP) - How many actual negatives were correctly predicted

**Visualizations Generated**:
- **Confusion Matrix**: Shows TP, TN, FP, FN counts
- **ROC Curve**: Shows TPR vs FPR at various thresholds

**Key Functions**:
- `evaluate_model()`: Computes all metrics and generates plots
- `generate_confusion_matrix()`: Creates confusion matrix heatmap
- `generate_roc_curve()`: Creates ROC curve plot

**Output**: Metrics dictionary + visualization plots

---

### Step 4: Select Evaluation Samples (`dataset.py`)

**Purpose**: Choose a balanced subset of test samples for XAI evaluation.

**Process**:
1. From test set, select N samples (default: 20)
2. Balance classes: N/2 Normal, N/2 Demented
3. If insufficient samples in a class, use all available
4. Random shuffle the selected samples

**Why?** 
- XAI evaluation is computationally expensive
- We want a representative, balanced subset
- Ensures fair evaluation across both classes

**Key Function**:
- `select_stratified_evaluation_samples()`: Performs stratified sampling

**Output**: `eval_df` - DataFrame with selected evaluation samples

---

### Step 5: SHAP Explanation (`shap_explainer.py`)

**Purpose**: Compute feature importance for each pixel in the MRI images.

**What is SHAP?**
- SHAP (SHapley Additive exPlanations) allocates contribution scores to each input feature
- Based on game theory: fair distribution of "credit" among features
- Values can be positive (pushes prediction toward Demented) or negative (pushes toward Normal)

**Process**:
1. Select background samples from training set (default: 25)
   - These serve as reference for computing SHAP values
2. Initialize SHAP explainer (GradientExplainer or DeepExplainer)
3. Compute SHAP values for each evaluation sample
4. Generate visualizations:
   - Original MRI
   - Signed SHAP map (red=positive, blue=negative)
   - Absolute SHAP map (magnitude of importance)
   - SHAP overlay on original MRI

**Key Functions**:
- `create_shap_explainer()`: Initializes SHAP explainer with fallbacks
- `compute_shap_values()`: Computes raw and absolute SHAP values
- `save_shap_visualizations()`: Generates 4-panel visualization

**Output**: 
- `raw_shap_batch`: Can be positive or negative (N_samples, 128, 128)
- `abs_shap_batch`: Absolute importance (N_samples, 128, 128)

---

### Step 6: Region Analysis (`region_analysis.py`)

**Purpose**: Partition the SHAP map into spatial regions for the genetic algorithm.

**Why?**
- Direct pixel-level optimization would have 2^16,384 possibilities (impossible)
- Region-level optimization has 2^64 possibilities (manageable)
- Each region is 16x16 pixels (8x8 grid = 64 regions)

**Process**:
1. Divide 128x128 SHAP map into 8x8 grid
2. Each grid cell = 16x16 pixels
3. Compute mean SHAP score for each region
4. Normalize scores to sum to 1 (represents percentage of total importance)

**Example**:
```
Region 0 (top-left):   Mean SHAP = 0.023 (2.3% of total importance)
Region 1:              Mean SHAP = 0.045 (4.5% of total importance)
...
Region 63 (bottom-right): Mean SHAP = 0.001 (0.1% of total importance)
```

**Key Functions**:
- `calculate_region_shap_scores()`: Partitions SHAP map and computes region scores
- `plot_region_grid()`: Visualizes the 8x8 grid overlay
- `plot_region_importance()`: Heatmap of region importance

**Output**: 
- `region_scores`: Array of 64 SHAP scores (one per region)
- `df_scores`: DataFrame with Region_ID, Row, Column, SHAP_Score

---

### Step 7: Genetic Algorithm (`genetic_algorithm.py`)

**Purpose**: Use multi-objective optimization to find the best region selection.

**What is NSGA-II?**
- NSGA-II (Non-dominated Sorting Genetic Algorithm II) is a multi-objective evolutionary algorithm
- It optimizes multiple conflicting objectives simultaneously
- Maintains a diverse set of solutions (Pareto front) representing different trade-offs

**Chromosome Representation**:
- Binary list of length 64 (one bit per region)
- 1 = select region, 0 = don't select region
- Example: [1, 0, 1, 0, 1, 1, 0, ...] (selects regions 0, 2, 4, 5, ...)

**Three Objectives**:

1. **f1: Prediction Preservation**
   - Goal: Maintain the model's prediction when only selected regions are kept
   - Computation: 1 - |original_prob - masked_prob|
   - Range: [0, 1] (1 = perfect preservation)

2. **f2: SHAP Importance Retained**
   - Goal: Capture as much SHAP importance as possible
   - Computation: sum(SHAP of selected regions) / sum(SHAP of all regions)
   - Range: [0, 1] (1 = all important regions selected)

3. **f3: Compactness**
   - Goal: Use as few regions as possible (sparsity)
   - Computation: 1 - (k_selected / 64)
   - Range: [0, 1] (1 = only 1 region selected, 0 = all 64 selected)

**NSGA-II Algorithm Steps**:

1. **Initialization**: Create random population of 20 chromosomes
2. **Evaluation**: Compute f1, f2, f3 for each chromosome
3. **Selection**: Use tournament selection based on dominance
4. **Crossover**: Mix parents (70% probability, two-point crossover)
5. **Mutation**: Random bit flips (20% probability, 8% per bit)
6. **Replacement**: Combine parents + offspring, select best 20 using NSGA-II
7. **Repeat** for specified generations (default: 10)
8. **Knee Point Selection**: Choose solution closest to utopia point (1, 1, 1)

**Optimizations**:
- **Evaluation Cache**: Avoid redundant model predictions by caching chromosome evaluations
- Same chromosome may appear multiple times due to crossover/mutation

**Key Functions**:
- `run_genetic_algorithm()`: Main GA execution
- `calculate_objectives()`: Computes f1, f2, f3 for a chromosome
- `chromosome_to_mask()`: Converts 64-bit chromosome to 128x128 pixel mask
- `get_pareto_front()`: Returns non-dominated solutions
- `select_knee_point()`: Selects best compromise solution

**Output**:
- `best_chromosome`: The selected knee point (list of 64 bits)
- `history_df`: GA progress across generations
- `best_details`: Detailed metrics for best solution
- `pareto_front`: All non-dominated solutions

---

### Step 8: Method Comparison (`comparison.py`, `xai_metrics.py`)

**Purpose**: Compare GA-NSGA-II with baseline methods.

**Three Methods Compared**:

1. **Random-K** (Baseline)
   - Select K regions uniformly at random
   - Provides lower bound - any intelligent method should outperform this

2. **SHAP Top-K** (Greedy Baseline)
   - Select K regions with highest SHAP scores
   - Simple greedy approach - picks most important regions according to SHAP
   - Doesn't consider prediction preservation or compactness

3. **GA-NSGA-II** (Proposed Method)
   - Multi-objective optimization
   - Considers all three objectives simultaneously
   - Finds best trade-off between them

**Evaluation Metrics**:

1. **Prediction Preservation**: f1 from objectives
2. **SHAP Retention**: f2 from objectives
3. **Compactness**: f3 from objectives
4. **Deletion AUC**: How well prediction is maintained when regions are progressively removed
   - Higher = better (prediction maintained longer)
5. **Insertion AUC**: How quickly prediction recovers when regions are progressively added
   - Higher = better (prediction recovers faster)

**Key Functions**:
- `compare_all_methods()`: Main comparison function
- `create_random_k_mask()`: Creates random baseline
- `create_shap_top_k_mask()`: Creates SHAP greedy baseline
- `compute_deletion_auc()`: Computes deletion AUC metric
- `compute_insertion_auc()`: Computes insertion AUC metric

**Output**: Comparison CSV with metrics for all three methods

---

### Step 9: Aggregate & Save Results (`main.py`)

**Purpose**: Compile results from all samples and seeds.

**Process**:
1. Aggregate results across all evaluation samples and GA seeds
2. Compute mean ± standard deviation for each metric
3. Save aggregated results to CSV
4. Save detailed per-sample results to CSV
5. Save experiment configuration to JSON (for reproducibility)
6. Generate final summary CSV with key metrics

**Output Files**:
- `results/comparisons/aggregate_results.csv`: Mean ± std across methods
- `results/detailed_results.csv`: Per-sample, per-seed detailed results
- `results/experiment_config.json`: Experiment settings
- `results/final_summary.csv`: Key metrics summary

---

## Key Innovations

1. **Multi-Objective Optimization**: Unlike single-objective methods, GA-NSGA-II balances competing objectives
2. **Region-Level Abstraction**: Reduces search space from pixel-level (2^16,384) to region-level (2^64)
3. **Comprehensive Evaluation**: Uses both objective metrics and perturbation-based metrics (Deletion/Insertion AUC)
4. **Robustness**: Multiple GA seeds ensure results are not due to random chance

---

## Expected Results

The GA-NSGA-II method should:
- **Outperform Random-K**: Significantly better on all metrics
- **Outperform SHAP Top-K**: Better balance between prediction preservation and compactness
- **Maintain High Prediction Preservation**: Close to original model prediction
- **Achieve High SHAP Retention**: Capture most important regions
- **Be Compact**: Use fewer regions than SHAP Top-K for same performance

---

## How to Run

```bash
# Run with default settings
python main.py

# Force retrain model
python main.py --retrain

# Customize GA parameters
python main.py --pop-size 30 --generations 20 --eval-samples 50

# Run with multiple GA seeds for robustness
python main.py --ga-seeds 42 123 456
```

---

## File Structure

```
.
├── main.py                 # Main pipeline orchestration
├── config.py               # Configuration parameters
├── dataset.py              # Dataset loading and splitting
├── preprocessing.py       # Image preprocessing
├── model.py               # CNN architecture
├── training.py            # Model training and loading
├── evaluation.py          # Model evaluation metrics
├── shap_explainer.py      # SHAP explanation computation
├── region_analysis.py     # Spatial region partitioning
├── genetic_algorithm.py   # NSGA-II genetic algorithm
├── comparison.py          # Method comparison
├── xai_metrics.py         # XAI evaluation metrics
├── visualization.py       # Plotting functions
├── demo_quick_test.py     # Quick verification test
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── PIPELINE_FLOW.md       # This file

data/
└── OriginalDataset/        # MRI images organized by class
    ├── NonDemented/
    ├── VeryMildDemented/
    ├── MildDemented/
    └── ModerateDemented/

results/
├── shap/                   # SHAP visualizations
├── final/                  # Final comparison plots
└── comparisons/            # Method comparison CSVs

models/
└── baseline_2d_model.keras # Saved trained model
```

---

## Common Questions

**Q: Why use 64 regions instead of pixel-level?**
A: Pixel-level would have 2^16,384 possibilities (impossible to search). Region-level has 2^64 possibilities (manageable for GA).

**Q: Why three objectives instead of one?**
A: The objectives conflict: preserving prediction (f1) requires more regions, but compactness (f3) requires fewer. Multi-objective optimization finds the best trade-off.

**Q: What is the knee point?**
A: The knee point is the solution on the Pareto front closest to the "utopia point" (1, 1, 1), representing the best compromise between all objectives.

**Q: Why use multiple GA seeds?**
A: Genetic algorithms are stochastic. Multiple seeds ensure results are robust and not due to random chance.

**Q: What if the dataset is not available?**
A: The pipeline automatically generates synthetic MRI data if the dataset directory is not found.

---

## References

- SHAP: Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. NeurIPS.
- NSGA-II: Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. IEEE TEVC.
- Deletion/Insertion AUC: Petsiuk, T., Das, A., & Saenko, K. (2018). Rise: Randomized input sampling for explanation of black-box models. BMVC.
