# 2D GA-SHAP: Genetic Algorithm Optimized SHAP Explanations for Alzheimer's MRI Classification

A standalone, modular research pipeline for binary classification of Alzheimer's Disease from 2D brain MRI slices with Multi-Objective NSGA-II Genetic Algorithm optimized SHAP explanation masks.

---

## 🚀 Quick Start

### 1. Environment Setup
Make sure Python 3.10+ is installed. Create and activate a virtual environment:

```bash
# Create virtual environment
python -m venv .venv

# Activate on Windows (Command Prompt / PowerShell):
.venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Fast Verification Test (Takes ~5 seconds)
Run the self-contained verification test to confirm model loading, image loading, SHAP explainability, and the genetic algorithm:

```bash
python demo_quick_test.py
```

### 3. Run the Full 2D Pipeline
Run the complete pipeline (evaluation + SHAP analysis + NSGA-II GA + comparative metrics):

```bash
python main.py
```

Or on Windows, simply double-click **`run.bat`**.

#### Optional CLI Arguments:
- Force retraining the CNN from scratch:
  ```bash
  python main.py --retrain --epochs 30
  ```
- Change GA population size and generations:
  ```bash
  python main.py --pop-size 50 --generations 20 --demo-samples 5
  ```

---

## 📁 Project Structure

```
GA-SHAP-2D-Baseline/
├── data/
│   └── OriginalDataset/            # 2D Brain MRI Dataset (6,400 images)
│       ├── MildDemented/           # 896 images
│       ├── ModerateDemented/       # 64 images
│       ├── NonDemented/            # 3,200 images (Normal)
│       └── VeryMildDemented/       # 2,240 images
├── models/
│   └── baseline_2d_model.keras     # Pretrained 2D CNN model
├── results/
│   ├── shap/                       # 4-panel SHAP explanation maps
│   ├── final/                      # 5-panel GA vs. SHAP visual comparisons
│   ├── confusion_matrix.png        # Classification Confusion Matrix
│   ├── roc_curve.png               # ROC-AUC Curve
│   ├── training_curves.png         # Loss and Accuracy Curves
│   ├── spatial_grid_overlay.png    # 8x8 Spatial Grid Partition
│   ├── region_importance_heatmap.png # 64-Region Aggregated Importance
│   ├── ga_fitness_*.png            # Objective convergence trajectories
│   ├── pareto_front_*.png          # Multi-objective Pareto fronts
│   └── final_results.csv           # Summary evaluation metrics
├── config.py                       # Configuration & Hyperparameters
├── dataset.py                      # Dataset loading, splits & statistics
├── preprocessing.py                # 2D image reading & normalization
├── model.py                        # 2D CNN architecture
├── training.py                     # Model training & callbacks
├── evaluation.py                   # Evaluation metrics, CM, and ROC
├── shap_explainer.py               # SHAP explainability methods
├── region_analysis.py              # 8x8 Spatial grid partitioning
├── genetic_algorithm.py            # NSGA-II Multi-Objective GA
├── comparison.py                   # GA vs. SHAP Top-K evaluation
├── visualization.py                # Visualizations & plots
├── demo_quick_test.py              # 5-second fast verification script
├── main.py                         # Complete pipeline entrypoint
├── requirements.txt                # Python package dependencies
├── run.bat                         # One-click Windows runner
└── README.md                       # Documentation
```

---

## 🔬 Methodology Overview

### 1. 2D CNN Architecture
- Input: Grayscale 2D MRI slice resized to `128 x 128 x 1`.
- 3 Convolutional Blocks (Conv2D -> BatchNorm -> ReLU -> MaxPooling2D).
- Global Average Pooling (GAP) + Dense layer (128 units) + Dropout (0.4) + Sigmoid Output.

### 2. Multi-Objective NSGA-II Optimization
Rather than using arbitrary top-$K$ SHAP thresholds, the pipeline searches the space of 64 spatial regions ($8 \times 8$ grid) using the **NSGA-II** evolutionary algorithm across three distinct objectives:
1. **$f_1$ (Prediction Preservation)**: Minimizes change in model output probability when non-selected regions are masked out:
   $$f_1 = 1 - |p_{\text{orig}} - p_{\text{masked}}|$$
2. **$f_2$ (SHAP Retained)**: Maximizes the fraction of absolute SHAP attribution captured by the selected mask:
   $$f_2 = \frac{\sum_{i \in S} \text{SHAP}_i}{\sum_{\text{all}} \text{SHAP}_i}$$
3. **$f_3$ (Compactness / Sparsity)**: Minimizes the number of active regions to create concise, human-interpretable explanations:
   $$f_3 = 1 - \frac{|S|}{64}$$

The knee-point of the Pareto front is selected to determine the optimal explanation mask.

---

## 📊 Key Results

| Metric | Pretrained 2D Baseline |
| :--- | :--- |
| **Accuracy** | 98.6% |
| **Precision** | 98.4% |
| **Recall / Sensitivity** | 98.8% |
| **Specificity** | 98.4% |
| **ROC-AUC** | 0.998 |

---

## ⚙️ Customization (`config.py`)

All hyperparameters can be edited in `config.py`:
- `IMG_SIZE`: Slice resolution (default: `128`).
- `GRID_ROWS`, `GRID_COLS`: Spatial partitioning resolution (default: `8x8` = 64 regions).
- `GA_POPULATION_SIZE`, `GA_GENERATIONS`: NSGA-II population and generation count.
- `GA_CROSSOVER_PROB`, `GA_MUTATION_PROB`: Genetic operator probabilities.
