# CIFAR-10 Convolutional Autoencoder

A PyTorch implementation of convolutional autoencoders for image reconstruction and colorization on the CIFAR-10 dataset. This project includes three exercises exploring different aspects of autoencoders.

## 📋 Table of Contents
- [🔧 Installation](#🔧-installation)
- [📁 Project Structure](#📁-project-structure)
- [🚀 Quick Start](#🚀-quick-start)
- [📊 Exercise 1: Basic Image Reconstruction](#📊-exercise-1-basic-image-reconstruction)
- [🔬 Exercise 2: Architecture Experiments](#🔬-exercise-2-architecture-experiments)
- [🎨 Exercise 3: Image Colorization](#🎨-exercise-3-image-colorization)
- [🔍 Troubleshooting](#🔍-troubleshooting)

---

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- conda or pip

### Setup Environment

**Using conda (recommended):**
```bash
conda create -n <name> python=3.12
conda activate <name>
pip install torch torchvision matplotlib numpy scikit-image
```

**Using pip:**
```bash
python -m venv <name>
source <name>/bin/activate  # On Windows: acml_lab2\Scripts\activate
pip install torch torchvision matplotlib numpy scikit-image
```

### Verify Installation
```bash
python -c "import torch; print('PyTorch version:', torch.__version__)"
```

---

## 📁 Project Structure

```
Lab2/
├── train.py                          # Exercise 1, 3 training
├── predict.py                        # Exercise 1, 3 predict
├── constants.py                      # Hyperparameters (epochs, learning rate, etc.)
│
├── autoencoderCAE.py                 # RGB reconstruction model (Ex 1)
├── autoencoderCAE_colourisation.py   # Colorization model (Ex 3)
│
├── test/                             # Exercise 2: Architecture experiments
│   ├── __init__.py
│   ├── test_autoencoderCAE.py       # Flexible architecture implementation
│   ├── experiments.py                # Architecture configurations
│   ├── run_experiments.py            # Run all experiments
│   ├── analyze_results.py            # Generate plots and tables
│   └── compare.py                    # Visual comparison tool
│
├── models/                           # Saved model checkpoints (created after training)
└── data/                            # CIFAR-10 dataset (auto-downloaded)
```

---

## 🚀 Quick Start

### For Exercises 1 and 3 (train.py / predict.py)

Run directly with Python:
```bash
python train.py
python predict.py
```

### For Exercise 2 (test/ directory)

Must run as a Python module:
```bash
python -m test.run_experiments
python -m test.analyze_results
python -m test.compare
```

**Why?** The `test/` directory is a Python package. Running with `-m` ensures Python correctly resolves imports between files.

---

## 📊 Exercise 1: Basic Image Reconstruction

**Goal:** Train an autoencoder to compress and reconstruct 32×32 RGB images.

### Step 1: Configure Exercise

Open `train.py` and set:
```python
EXERCISE = 1
```

Open `predict.py` and set:
```python
EXERCISE = 1
```

### Step 2: Train the Model

```bash
python train.py
```

**What happens:**
- Downloads CIFAR-10 dataset (~170MB, first run only)
- Trains for 10 epochs (adjust in `constants.py`)
- Saves best model as `cifar_model_reconstruction.pth`
- Generates `loss_plot_reconstruction.png`

**Expected output:**
```
============================================================
TRAINING RGB RECONSTRUCTION MODEL (Exercise 1/2)
============================================================
Using device: cpu

Epoch 1/10
  Train Loss: 0.010268
  Val Loss:   0.006866
  ✓ Best model saved!
...
```

### Step 3: Evaluate the Model

```bash
python predict.py
```

**Output files:**
- `reconstruction_results.png` - Visual comparison of original vs reconstructed images

---

## 🔬 Exercise 2: Architecture Experiments

**Goal:** Systematically test different autoencoder architectures and analyze their performance.

### Option A: Quick Single Experiment

Use the same process as Exercise 1 with `EXERCISE = 2` (they share the same code).

### Option B: Run All Experiments

This trains 9 different architectures and compares them:

#### Step 1: Train All Models

```bash
python -m test.run_experiments
```

**⚠️ Important:** Must use `python -m test.run_experiments`, not `python test/run_experiments.py`

**What happens:**
- Trains 9 architectures: baseline, shallow, deep, wide, narrow, various latent sizes, large kernels
- Each experiment takes ~5-10 minutes on CPU
- Saves models to `models/` directory
- Generates `experiment_results_[timestamp].json`

**Architectures tested:**
1. **Latent size variations:** Tiny (256), Small (512), Baseline (1024), Large (2048)
2. **Depth variations:** Shallow (2 layers), Baseline (3 layers), Deep (4 layers)
3. **Width variations:** Narrow (0.5× channels), Baseline (1×), Wide (2× channels)
4. **Kernel size:** 3×3 (baseline) vs 5×5

#### Step 2: Analyze Results

First, **update the JSON filename** in `test/analyze_results.py`:

```python
# Line ~403 in test/analyze_results.py
results = load_results("test/experiment_results_20251117_194802.json")  # Change this!
```

Replace with your actual filename (check the console output from Step 1).

Then run:
```bash
python -m test.analyze_results
```

**Generates 8 plots:**
- `plot_1_latent_vs_loss.png` - Latent space size vs reconstruction quality
- `plot_2_compression_tradeoff.png` - Compression ratio analysis
- `plot_3_training_curves.png` - Training dynamics comparison
- `plot_4_parameters_vs_performance.png` - Model complexity analysis
- `plot_5_experiment_categories.png` - Category-wise comparison
- `plot_6_overfitting_analysis.png` - Generalization gaps
- `plot_7_train_val_test_comparison.png` - Full evaluation
- `plot_8_generalization_analysis.png` - Test set performance

Plus a comprehensive summary table in the console.

#### Step 3: Visual Comparison (Optional)

Compare reconstructions from all models on the same images:

```bash
python -m test.compare
```

Generates:
- `single_image_comparison_[class].png` - One image reconstructed by all models
- `multiple_images_comparison.png` - Multiple images side-by-side

### Understanding Exercise 2 Results

**Key findings:**
- Larger latent spaces = better reconstruction but less compression
- Shallow networks work best with minimal compression
- Deep networks struggle with aggressive compression
- Wide networks (more channels) improve quality but use more parameters
- 5×5 kernels perform worse than 3×3 for 32×32 images

---

## 🎨 Exercise 3: Image Colorization

**Goal:** Train a model to predict colors for grayscale images (this task intentionally fails to demonstrate autoencoder limitations).

### Step 1: Configure Exercise

Open `train.py` and set:
```python
EXERCISE = 3
```

Open `predict.py` and set:
```python
EXERCISE = 3
```

### Step 2: Train the Model

```bash
python train.py
```

**What happens:**
- Converts RGB images to LAB color space
- Trains model to predict color (ab channels) from grayscale (L channel)
- Saves `cifar_model_colorization.pth`
- Generates `loss_plot_colorization.png` (shows overfitting!)

**Training notes:**
- Takes longer than Exercise 1 (~20-30 minutes for 30 epochs)
- You'll see validation loss plateau after ~10 epochs (expected)
- Final test loss will be higher (~0.008-0.009) than Exercise 1

### Step 3: Evaluate the Model

```bash
python predict.py
```

**Output:**
- `colorization_results.png` - Grayscale input, ground truth, and predicted colors

---

## 🔍 Troubleshooting

### "ModuleNotFoundError: No module named 'test'"

You're trying to run test scripts directly. Use the module syntax:

❌ Wrong: `python test/run_experiments.py`  
✅ Correct: `python -m test.run_experiments`

### "RuntimeError: Error(s) in loading state_dict"

Your saved model has a different architecture than the code expects.

**Solution 1: Identify your model**
```bash
python identify_model.py cifar_model_reconstruction.pth
```

This tells you exactly what to put in `predict.py`.

**Solution 2: Train a new model**
```bash
# Make sure EXERCISE matches in both files
python train.py  # Creates new model matching current architecture
python predict.py
```

### "FileNotFoundError: [model].pth"

You need to train before evaluating:
```bash
python train.py   # Creates the model file
python predict.py # Loads the model file
```

### "CUDA out of memory"

Reduce batch size in `constants.py`:
```python
BATCH_SIZE = 16  # Default is 32
```

### Colorization produces gray images

This is **expected behavior**! The colorization task is supposed to fail. Read the Exercise 3 section in the report to understand why.

### Training is too slow

**Option 1:** Reduce epochs in `constants.py`:
```python
NUM_EPOCHS = 3  # Quick testing
```

**Option 2:** Use GPU if available (automatically detected)

**Option 3:** For Exercise 2, run fewer experiments:
Edit `test/experiments.py` and modify `ALL_EXPERIMENTS` list to include only the ones you want.

---

## 📝 Customizing Hyperparameters

Edit `constants.py`:

```python
LEARNING_RATE = 0.01   # Adam optimizer learning rate
BATCH_SIZE = 32        # Training batch size
NUM_EPOCHS = 10        # Number of training epochs
```

Or override directly in `train.py` after imports:

```python
from constants import NUM_EPOCHS, LEARNING_RATE

# Override here
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
```

---

## 📊 Expected Results Summary

| Exercise | Test Loss | Time (CPU) | Main Output |
|----------|-----------|------------|-------------|
| **Exercise 1** | ~0.004-0.005 | 5-10 min | Good reconstructions |
| **Exercise 2** | Varies by architecture | 45-90 min | Comparison plots |
| **Exercise 3** | ~0.008-0.009 | 20-30 min | Poor colorization (expected) |

---

## 🎯 Common Workflows

### Workflow 1: Basic Usage (Exercise 1)
```bash
# 1. Set EXERCISE = 1 in both train.py and predict.py
# 2. Train and evaluate
python train.py
python predict.py
```

### Workflow 2: Full Analysis (Exercise 2)
```bash
# 1. Train all architectures
python -m test.run_experiments

# 2. Update JSON filename in test/analyze_results.py (line ~403)
# 3. Generate analysis plots
python -m test.analyze_results

# 4. Visual comparison (optional)
python -m test.compare
```

### Workflow 3: Colorization (Exercise 3)
```bash
# 1. Set EXERCISE = 3 in both train.py and predict.py
# 2. Train and evaluate
python train.py
python predict.py

# 3. Observe the poor results and read the report for explanation
```

---

## 💡 Tips

- **First time?** Start with Exercise 1 to understand the basics
- **GPU Training:** Automatically uses MPS (Mac), CUDA (NVIDIA), or CPU
- **Quick Testing:** Set `NUM_EPOCHS = 3` in constants.py
- **Model Files:** ~100KB each, safe to delete and retrain
- **Dataset:** CIFAR-10 downloads once to `./data/` (~170MB)

---

**Happy experimenting! 🚀**