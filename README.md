# CIFAR-10 CNN Autoencoder

A convolutional neural network autoencoder built with PyTorch to compress and reconstruct CIFAR-10 images (32×32 RGB).

## Overview

This autoencoder uses convolutional layers to compress CIFAR-10 images into a compact representation (8×8×16 latent space), then reconstructs them back to the original 32×32×3 dimensions.

**Architecture**: 32×32×3 → 8×8×16 (latent) → 32×32×3

## Features

- **CNN-based compression** using convolutional layers and max pooling
- **Symmetric architecture** with encoder and decoder
- **Image reconstruction** from compressed representations
- **Training & evaluation scripts** with visualization tools

## Quick Start

### 1. Train the Model

```bash
python train_cnn.py
```

This will:
- Download CIFAR-10 dataset automatically
- Train for 10 epochs
- Save the model as `cifar_autoencoder.pth`

### 2. Test & Visualize

```bash
python predict_cnn.py
```

This will:
- Load the trained model
- Show original vs reconstructed images
- Calculate test set loss
- Save comparison image as `reconstructions.png`

## Architecture

```
Encoder:
  Conv2D (3→8)   + ReLU  →  32×32×8
  MaxPool (2×2)           →  16×16×8
  Conv2D (8→12)  + ReLU  →  16×16×12
  MaxPool (2×2)           →  8×8×12
  Conv2D (12→16) + ReLU  →  8×8×16  (latent space)

Decoder:
  Upsample (×2)           →  16×16×16
  Conv2D (16→12) + ReLU  →  16×16×12
  Upsample (×2)           →  32×32×12
  Conv2D (12→3)  + Sigmoid → 32×32×3
```

## Requirements

```bash
pip install torch torchvision matplotlib numpy
```

## Configuration

**Training Parameters** (in `train_cnn.py`):
```python
batch_size = 4
num_epochs = 10
learning_rate = 0.001
optimizer = Adam
loss_function = MSELoss
```

## Example Usage

```python
from autoencoderCNN import AutoencoderCNN
import torch

# Load trained model
model = AutoencoderCNN()
model.load_state_dict(torch.load('cifar_autoencoder.pth'))
model.eval()

# Reconstruct an image
with torch.no_grad():
    reconstructed = model(image_tensor)
```

## Project Structure

```
cifar10-cnn-autoencoder/
├── autoencoderCNN.py       # Model architecture
├── train_cnn.py            # Training script
├── predict_cnn.py          # Evaluation & visualization
├── .gitignore              # Git ignore (data, cache, model)
└── README.md               # This file
```

## What It Does

1. **Compression**: Reduces 32×32×3 images (3,072 values) to 8×8×16 (1,024 values)
2. **Learning**: Model learns to preserve important features during compression
3. **Reconstruction**: Decodes compressed representation back to original size
4. **Visualization**: Shows how well the model reconstructs CIFAR-10 images

## Results

The model outputs:
- Training loss per epoch
- Test set average loss
- Side-by-side comparison of original vs reconstructed images
- Predictions for all 10 CIFAR-10 classes (plane, car, bird, cat, deer, dog, frog, horse, ship, truck)

## CIFAR-10 Dataset

- **Training images**: 50,000
- **Test images**: 10,000
- **Image size**: 32×32 RGB
- **Classes**: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)

## Notes

- First run will download CIFAR-10 dataset (~170MB)
- Model file is ~100KB
- Training on CPU takes ~5-10 minutes per epoch
- GPU acceleration recommended for faster training

## License

Educational project.

---

*A simple CNN autoencoder for image compression and reconstruction*
