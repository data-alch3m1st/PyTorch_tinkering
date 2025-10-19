# FoodVision w/ EfficientNetB0 (internal layers frozen) - Version 3-3
# Not an actual standalone script, meant to be run in parts in a notebook (this is for display purposes;)

# -------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------- #
import torch
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import os

from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from datetime import datetime
from matplotlib import pyplot as plt

# -------------------------------------------------------------------- #
# Device Agnostic Code (MacOS friendly)
# -------------------------------------------------------------------- #
if torch.cuda.is_available():
    device = "cuda" # Use NVIDIA GPU (if available)
elif torch.backends.mps.is_available():
    device = "mps" # Use Apple Silicon GPU (if available)
else:
    device = "cpu" # Default to CPU if no GPU is available
    
print(device)

# -------------------------------------------------------------------- #
# Google Colab Mounting (if running in Google Colab)
# -------------------------------------------------------------------- #
# # from google.colab import drive
# drive.mount('/content/drive')

# -------------------------------------------------------------------- #
# Load the data from the drive (local): *(assuming these are already downloaded and unzipped)*
# -------------------------------------------------------------------- #
train_dir = "./data/pizza_steak_sushi/train/"
test_dir = "./data/pizza_steak_sushi/test/"

# -------------------------------------------------------------------- #
# Load EfficientNet with B0 pretrained weights
# -------------------------------------------------------------------- #
weights_eff = models.EfficientNet_B0_Weights.IMAGENET1K_V1
model_eff = models.efficientnet_b0(weights=weights_eff)
preprocess_eff = weights_eff.transforms()


# -------------------------------------------------------------------- #

# -------------------------------------------------------------------- #



# -------------------------------------------------------------------- #

# -------------------------------------------------------------------- #



# -------------------------------------------------------------------- #

# -------------------------------------------------------------------- #



# -------------------------------------------------------------------- #

# -------------------------------------------------------------------- #



# -------------------------------------------------------------------- #

# -------------------------------------------------------------------- #



# -------------------------------------------------------------------- #

# -------------------------------------------------------------------- #