# --------------------------------------------------------------------------- #
# Imports

import torch
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torchinfo import summary

import os
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from timeit import default_timer as timer
from tqdm import tqdm

import requests
import zipfile
from pathlib import Path
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings("ignore")


if torch.cuda.is_available():
    device = "cuda" # Use NVIDIA GPU (if available)
elif torch.backends.mps.is_available():
    device = "mps" # Use Apple Silicon GPU (if available)
else:
    device = "cpu" # Default to CPU if no GPU is available

print(device)


# --------------------------------------------------------------------------- #
# Load the data from the drive:

train_dir = "./data/pizza_steak_sushi/train/"
test_dir = "./data/pizza_steak_sushi/test/"


# --------------------------------------------------------------------------- #
# Load EfficientNet with B2 pretrained weights

weights = models.EfficientNet_B2_Weights.IMAGENET1K_V1
model = models.efficientnet_b2(weights=weights)
preprocess = weights.transforms()

# --------------------------------------------------------------------------- #
# Use the preprocess transform for EfficientNet_B2 (preprocess)

train_dataset = datasets.ImageFolder(
    root=train_dir
    , transform=preprocess
)

test_dataset = datasets.ImageFolder(
    root=test_dir
    , transform=preprocess
)

train_loader = DataLoader(
    dataset=train_dataset
    , batch_size=32
    , shuffle=True
    , num_workers=2
)

test_loader = DataLoader(
    dataset=test_dataset
    , batch_size=32
    , shuffle=False
    , num_workers=2
)

test_dataset.classes

# --------------------------------------------------------------------------- #

# EfficientNet_B2 model, class, device, criterion & optimizer instantiation

model = model

# Number of classes (Pizza–Steak–Sushi)
num_classes = len(train_dataset.classes)  # should be 3

# Freeze all layers except classifier head
for param in model.parameters():
    param.requires_grad = False

# Replace the final classifier
# EfficientNet_B2 has model.classifier = Sequential(Dropout, Linear)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, num_classes)

# Move model to device
model = model.to(device)

# Loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)

# Learning rate scheduler
scheduler = StepLR(optimizer, step_size=5, gamma=0.8)


# --------------------------------------------------------------------------- #
# EfficientNet_B2 Training Loop *** With tqdm progress bar AND stored loss/acc for plotting in next cell;

# Lists to store metrics over epochs
train_losses = []
train_accuracies = []

# 27 epochs for the heck of it'
epochs = 27

for epoch in tqdm(range(epochs)):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels).item()
        total += labels.size(0)

    # Compute metrics for this epoch
    train_loss = running_loss / total
    train_acc = correct / total * 100
    
    # lr scheduler
    scheduler.step()

    # Save metrics (so we can plot a loss/acc curve!)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    
   # Print epoch metrics and current learning rate
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Acc: {train_acc:.4f}% | LR: {current_lr:.6f}")


# --------------------------------------------------------------------------- #
# Plot Loss & Accuracy

fig, ax1 = plt.subplots(figsize=(8, 5))

# Plot loss on left y-axis
ax1.plot(range(1, epochs+1), train_losses, color='red', marker='o', label='Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color='red')
ax1.tick_params(axis='y', labelcolor='red')

# Create a second y-axis for accuracy
ax2 = ax1.twinx()
ax2.plot(range(1, epochs+1), train_accuracies, color='blue', marker='o', label='Accuracy')
ax2.set_ylabel('Accuracy (%)', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

# Title and layout
plt.title('Training Loss & Accuracy Curves')
fig.tight_layout()
plt.show();


# --------------------------------------------------------------------------- #
# Running Test Loop on EfficientNet

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels).item()
        total += labels.size(0)

test_acc = correct / total * 100
print(f"Test Accuracy: {test_acc:.4f}%")


# --------------------------------------------------------------------------- #
# Set up a 'saved_models' directory (if it doesnt already exist):
from datetime import datetime

# Derive model name automatically 
weights_str = str(weights) # 'EfficientNet_B2_Weights.IMAGENET1K_V1' 
# ^^^Need to update this in other scripts as I wont always be using 'eff'...

named_model = weights_str.split('_Weights')[0]  # 'EfficientNet_B2'

# Save directory setup
save_dir = "./saved_models/"
os.makedirs(save_dir, exist_ok=True)

# Add the current date to the filename
current_date = datetime.now().strftime("%Y%m%d")

# --- Build filename dynamically ---
model_save_path = os.path.join(
    save_dir
    , f"{named_model.lower()}_pizza_steak_sushi_{current_date}.pth"
)


# --------------------------------------------------------------------------- #
# Save the trained model to .pth

# Save only the parameters
torch.save(model.state_dict(), model_save_path)
print(f"Model state_dict saved to: {model_save_path}")


# --------------------------------------------------------------------------- #
import os
from PIL import Image
import torch
from torchvision import models

# Folder containing unseen images (these are images of pizza/ steak/ sushi randomly grabbed fm the internet;)
folder_path = "./data/unseen_pza_stk_ssh"

# Make a list of image files
image_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
image_files.sort()

# Class names (same as your dataset)
class_names = ["pizza", "steak", "sushi"]

# Ensure the model is in evaluation mode
model.eval()
model = model.to(device)

# Actual predix on unseen imgs:

predicted_labels = []
for img_name in image_files:
    img_path = os.path.join(folder_path, img_name)
    image = Image.open(img_path).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_class = class_names[probabilities.argmax()]
        predicted_labels.append(predicted_class)

    print(f"{img_name} → Predicted: {predicted_class}")



# --------------------------------------------------------------------------- #
# Extract true labels from filenames
true_labels = [f.split("_")[0] for f in image_files]
print(true_labels)



# --------------------------------------------------------------------------- #
from sklearn.metrics import classification_report, confusion_matrix

print("Classification Report:\n")
print(classification_report(true_labels, predicted_labels))

print("Confusion Matrix:\n")
print(confusion_matrix(true_labels, predicted_labels))


# --------------------------------------------------------------------------- #
# ConfusionMatrixDisplay

disp = ConfusionMatrixDisplay.from_predictions(
    true_labels, predicted_labels
    , normalize="true", cmap="Blues"
)

# replace default annotations with two-decimal strings
for txt in plt.gca().texts: txt.set_text(f"{float(txt.get_text()):.2f}")
plt.show();



# --------------------------------------------------------------------------- #
import os
from PIL import Image
import torch
from torchvision import models
from matplotlib import pyplot as plt

# Updated predix function w/ probabilities displayed;

def show_image_prediction(img_path):
    # Load and preprocess
    image = Image.open(img_path).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        pred_idx = probabilities.argmax()
        pred_class = class_names[pred_idx]
        pred_prob = probabilities[pred_idx].item() * 100

    # Display image with prediction
    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"Predicted: {pred_class} ({pred_prob:.2f}%)", fontsize=14, color='green')
    plt.show()

    # Print probability distribution below the image
    print("Class probabilities:")
    for i, cls in enumerate(class_names):
        prob = probabilities[i].item() * 100
        mark = "True" if i == pred_idx else "False"
        print(f"{mark} {cls:10s}: {prob:.4f}%")


# --------------------------------------------------------------------------- #
show_image_prediction(os.path.join(folder_path, image_files[0]))


# --------------------------------------------------------------------------- #
# Display a random img from the unseen data to show predix & probs:
rand_idx = random.randrange(len(image_files))
rand_image_path = os.path.join(folder_path, image_files[rand_idx])

show_image_prediction(rand_image_path)

# --------------------------------------------------------------------------- #
# Display a ALL imgs from the unseen data to show predix & probs:
# Warning: this can take up a sizeable area in the notebook - even with only a few imgs!

for img_name in image_files:
    img_path = os.path.join(folder_path, img_name)
    show_image_prediction(img_path)

# --------------------------------------------------------------------------- #

# for sneaking peeks at the batches, single imgs, etc.
# get one batch
images, labels = next(iter(train_loader))

print("images.shape:", images.shape)   # e.g. torch.Size([32, 3, H, W])
print("labels.shape:", labels.shape)   # e.g. torch.Size([32])


# --------------------------------------------------------------------------- #
