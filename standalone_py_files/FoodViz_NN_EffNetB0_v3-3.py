# FoodVision w/ EfficientNetB0 (internal layers frozen) - Version 3-3
# Not an actual standalone script, meant to be run in parts in a notebook (this is for display purposes;)

# -------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------- #
import torch
from torchvision import (
    models, datasets, transforms
    )
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import os

from PIL import Image
from sklearn.metrics import (
    classification_report
    , confusion_matrix
    , ConfusionMatrixDisplay
    )

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

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
# Use the preprocess transform for EfficientNet_B0 (preprocess_eff) and set up train/test dataset(s) and dataloader(s) from img directory;
# -------------------------------------------------------------------- #
train_dataset = datasets.ImageFolder(
    root=train_dir
    , transform=preprocess_eff
)

test_dataset = datasets.ImageFolder(
    root=test_dir
    , transform=preprocess_eff
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


# -------------------------------------------------------------------- #
#  EfficientNet_B0 model, class, device, criterion & optimizer instantiation;
# -------------------------------------------------------------------- #
# Number of classes in Pizza–Steak–Sushi
num_classes = len(train_dataset.classes)  # should be 3

# Use EfficientNet_B0 backbone
model = model_eff

# Freeze all layers except the final classifier head
for param in model.parameters():
    param.requires_grad = False

# Replace the final layer w/ the num_classes from the foodnet dataset --> EfficientNet uses model.classifier
model.classifier = nn.Linear(
      in_features=model.classifier[1].in_features
      , out_features=num_classes
)

# Move to device (device agnostic code should have already been run early in nb;)
model = model.to(device)

# Loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
      model.classifier.parameters()
      , lr=1e-3
)


# -------------------------------------------------------------------- #
# EfficientNet_B0 Training Loop *** With tqdm progress bar AND stored loss/acc for plotting in next cell;
# -------------------------------------------------------------------- #
# Lists to store metrics over epochs
train_losses = []
train_accuracies = []

# Since we already know it works well, just going with 20 epochs;)
epochs = 20

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

    # Save metrics (so we can plot a loss/acc curve!)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Acc: {train_acc:.4f}%")

# -------------------------------------------------------------------- #
# Plot Loss & Accuracy
# -------------------------------------------------------------------- #
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


# -------------------------------------------------------------------- #
# Running Test Loop on EfficientNet
# -------------------------------------------------------------------- #
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


# -------------------------------------------------------------------- #
# Set up a 'saved_models' directory (if it doesnt already exist):
# -------------------------------------------------------------------- #
from datetime import datetime

save_dir = "saved_models"
os.makedirs(save_dir, exist_ok=True)  # creates it if it doesn't exist

# Add the current date to the filename
current_date = datetime.now().strftime("%Y%m%d")
model_save_path = os.path.join(save_dir, f"efficientnet_b0_pizza_steak_sushi_{current_date}.pth")

# -------------------------------------------------------------------- #
# Save the trained model to .pth
# -------------------------------------------------------------------- #
# Save only the parameters
torch.save(model.state_dict(), model_save_path)
print(f"Model state_dict saved to: {model_save_path}")


# -------------------------------------------------------------------- #
# Test the trained model on a few 'unseen' images from the web;
# -------------------------------------------------------------------- #
# Folder containing unseen images (these are images of pizza/ steak/ sushi randomly grabbed fm the internet;)
folder_path = "./data/unseen_pza_stk_ssh"

# Make a list of image files
image_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

# Preprocessing used for EfficientNet_B0
weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
preprocess = weights.transforms()

# Class names (same as your dataset)
class_names = ["pizza", "steak", "sushi"]

# Ensure the model is in evaluation mode
model.eval()
model = model.to(device)


# -------------------------------------------------------------------- #
# Actual predix on unseen imgs:
# -------------------------------------------------------------------- #
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

# -------------------------------------------------------------------- #
# Extract the true labels from the filenames and generate classification report & confusion matrix;
# -------------------------------------------------------------------- #
# Extract true labels from filenames
true_labels = [f.split("_")[0] for f in image_files]

print("Classification Report:\n")
print(classification_report(true_labels, predicted_labels))

print("Confusion Matrix:\n")
print(confusion_matrix(true_labels, predicted_labels))

# -------------------------------------------------------------------- #
# ConfusionMatrixDisplay - plotted (note ~ this is '.from_predictions' method);
# -------------------------------------------------------------------- #
disp = ConfusionMatrixDisplay.from_predictions(
    true_labels, predicted_labels
    , normalize="true", cmap="Blues"
)

# replace default annotations with two-decimal strings
for txt in plt.gca().texts: txt.set_text(f"{float(txt.get_text()):.2f}")
plt.show();


# -------------------------------------------------------------------- #
# Visualizing predix imgs (with labels affixed to each img):
# -------------------------------------------------------------------- #
# Img / label viz function:
def show_image_prediction(img_path):
    # Load and preprocess
    image = Image.open(img_path).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        pred_class = class_names[probabilities.argmax()]

    # Display with matplotlib
    plt.figure(figsize=(5,5))
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"Predicted: {pred_class}", fontsize=14, color='green')
    plt.show();

# Visualize one unseen image with prediction (e.g., '[0]'th image);
show_image_prediction(os.path.join(folder_path, image_files[0]))

# -------------------------------------------------------------------- #
# Loop through all unseen images and display with predictions;
# -------------------------------------------------------------------- #
for img_name in image_files:
    img_path = os.path.join(folder_path, img_name)
    show_image_prediction(img_path)


# -------------------------------------------------------------------- #
#    <<<<<<<<<<<<<<<<<<<<<<<<<<< EXTRAS >>>>>>>>>>>>>>>>>>>>>>>>>>>    #
# -------------------------------------------------------------------- #

# -------------------------------------------------------------------- #
# Viewing shapes of random imgs from the transformed dataset:
# -------------------------------------------------------------------- #
# get one batch
images, labels = next(iter(train_loader))

print("images.shape:", images.shape)   # e.g. torch.Size([32, 3, H, W])
print("labels.shape:", labels.shape)   # e.g. torch.Size([32])

# -------------------------------------------------------------------- #
images, labels = next(iter(train_loader))
i = torch.randint(len(images), (1,)).item()   # random index in the batch
print(f"sample index: {i}")
print("single image shape:", images[i].shape)  # e.g. torch.Size([3, H, W])
print("single label:", labels[i].item())

# -------------------------------------------------------------------- #
import random
idx = random.randrange(len(train_dataset))
img, label = train_dataset[idx]
print("dataset sample shape:", img.shape)   # e.g. torch.Size([3, H, W])
print("label:", label)

# -------------------------------------------------------------------- #
# -------------------------------------------------------------------- #
# -------------------------------------------------------------------- #
