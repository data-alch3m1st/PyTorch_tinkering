# t0rch_h3lp3r_0x01.py #

# Lightweight Model-Agnostic PyTorch training/evaluation helpers for notebooks & scripts. 

import os
import random
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import hashlib

from sklearn.metrics import (confusion_matrix, classification_report, ConfusionMatrixDisplay)

# MacOS-friendly device agnostic code #
# --- Device selection (CUDA -> MPS -> CPU) --- #
if torch.cuda.is_available():
    device = "cuda"  # NVIDIA GPU
elif torch.backends.mps.is_available():
    device = "mps"   # Apple Silicon
else:
    device = "cpu"   # CPU fallback
print("Device:", device)

# Set current date string for plot titles
current_date = datetime.now().strftime("%Y-%m-%d")

# --------------------------------------------------------------------------- #

# Functionized train loop;

def train_model(model, train_loader, criterion, optimizer, scheduler, epochs=6, device=device):
    """
    Train a PyTorch model and track training metrics.

    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimization algorithm
        scheduler: Learning rate scheduler
        epochs: Number of training epochs (default 6 epochs for quick initial run; titrate on retrain.)
        device: Device will have been set earlier in process w/ Mac-friendly device-agnostic code;

    Returns:
        tuple: (train_losses, train_accuracies) - lists of training metrics
    """
    # Lists to store metrics over epochs
    train_losses = []
    train_accuracies = []

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

        # Save metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Print epoch metrics and current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Acc: {train_acc:.4f}% | LR: {current_lr:.6f}")

    return train_losses, train_accuracies

    """
    # Example Usage:
    
    train_losses, train_accuracies = train_model(
    model=model
    , train_loader=train_loader
    , criterion=criterion
    , optimizer=optimizer
    , scheduler=scheduler
    , epochs=6
    , device=device
)
    
    """
    
# --------------------------------------------------------------------------- #
# Test/evaluation loop;

import torch

def evaluate_model(model, test_loader, device=device):
    """
    Evaluate a PyTorch model on test data.

    Args:
        model: PyTorch model to evaluate
        test_loader: DataLoader for test data
        device: Device will have been set earlier as 'device' w/ Mac-friendly device-agnostic code;

    Returns:
        float: Test accuracy (0-100)
    """
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

    return test_acc

    """
    # Example Usage:
    
    test_accuracy = evaluate_model(
    model=model
    , test_loader=test_loader
    , device=device
)
    """
# --------------------------------------------------------------------------- #
# Test/evaluation loop that also returns predictions and labels for confusion matrix / classification report:

import torch

def evaluate_model_with_cm_params(model, test_loader, device):
    """
    Evaluate a PyTorch model on test data and return accuracy plus
    the predictions and true labels needed for a confusion matrix / classification report.

    Args:
        model:       PyTorch model to evaluate (already moved to device)
        test_loader: DataLoader for test data
        device:      torch.device set earlier in your notebook

    Returns:
        test_acc (float): Test accuracy in percent (0-100)
        all_preds (np.ndarray): Predicted class indices for the entire test set
        all_labels (np.ndarray): True class indices for the entire test set
    """
    model.eval()
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # accumulate accuracy numerators/denominators
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)

            # store for CM / report later
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

    test_acc = (correct / total * 100) if total > 0 else 0.0
    print(f"Test Accuracy: {test_acc:.4f}%")

    return test_acc, all_preds, all_labels    
    
"""
# Example Usage:

test_acc, all_preds, all_labels = evaluate_model_with_cm_params(
    model=model
    , test_loader=test_loader
    , device=device
)
"""

# --------------------------------------------------------------------------- #
# Build a catalog of test images (recursive) with true labels from folder names

import os
import random

# Allowed extensions
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def list_test_images(
    root_dir
    , class_names
):
    """
    Walks the test directory (split by class subfolders) and returns:
      - all_items: list of (img_path, true_label_str)
      - by_class: dict true_label_str -> list[img_path]
    """
    all_items = []
    by_class = {c: [] for c in class_names}

    # Walk each class subfolder
    for cls in class_names:
        cls_dir = os.path.join(root_dir, cls)
        if not os.path.isdir(cls_dir):
            # skip silently if class not present
            continue
        for fn in os.listdir(cls_dir):
            ext = os.path.splitext(fn)[1].lower()
            if ext in IMG_EXTS:
                p = os.path.join(cls_dir, fn)
                all_items.append((p, cls))
                by_class[cls].append(p)

    return all_items, by_class

"""
# Example Usage:
test_items, test_by_class = list_test_images(
    test_dir
    , class_names
)

print(f"Found {len(test_items)} test images across {len(class_names)} classes.")
for cls in class_names:
    print(f"  {cls:20s}: {len(test_by_class.get(cls, []))}")

"""    
    

# --------------------------------------------------------------------------- #
# show_image_prediction(), extended to optionally accept true_label
import torch
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image

def show_image_prediction(
    img_path
    , true_label=None
    , model=None
    , preprocess=None
    , class_names=None
    , device=device
):
    """
    Displays an image, model prediction, and full class probability table.
    Optionally prints the provided true_label for quick correctness check.
    """
    if model is None or preprocess is None or class_names is None:
        raise ValueError("model, preprocess, and class_names must be provided.")

    image = Image.open(img_path).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        pred_idx = int(probabilities.argmax().item())
        pred_class = class_names[pred_idx]
        pred_prob = float(probabilities[pred_idx].item() * 100.0)

    title_str = f"Predicted: {pred_class} ({pred_prob:.2f}%)"
    if true_label is not None:
        correctness = "CORRECT!" if pred_class == true_label else "FALSE!"
        title_str = f"{title_str} | True: {true_label} {correctness}"

    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    plt.axis("off")
    plt.title(title_str, fontsize=14, color='green')
    plt.show()

    print("Class probabilities:")
    for i, cls in enumerate(class_names):
        prob = float(probabilities[i].item() * 100.0)
        mark = "True " if (true_label is not None and cls == true_label) else "     "
        star = "*" if i == pred_idx else " "
        print(f"{star} {mark}{cls:20s}: {prob:7.4f}%")



        """
# Example Usage 1:
        
# Show ONE truly random test image (from any class)

assert len(test_items) > 0, "No test images found."
rand_path, rand_true = random.choice(test_items)

show_image_prediction(
    img_path=rand_path
    , model=model
    , preprocess=preprocess
    , class_names=class_names
    , device=device
    , true_label=rand_true
)


# Example Usage 2:

# Show ONE random image PER CLASS (balanced peek)

for cls in class_names:
    candidates = test_by_class.get(cls, [])
    if not candidates:
        print(f"[skip] No images found for class: {cls}")
        continue
    img_path = random.choice(candidates)
    show_image_prediction(
        img_path=img_path   
        , model=model    
        , preprocess=preprocess    
        , class_names=class_names   
        , device=device    
        , true_label=cls
    )
    
        """

# --------------------------------------------------------------------------- #
# Functionized Plot training curves with Seaborn

import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def plot_training_curves_seaborn(
    df, x_col="Epoch"
    , loss_col="Train Loss", acc_col="Train Accuracy"
    , title="Training Loss & Accuracy over Epochs"
    , add_date=True
):
    """
    Plots training loss and accuracy curves using Seaborn and matplotlib.

    Parameters:
    - df: DataFrame containing the metrics.
    - x_col: Column name for the x-axis (default: "Epoch").
    - loss_col: Column name for the loss values (default: "Train Loss").
    - acc_col: Column name for the accuracy values (default: "Train Accuracy").
    - title: Title of the plot (default: "Training Loss & Accuracy over Epochs").
    """
    
    # Set current date from within script;
    if add_date:
        current_date = datetime.now().strftime("%Y-%m-%d")
        title = f"{title} ~ {current_date}"
    
    sns.set_style('darkgrid')
    fig, ax1 = plt.subplots(figsize=(10, 6))   

# Plot loss on the left y-axis using Seaborn
    sns.lineplot(
        data=df
        , x=x_col
        , y=loss_col
        , ax=ax1
        , color='tab:red'
        , marker='o'
        , label=loss_col
        , linewidth=2
        , markersize=6
    )
    ax1.set_xlabel(x_col)
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')

# Create a second y-axis for accuracy
    ax2 = ax1.twinx()
    sns.lineplot(
        data=df
        , x=x_col
        , y=acc_col
        , ax=ax2
        , color='tab:blue'
        , marker='o'
        , label=acc_col
        , linewidth=2
        , markersize=6
    )
    ax2.set_ylabel('Accuracy (%)', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

# Add title and legend
    plt.title(
        title
        , fontsize=18
        , pad=20
    )
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2
        , labels1 + labels2
        , loc='upper center'
        , bbox_to_anchor=(0.5, -0.15)
        , ncol=2
    )

    plt.tight_layout()
    plt.show();

    """
# Example Usage:

plot_training_curves_seaborn(
    df_metrics
    , x_col="Epoch"
    , loss_col="Train Loss"
    , acc_col="Train Accuracy"
    , title=f"Training Loss & Accuracy over Epochs (EfficientNetB2) ~ {current_date}"
)
    
    """

# --------------------------------------------------------------------------- #
# Functionized Plot training curves with Plotly Express

import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

def plot_training_curves(
    df
    , x_col="Epoch"
    , loss_col="Train Loss"
    , acc_col="Train Accuracy"
    , title="Training Loss & Accuracy over Epochs"
    , add_date=True
):
    """
    Plots training loss and accuracy curves using Plotly Express.

    Parameters:
    - df: DataFrame containing the metrics.
    - x_col: Column name for the x-axis (default: "Epoch").
    - loss_col: Column name for the loss values (default: "Train Loss").
    - acc_col: Column name for the accuracy values (default: "Train Accuracy").
    - title: Title of the plot (default: "Training Loss & Accuracy over Epochs").
    """
    
    # Set current date from within script;
    if add_date:
        current_date = datetime.now().strftime("%Y-%m-%d")
        title = f"{title} ~ {current_date}" 
    
    fig = px.line(
        df
        , x=x_col
        , y=loss_col
        , height=600
        , width=900
        , title=title
        , markers=True
    )

    # Add Accuracy trace (assign to y2 axis)
    fig.add_scatter(
        x=df[x_col]
        , y=df[acc_col]
        , mode="lines+markers"
        , name=acc_col
        , yaxis="y2"
        , line=dict(width=3)
    )

    # Configure dual axes
    fig.update_layout(
        template="seaborn"
        , title_x=0.5
        , title_font_size=18
        , font=dict(size=14)
        , legend=dict(
            title=None
            , orientation="h"
            , yanchor="bottom"
            , y=1.02
            , xanchor="center"
            , x=0.5
        ),
        xaxis=dict(title=x_col)
        , yaxis=dict(
            title="Loss"
            , titlefont=dict(color="red")
            , tickfont=dict(color="red")
        ),
        yaxis2=dict(
            title="Accuracy (%)"
            , titlefont=dict(color="blue")
            , tickfont=dict(color="blue")
            , overlaying="y"
            , side="right"
        ), 
        hovermode="x unified"
    )

    fig.update_traces(line=dict(width=3), marker=dict(size=8))
    fig.show()

"""
# Example usage:

plot_training_curves(
    df_metrics
    , x_col="Epoch"
    , loss_col="Train Loss"
    , acc_col="Train Accuracy"
    , title=f"Training Loss & Accuracy over Epochs (EfficientNetB2)"
)
"""

# --------------------------------------------------------------------------- #
# Function to check for duplicate images across directories:
import hashlib
import os
import PIL.Image as Image

# Dependency function - need to hash the images:
def compute_image_hash(image_path):
    """
    Compute a unique MD5 hash for an image file.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: MD5 hash of the image as a hexadecimal string.
    """
    with Image.open(image_path) as img:
        # Convert image to bytes
        img_bytes = img.tobytes()
        # Compute hash
        return hashlib.md5(img_bytes).hexdigest()

# Actual dupe finder (with above hashing function built-in):
def find_duplicate_images(unseen_dir, train_dir, test_dir, verbose=True):
    """
    Check if images in the 'unseen' directory are already present in the train or test directories.
    This is useful for ensuring that images used for testing are truly unseen by the model.

    Args:
        unseen_dir (str): Path to the directory containing images to check for duplicates.
        train_dir (str): Path to the directory containing training images.
        test_dir (str): Path to the directory containing test images.
        verbose (bool): If True, prints the results directly. If False, returns the list of duplicates silently.

    Returns:
        list: A list of tuples, where each tuple contains the path to a duplicate image in the unseen directory
              and the path to the matching image in the train or test directory.
              If no duplicates are found, the list will be empty.
    """
    # Get all image paths
    unseen_images = [os.path.join(unseen_dir, f) for f in os.listdir(unseen_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    train_images = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    test_images = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    # Compute hashes for train and test images
    train_hashes = {compute_image_hash(img): img for img in train_images}
    test_hashes = {compute_image_hash(img): img for img in test_images}

    # Check unseen images
    duplicates = []
    for img in unseen_images:
        img_hash = compute_image_hash(img)
        if img_hash in train_hashes or img_hash in test_hashes:
            duplicates.append((img, train_hashes.get(img_hash, test_hashes.get(img_hash))))

    # Print results if verbose is True
    if verbose:
        if duplicates:
            print("Duplicate images found:")
            for unseen, existing in duplicates:
                print(f"Unseen image: {unseen} is a duplicate of: {existing}")
        else:
            print("No duplicate images found. All images are truly unseen.")

    return duplicates

"""
# Example Usage:

unseen_dir = '../data/unseen_imgs/'
train_dir = '../data/train/'
test_dir = '../data/test/'

# Call the function with verbose=True (default)
duplicates = find_duplicate_images(unseen_dir, train_dir, test_dir)

    
"""

# ------------------------------- #

