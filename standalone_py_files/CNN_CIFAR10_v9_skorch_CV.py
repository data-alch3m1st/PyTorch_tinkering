### CIFAR-10 CNN (v009) ###

# BLUF: The intent of this standalone .py file is for modularity/ portability/ process rather than to be run as a standalone; (this code is primarily for execution via Jupyter Notebook.) Essentially, until I become more proficient with PyTorch, I want to be able to use the processes developed here using skorch & PyTorch to build other models and its just easier to do that (IMHO) from a VS Code .py file.


## If running on Colab, will need to pip install skorch & torchmetrics each time;

# !pip install skorch
# !pip install torchmetrics


# ---------------------------------------------------------
# 0. Imports, Device-Agnostic Code
# ---------------------------------------------------------

# Imports (I tend to do most up front and add on as needed;)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from skorch import NeuralNetClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix
    , accuracy_score, ConfusionMatrixDisplay
    )
from torchmetrics import Accuracy, ConfusionMatrix
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
from timeit import default_timer as timer
from tqdm import tqdm

warnings.filterwarnings("ignore")

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# MacoOS device agnostic code:

if torch.cuda.is_available():
    device = "cuda"  # Use NVIDIA GPU (if available)
elif torch.backends.mps.is_available():
    device = "mps"  # Use Apple Silicon GPU (if available)
else:
    device = "cpu"  # Default to CPU if no GPU is available

device

# ---------------------------------------------------------
# 1. Prepare dataset (CIFAR-10) with Augmentations
# ---------------------------------------------------------

# Train transforms (with augmentations)
train_transform_batch = transforms.Compose([
    transforms.RandomHorizontalFlip()  # Flip images with 50% probability
    , transforms.RandomCrop(32, padding=4)  # Add padding and crop randomly
    , transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR-10 stats
])

# Test transforms (only normalize)
test_transform_batch = transforms.Compose([
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465)
        , (0.2023, 0.1994, 0.2010))  # CIFAR-10 stats
])

# CIFAR-10 dataset download
dataset = datasets.CIFAR10(
    './data', train=True, download=True
)

# Extract raw data and labels
X = torch.tensor(dataset.data).permute(0, 3, 1, 2).float() / 255.0  # Convert to [N, C, H, W]
y = torch.tensor(dataset.targets)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=27, stratify=y
)

X_train.shape, X_test.shape, y_train.shape, y_test .shape

# ---------------------------------------------------------
# 2b. Wrap in DataLoader for batching
# ---------------------------------------------------------

train_loader = DataLoader(
    list(zip(X_train, y_train))
    , batch_size=128
    , shuffle=True
)

test_loader = DataLoader(
    list(zip(X_test, y_test))
    , batch_size=128
    , shuffle=False
)

# ---------------------------------------------------------
# 3. Build the model class; (layers, blocks, forward, flatten, dropout, etc.)
# ---------------------------------------------------------

class Cifar10CNN_v9(nn.Module):
    def __init__(self, channels=64, dropout=0.3):
        super().__init__()
        # Convolutional block 1
        self.conv1 = nn.Conv2d(3, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)

        # Convolutional block 2
        self.conv2 = nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels * 2)

        # Convolutional block 3
        self.conv3 = nn.Conv2d(channels * 2, channels * 4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(channels * 4)

        # Convolutional block 4
        self.conv4 = nn.Conv2d(channels * 4, channels * 8, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(channels * 8)

        # Global average pooling to reduce overfitting
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(channels * 8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc_out = nn.Linear(256, 10)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)

        # Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)

        # Block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)

        # Block 4
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)

        # Global average pooling
        x = self.global_avgpool(x)
        x = torch.flatten(x, 1)

        # Fully connected layers
        x = F.relu(self.fc1(self.dropout(x)))
        x = F.relu(self.fc2(self.dropout(x)))
        return self.fc_out(x)

%%time
# ---------------------------------------------------------
# 4. Wrap (base model) with skorch
# ---------------------------------------------------------

net = NeuralNetClassifier(
    module=Cifar10CNN_v9
    , criterion=nn.CrossEntropyLoss
    , max_epochs=10
    , lr=0.0001
    , optimizer=torch.optim.RMSprop
    , batch_size=128
    , iterator_train__shuffle=True
    , device=device
)

net.fit(X_train, y_train)

print("Train accuracy:", net.score(X_train, y_train))

# ---------------------------------------------------------
# 5a. RandomizedSearchCV for Hyperparams Tuning
# ---------------------------------------------------------

# (param_dist_v6) # Removed torch.optim.Adam, and 64 batch_size to increase coverage in grid_search (as these are not likely ideal for this model and are just wasting space/time in a grid_search;)

param_dist = {
    "module__channels": [64, 128]  # Increasing upper range
    , "module__dropout": [0.2, 0.3]  # Slightly reduced dropout values
    , "lr": [1e-4, 3e-4, 1e-3]  # Removed extreme values for better control
    , "optimizer": [torch.optim.Adam, torch.optim.SGD, torch.optim.RMSprop]   # torch.optim.Adam removed for CIFAR but added back for consistency;
    , "batch_size": [128]  # Added smaller batch size for exploration # 64,
    , "optimizer__momentum": [0.0, 0.9]
    , "optimizer__weight_decay": [0.0, 1e-4, 5e-4]
}


%%time
# ---------------------------------------------------------
# 5b. Randomized Search Actual (sample combos, actual run;)
# ---------------------------------------------------------

rs = RandomizedSearchCV(
    estimator=net
    , param_distributions=param_dist
    , n_iter=20
    , cv=2
    , scoring="accuracy"
    , verbose=1
    , random_state=27
    , n_jobs=-1
)

rs.fit(X_train, y_train)

print("Best CV accuracy:", rs.best_score_)
print("Best params:", rs.best_params_)

rs.best_params_


# ---------------------------------------------------------
# 6. Send results to a dataframe for clean review;
# ---------------------------------------------------------

results_v9 = pd.DataFrame(rs.cv_results_)
results_v9[[
    "mean_test_score"
    , "std_test_score"
    , "params"
]].sort_values("mean_test_score", ascending=False, inplace=True)
results_v9.head()

results_v9.sort_values(by='rank_test_score', ascending=True).head(5)

# Printing out 'Top 3' Params (if after a full 100-epoch PyTorch native train loop and/or subsequent retrain the results are still not ideal, can pivot to the #2 and/or #3 params;)
print(results_v9.sort_values(by='rank_test_score', ascending=True)['params'].iloc[0])
print(results_v9.sort_values(by='rank_test_score', ascending=True)['params'].iloc[1])
print(results_v9.sort_values(by='rank_test_score', ascending=True)['params'].iloc[2])

# from google.colab import drive
# drive.mount('/content/drive')   # follow the auth steps
results_v9.to_csv('/content/drive/MyDrive/results_v9.csv', index=False)

# 2) Inspect the current Colab VM working directory
import os
print("Colab VM cwd:", os.getcwd())
print("Files in /content:", os.listdir('/content'))

# 3) Define the Drive path you want to use
# Note: "MyDrive" is the mount point for "My Drive"
drive_folder = '/content/drive/MyDrive/Colab Notebooks/PyTorch_tinkering'

# 4) Create the target folder if it doesn't exist
os.makedirs(drive_folder, exist_ok=True)
print("Target folder ready at:", drive_folder)

# 5) Save the DataFrame to that folder
# replace `results` with your actual DataFrame variable name
csv_path = os.path.join(drive_folder, 'results_v9.csv')
results_v9.to_csv(csv_path, index=False)
print("Saved CSV to:", csv_path)

results_v9.to_csv("results_v9.csv", index=False)

from google.colab import files
files.download("results_v9.csv")

rs.best_estimator_
rs.best_params_
rs.get_params
rs.cv_results_

# rs.cv_results_

rs.best_index_

# ---------------------------------------------------------
# 7a. Predict & Evaluate Performance with Best Model (Metrics)
# ---------------------------------------------------------

best_model = rs.best_estimator_
y_pred = best_model.predict(X_test)

print("Accuracy:\n", accuracy_score(y_test, y_pred))
print('\n=================================================================\n')
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('\n=================================================================\n')
print("Classification Report:\n", classification_report(y_test, y_pred))

cm = ConfusionMatrix(num_classes=10, task='multiclass')
cm_tensor = cm(torch.tensor(y_pred), y_test)

# 7b. Plot the Confusion Matrix (Visualized Metrics)

fig, ax = plt.subplots(figsize=(12, 10))

# disp = ConfusionMatrixDisplay.from_predictions(
#     y_test, y_pred
#     , display_labels=dataset.classes
#     , cmap='Blues'
#     , normalize='true', xticks_rotation=45
# )

disp.plot(ax=ax, cmap='Blues')
plt.show();

# 8. Functionize the train & test loop(s):

def train_step(model, loader, loss_fn, optimizer, accuracy_fn, device):
    model.train()
    total_loss, total_acc = 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        # Apply train transforms to the batch
        X = train_transform_batch(X)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_acc += accuracy_fn(y_pred.argmax(dim=1), y).item()
    return total_loss / len(loader), total_acc / len(loader)

def test_step(model, loader, loss_fn, accuracy_fn, device):
    model.eval()
    total_loss, total_acc = 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            # Apply test transforms to the batch
            X = test_transform_batch(X)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            total_loss += loss.item()
            total_acc += accuracy_fn(y_pred.argmax(dim=1), y).item()
    return total_loss / len(loader), total_acc / len(loader) # Return test loss

# ---------------------------------------------------------
# 9. Set the final model with the best params (including channels & dropout)
# ---------------------------------------------------------

model = Cifar10CNN_v9(
    channels=rs.best_params_["module__channels"]
    , dropout=rs.best_params_["module__dropout"]
).to(device)

model

"""
# Old model.train params (keeping handy for a rainy day;;;):
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

"""

%%time

# 10. Train the Best Model

epochs = 100  # For CIFAR-10, can start w/ 50 and do add'l 50 retrain if needed OR just go full 100;

optimizer = rs.best_params_["optimizer"](model.parameters(), lr=rs.best_params_["lr"])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=7, min_lr=0.00001)
loss_fn = nn.CrossEntropyLoss()
accuracy_fn = Accuracy(task='multiclass', num_classes=10).to(device)


for epoch in tqdm(range(epochs)):
    train_loss, train_acc = train_step(
        model, train_loader, loss_fn, optimizer, accuracy_fn, device)
    test_loss, test_acc = test_step(
        model, test_loader, loss_fn, accuracy_fn, device)
    scheduler.step(test_loss)  # Update learning rate with test_loss
    print(
        f"Epoch {epoch+1}/{epochs} - "
        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f} - "
        f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}"
    )

# print best train & test acc's;
# train_acc, test_acc

# print best train & test acc's;
train_acc, test_acc

# ---------------------------------------------------------
# 11: [UPDATED] Eval Mode for Final Predix, Metrics on Best & Fully Trained Model
# ---------------------------------------------------------

# 11a. Final evaluation on test set with sklearn metrics
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        X = test_transform_batch(X)   # normalize just like during training
        y_pred = model(X)
        all_preds.extend(y_pred.argmax(dim=1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())

# 11b. Compute metrics

print("\n Final Evaluation with Sklearn Metrics")
print("Accuracy:", accuracy_score(all_labels, all_preds))
print("\nConfusion Matrix:\n", confusion_matrix(all_labels, all_preds))
print("\nClassification Report:\n", classification_report(all_labels, all_preds))

# ---------------------------------------------------------
# 12. Final Confusion Matrix for Best & Fully Trained Model
# ---------------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 10))

# disp = ConfusionMatrixDisplay.from_predictions(
#     all_labels, all_preds
#     , display_labels=dataset.classes
#     , normalize='true'
# )

disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix for CIFAR-10 - Final Model")
plt.show();

model

# Save both model weights and optimizer state
torch.save({
    'epoch': epochs
    , 'model_state_dict': model.state_dict()
    , 'optimizer_state_dict': optimizer.state_dict()
    , 'loss': loss_fn  # optional
}, "Cifar10CNN_v9_best_100_epochs.pth")

print("Model saved to Cifar10CNN_v9_best_100_epochs.pth")

# ---------------------------------------------------------
# 13. Retrain (If needed)
# ---------------------------------------------------------

more_epochs = 100

for epoch in tqdm(range(more_epochs)):
    train_loss, train_acc = train_step(
        model, train_loader, loss_fn, optimizer, accuracy_fn, device)
    test_loss, test_acc = test_step(
        model, test_loader, loss_fn, accuracy_fn, device)
    scheduler.step(test_loss)  # Update learning rate
    print(
        f"Epoch {epoch+1}/{epochs} - "
        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
        f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}"
    )

# train_acc, test_acc

train_acc, test_acc

# ---------------------------------------------------------
# 14a. Final evaluation on test set with sklearn metrics
# ---------------------------------------------------------

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        X = test_transform_batch(X)   # normalize just like during training
        y_pred = model(X)
        all_preds.extend(y_pred.argmax(dim=1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())

# Convert lists to numpy arrays
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# ---------------------------------------------------------
# 14b. Compute metrics
# ---------------------------------------------------------

print("\nFinal Evaluation with Sklearn Metrics")
print("Accuracy:", accuracy_score(all_labels, all_preds))
# print("\nConfusion Matrix:\n", confusion_matrix(all_labels, all_preds))
# print("\nClassification Report:\n", classification_report(all_labels, all_preds))

# Convert back to tensors for torchmetrics
all_preds_tensor = torch.tensor(all_preds)
all_labels_tensor = torch.tensor(all_labels)

print("\nFinal Evaluation with torchmetrics Metrics\n", cm_tensor)
cm = ConfusionMatrix(num_classes=10, task='multiclass')
cm_tensor = cm(all_preds_tensor, all_labels_tensor)
# cm_tensor
print("\nClassification Report:\n", classification_report(all_labels, all_preds))

# ---------------------------------------------------------
# 15. Final Confusion Matrix for Best & Fully Trained Model
# ---------------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 10))

# NOTE: After running 'disp' (ConfusionMatrixDisplay), COMMENT IT OUT AND RE-RUN!
# ^^^ Otherwise, plt will give you two very bad-looking conjoined CM plots!

disp = ConfusionMatrixDisplay.from_predictions(
    all_labels, all_preds
    , display_labels=dataset.classes
    , normalize='true'
)

disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix for CIFAR-10 - Final Model")
plt.show();

# ---------------------------------------------------------
# 16a. Save both model weights and optimizer state (in case you want to retrain, run add'l epochs, etc.)
# ---------------------------------------------------------

torch.save({
    'epoch': epochs + more_epochs
    , 'model_state_dict': model.state_dict()
    , 'optimizer_state_dict': optimizer.state_dict()
    , 'loss': loss_fn,  # optional
}, "Cifar10CNN_v9_more_retrained_200_epochs_best.pth")

print("Model saved to Cifar10CNN_v9_more_retrained_200_epochs__best.pth")

# ---------------------------------------------------------
# 16b. Save best weights  (for eval mode only, so can only eval new data w/ weights; no add'l training;)
# ---------------------------------------------------------

torch.save(model.state_dict(), "Cifar10CNN_v9_more_retrained_best_weights.pth")

# Cifar10CNN_v9_best

torch.save(model.state_dict(), "Cifar10CNN_v9_best_weights.pth")

print("\nClassification Report:\n", classification_report(all_labels, all_preds))


# NOTES:

"""
***UPDATE:*** 

*I re-ran a skorch RandomSearchCV but with some adjustments to get better efficiency out of the grid search in order to exposed a greater number of hyper-params combinations. As such, I upped the n_iter=20 (prev. n_iter=10) and lowered the cv=2 (prev. cv=3). which increased total iterations to 40, but getting 20 hyperparam combos instead of 10.*

*Regardless, while model performed quite well, even hitting higher numbers in early epochs, I STILL could not break 90.00%... Can pretty close though:*

`Epoch 75/100 - Train Loss: 0.0748, Train Acc: 0.9730 - Test Loss: 0.4307, Test Acc: 0.8994`
"""