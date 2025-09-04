import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from skorch import NeuralNetClassifier
from sklearn.model_selection import RandomizedSearchCV
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


# MacoOS device agnostic code:

if torch.cuda.is_available():
    device = "cuda" # Use NVIDIA GPU (if available)
elif torch.backends.mps.is_available():
    device = "mps" # Use Apple Silicon GPU (if available)
else:
    device = "cpu" # Default to CPU if no GPU is available

device


# ---------------------------------------------------------
# Dataset Preparation
# ---------------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor()
    , transforms.Normalize((0.5,), (0.5,))
])

# Load datasets (train/test split handled by FashionMNIST)
train_dataset = datasets.FashionMNIST(
    './data', train=True
    , download=True, transform=transform)

test_dataset = datasets.FashionMNIST(
    './data', train=False
    , download=True, transform=transform)

# Use DataLoaders directly, avoiding manual tensor creation
train_loader = DataLoader(
    train_dataset
    , batch_size=256
    , shuffle=True)

test_loader = DataLoader(
    test_dataset
    , batch_size=256
    , shuffle=False)




# Define the CNN Module

class SimpleCNN(nn.Module):
    def __init__(self, channels=32, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(1, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1)
        self.fc1 = nn.Linear((channels * 2) * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(self.dropout(x)))
        return self.fc2(x)



# Wrap with Skorch NeuralNetClassifier
net = NeuralNetClassifier(
    module=SimpleCNN
    , criterion=nn.CrossEntropyLoss
    , max_epochs=3
    , lr=0.001
    , optimizer=torch.optim.Adam
    , batch_size=256
    , iterator_train__shuffle=True
    , device=device
)

# Fit the model
net.fit(train_dataset.data.float().unsqueeze(1), train_dataset.targets)

print("Train accuracy:", net.score(
    train_dataset.data.float().unsqueeze(1)
    , train_dataset.targets))



%%time

# RandomizedSearchCV for Hyperparameter Tuning

param_dist = {
    "module__channels": [32, 64]
    , "module__dropout": [0.0, 0.25, 0.5]
    , "lr": [1e-4, 3e-4, 1e-3]
    , "optimizer": [torch.optim.Adam]
    , "batch_size": [256]
}

rs = RandomizedSearchCV(
    estimator=net
    , param_distributions=param_dist
    , n_iter=10
    , cv=2
    , scoring="accuracy"
    , verbose=1
    , random_state=27
    , n_jobs=-1
)

rs.fit(train_dataset.data.float().unsqueeze(1), train_dataset.targets)

print("Best CV accuracy:", rs.best_score_)
print("Best params:", rs.best_params_)



# Send results to a dataframe for clean review;

results = pd.DataFrame(rs.cv_results_)
results[[
    "mean_test_score"
    , "std_test_score"
    , "params"
]].sort_values("mean_test_score", ascending=False, inplace=True)
results.head(3)



# Evaluate Performance with Best Model
# ---------------------------------------------------------
best_model = rs.best_estimator_
best_model.get_params_for_optimizer


y_pred = best_model.predict(test_dataset.data.float().unsqueeze(1))

# Print metrics
print("Accuracy:", accuracy_score(test_dataset.targets, y_pred))
print("Classification Report:\\n", classification_report(test_dataset.targets, y_pred))

# Confusion Matrix
cm = ConfusionMatrix(num_classes=10, task='multiclass')
cm_tensor = cm(torch.tensor(y_pred), test_dataset.targets)



fig, ax = plt.subplots(figsize=(12, 10)) 

# Plot confusion matrix
# disp = ConfusionMatrixDisplay.from_predictions(
#     test_dataset.targets, y_pred
#     , display_labels=train_dataset.classes
#     , cmap='Blues'
#     , normalize='true', xticks_rotation=45
# );

disp.plot(ax=ax, cmap='Blues')
plt.show();



# Training/Test Loops (PyTorch Native)

def train_step(model, loader, loss_fn, optimizer, accuracy_fn, device):
    model.train()
    total_loss, total_acc = 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
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
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            total_loss += loss.item()
            total_acc += accuracy_fn(y_pred.argmax(dim=1), y).item()
    return total_loss / len(loader), total_acc / len(loader)



# Set the final model with the best params (need to incl channels & dropout as such:)
model = SimpleCNN(
    channels=rs.best_params_["module__channels"]
    , dropout=rs.best_params_["module__dropout"]
).to(device)

model



%%time

# Train the Best Model

optimizer = rs.best_params_["optimizer"](model.parameters(), lr=rs.best_params_["lr"])
loss_fn = nn.CrossEntropyLoss()
accuracy_fn = Accuracy(task='multiclass', num_classes=10).to(device)

epochs = 20
for epoch in tqdm(range(epochs)):
    train_loss, train_acc = train_step(
        model, train_loader, loss_fn, optimizer, accuracy_fn, device)
    test_loss, test_acc = test_step(
        model, test_loader, loss_fn, accuracy_fn, device)
    print(
        f"Epoch {epoch+1}/{epochs} - "
        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f} - "
        f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}"
    )



   # Final Confusion Matrix for FashionMNIST

# Ensure the model is in evaluation mode
model.eval()

# Collect all predictions and true labels
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        # Send data to device
        images, labels = images.to(device), labels.to(device)

        # Get predictions
        outputs = model(images)
        preds = outputs.argmax(dim=1)

        # Store predictions and true labels
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

# Concatenate all predictions and labels into single tensors
all_preds = torch.cat(all_preds)
all_labels = torch.cat(all_labels)



print(confusion_matrix(all_labels, all_preds))
print('\n=================================================================\n')
print(classification_report(all_labels, all_preds))



# Compute the confusion matrix

fig, ax = plt.subplots(figsize=(12, 10)) 

# Display the normalized confusion matrix
# disp = ConfusionMatrixDisplay.from_predictions(
#     all_labels, all_preds
#     , display_labels=train_dataset.classes  # Use FashionMNIST class names
#     , normalize='true'
# )
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix for FashionMNIST - Final Model")
plt.show();
