# ---------------------------------------------------------
# Full training pipeline with FoodResNetLite (ResNet-lite)
# ---------------------------------------------------------

# 0. Imports & Device
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from skorch import NeuralNetClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix
    , accuracy_score, ConfusionMatrixDisplay
)
from torchmetrics import Accuracy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
from timeit import default_timer as timer
from tqdm import tqdm
from pathlib import Path
from PIL import Image

warnings.filterwarnings("ignore")

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

device

# ---------------------------------------------------------
# 1. Paths and preprocessing
# ---------------------------------------------------------
train_dir = Path("data/pizza_steak_sushi/train")
test_dir  = Path("data/pizza_steak_sushi/test")

target_size = (128, 128)
imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std  = (0.229, 0.224, 0.225)

preprocess_to_tensor = transforms.Compose([
    transforms.Resize(target_size)
    , transforms.CenterCrop(target_size)
    , transforms.ToTensor()
    , transforms.Normalize(imagenet_mean, imagenet_std)
])

def load_imagefolder_to_tensors(root_dir: Path, transform):
    images, labels = [], []
    classes = sorted([d.name for d in root_dir.iterdir() if d.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    for class_name in classes:
        class_dir = root_dir / class_name
        for img_name in sorted(os.listdir(class_dir)):
            if img_name.startswith('.'):
                continue
            img_path = class_dir / img_name
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Skipping {img_path}: {e}")
                continue
            tensor = transform(img)
            images.append(tensor)
            labels.append(class_to_idx[class_name])
    if len(images) == 0:
        raise RuntimeError(f"No images found in {root_dir}")
    X = torch.stack(images)
    y = torch.tensor(labels, dtype=torch.long)
    return X, y, classes

X_train, y_train, class_names = load_imagefolder_to_tensors(train_dir, preprocess_to_tensor)
X_test,  y_test,  _          = load_imagefolder_to_tensors(test_dir,  preprocess_to_tensor)

num_classes = len(class_names)
print("Classes:", class_names)
print("X_train shape:", X_train.shape, "y_train shape:", y_train.shape)
print("X_test shape :", X_test.shape,  "y_test shape :", y_test.shape)

# ---------------------------------------------------------
# 2. Augmentations (for training loop only)
# ---------------------------------------------------------
train_transform_batch = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5)
    , transforms.RandomRotation(15)
    , transforms.RandomCrop(target_size)
    , transforms.Normalize(imagenet_mean, imagenet_std)
])

test_transform_batch = transforms.Compose([
    transforms.Normalize(imagenet_mean, imagenet_std)
])

train_loader = DataLoader(
    TensorDataset(X_train, y_train)
    , batch_size=32
    , shuffle=True
)

test_loader = DataLoader(
    TensorDataset(X_test, y_test)
    , batch_size=32
    , shuffle=False
)

# ---------------------------------------------------------
# 3. FoodResNetLite model
# ---------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3 
                               , stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3 
                               , stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1 
                          , stride=stride, bias=False)
                , nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class FoodResNetLite(nn.Module):
    def __init__(self, num_classes=3, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(
            ResidualBlock(64, 64)
            , ResidualBlock(64, 64)
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2)
            , ResidualBlock(128, 128)
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2)
            , ResidualBlock(256, 256)
        )
        self.layer4 = nn.Sequential(
            ResidualBlock(256, 512, stride=2)
            , ResidualBlock(512, 512)
        )

        self.global_avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(
            nn.Dropout(dropout)
            , nn.Linear(512, num_classes)
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.pool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.global_avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

# ---------------------------------------------------------
# 4. Skorch wrapper & baseline fit
# ---------------------------------------------------------
net = NeuralNetClassifier(
    module=FoodResNetLite
    , criterion=nn.CrossEntropyLoss
    , max_epochs=3
    , lr=1e-4
    , optimizer=torch.optim.Adam
    , batch_size=32
    , device=device
)

net.fit(X_train, y_train)
print("Baseline train accuracy:", net.score(X_train, y_train))

# ---------------------------------------------------------
# 5. Hyperparameter search
# ---------------------------------------------------------
param_dist = {
    "module__dropout": [0.2, 0.3, 0.4]
    , "lr": [1e-4, 3e-4, 1e-3]
    , "optimizer": [torch.optim.Adam, torch.optim.SGD, torch.optim.RMSprop]
    , "batch_size": [16, 32, 64]
    , "optimizer__momentum": [0.0, 0.9]
    , "optimizer__weight_decay": [0.0, 1e-4, 5e-4]
}

rs = RandomizedSearchCV(
    estimator=net
    , param_distributions=param_dist
    , n_iter=10
    , cv=2
    , scoring="accuracy"
    , verbose=2
    , random_state=27
    , n_jobs=-1
)

rs.fit(X_train, y_train)

print("Best CV accuracy:", rs.best_score_)
print("Best params:", rs.best_params_)
pd.DataFrame(rs.cv_results_).to_csv("food_resnetlite_rs_results.csv", index=False)

# ---------------------------------------------------------
# 6. Final model from best params
# ---------------------------------------------------------
best_params = rs.best_params_

model = FoodResNetLite(
    num_classes=num_classes
    , dropout=best_params["module__dropout"]
).to(device)

batch_size = int(best_params.get("batch_size", 32))

train_loader = DataLoader(
    TensorDataset(X_train, y_train)
    , batch_size=batch_size
    , shuffle=True
)
test_loader = DataLoader(
    TensorDataset(X_test, y_test)
    , batch_size=batch_size
    , shuffle=False
)

# ---------------------------------------------------------
# 7. Train/Test step functions
# ---------------------------------------------------------
def apply_train_transforms(batch_X):
    return train_transform_batch(batch_X)

def train_step(model, loader, loss_fn, optimizer, accuracy_fn, device):
    model.train()
    total_loss, total_acc = 0.0, 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        try:
            X = apply_train_transforms(X)
        except:
            pass
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
    total_loss, total_acc = 0.0, 0.0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            total_loss += loss.item()
            total_acc += accuracy_fn(y_pred.argmax(dim=1), y).item()
    return total_loss / len(loader), total_acc / len(loader)

# ---------------------------------------------------------
# 8. Final training loop
# ---------------------------------------------------------
epochs = 50  # ResNet-lite is heavier, start with 50 epochs
optimizer = best_params["optimizer"](model.parameters(), lr=best_params["lr"])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-5
)
loss_fn = nn.CrossEntropyLoss()
accuracy_fn = Accuracy(task='multiclass', num_classes=num_classes).to(device)

best_test_acc = 0.0
checkpoint_path = "food_resnetlite_best_checkpoint.pth"

for epoch in tqdm(range(epochs)):
    train_loss, train_acc = train_step(model, train_loader, loss_fn, optimizer, accuracy_fn, device)
    test_loss, test_acc = test_step(model, test_loader, loss_fn, accuracy_fn, device)
    scheduler.step(test_loss)
    print(
        f"Epoch {epoch+1}/{epochs} - "
        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
        f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}"
    )
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        torch.save({
            'epoch': epoch+1
            , 'model_state_dict': model.state_dict()
            , 'optimizer_state_dict': optimizer.state_dict()
            , 'test_acc': test_acc
            , 'class_names': class_names
        }, checkpoint_path)
        print(f"Saved best checkpoint (test_acc={test_acc:.4f}) -> {checkpoint_path}")

print("Best test acc observed:", best_test_acc)

# ---------------------------------------------------------
# 9. Final evaluation
# ---------------------------------------------------------
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        all_preds.extend(y_pred.argmax(dim=1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())

print("\nFinal Evaluation")
print("Accuracy:", accuracy_score(all_labels, all_preds))
print("\nClassification Report:\n", classification_report(all_labels, all_preds, target_names=class_names))

fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(all_labels, all_preds, display_labels=class_names, normalize='true', ax=ax, cmap="Blues")
plt.title("Confusion Matrix - FoodResNetLite Final Model")
plt.show()

# ---------------------------------------------------------
# 10. Save final model
# ---------------------------------------------------------
torch.save(model.state_dict(), "food_resnetlite_final_weights.pth")
print("Saved final weights.")
