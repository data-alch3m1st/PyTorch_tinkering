# Imports 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix
    , accuracy_score, ConfusionMatrixDisplay
    )

import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# MacoOS device agnostic code:
# ---------------------------------------------------------

if torch.cuda.is_available():
    device = "cuda" # Use NVIDIA GPU (if available)
elif torch.backends.mps.is_available():
    device = "mps" # Use Apple Silicon GPU (if available)
else:
    device = "cpu" # Default to CPU if no GPU is available


# ---------------------------------------------------------
# 1. Prepare dataset (FashionMNIST)
# ---------------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),    
    transforms.Normalize((0.5,), (0.5,))
])

# Note: to get natural train/test splits for FashionMNIST: train-> train=True ; test-> train=False ;
train_dataset = datasets.FashionMNIST(
    './data', train=True
    , download=True, transform=transform)

test_dataset = datasets.FashionMNIST(
    './data', train=False
    , download=True, transform=transform)

# Skorch can handle PyTorch Datasets directly
X_train, y_train = train_dataset.data.float().unsqueeze(1), train_dataset.targets
X_test, y_test = test_dataset.data.float().unsqueeze(1), test_dataset.targets

X_train.shape, y_train.shape, X_test.shape, y_test.shape


# ---------------------------------------------------------
# 2. Define your CNN module
# ---------------------------------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, channels=32, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(1, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1)
        self.fc1   = nn.Linear((channels * 2) * 7 * 7, 128)
        self.fc2   = nn.Linear(128, 10)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(self.dropout(x)))
        x = self.fc2(x)
        return x

# ---------------------------------------------------------
# 3. Wrap model with skorch
# ---------------------------------------------------------
net = NeuralNetClassifier(
    module=SimpleCNN
    , criterion=nn.CrossEntropyLoss
    , max_epochs=10
    , lr=0.001
    , optimizer=torch.optim.Adam
    , batch_size=256
    , iterator_train__shuffle=True
    , device=device
)

net.fit(X_train, y_train)
print("Train accuracy:", net.score(X_train, y_train))



# ---------------------------------------------------------
# 4a. RandomSearch Params (sample hyperparams combos)
# ---------------------------------------------------------

param_dist = {
    "module__channels": [32, 64]
    , "module__dropout": [0.0, 0.25, 0.5]
    , "lr": [1e-4, 3e-4, 1e-3]
    , "optimizer": [torch.optim.Adam]
    , "batch_size": [256]
}


# ---------------------------------------------------------
# 4b. Randomized Search Actual (sample combos, actual run;)
# ---------------------------------------------------------
rs = RandomizedSearchCV(
    estimator=net
    , param_distributions=param_dist
    , n_iter=10          # try only 10 random combos
    , cv=2
    , scoring="accuracy"
    , verbose=1
    , random_state=27
    , n_jobs=-1
)

# Fit the randomized search
rs.fit(X_train, y_train)

print("Best CV accuracy:", rs.best_score_)
print("Best params:", rs.best_params_)

# ---------------------------------------------------------
# 5. Send results to a dataframe for clean review;
# ---------------------------------------------------------
results = pd.DataFrame(rs.cv_results_)
results = results[[
    "mean_test_score"
    , "std_test_score"
    , "params"
]].sort_values("mean_test_score", ascending=False, inplace=True)
results.head(3)
results

# ---------------------------------------------------------
# iloc[0] will display our top result given the sort ^^^
results['params'].iloc[0] 

# ---------------------------------------------------------
# 6. Set up the best performing model using our best_estimator_
# ---------------------------------------------------------
best_rand_cv = rs.best_estimator_
best_rand_cv


# See classes
class_names = train_dataset.classes
class_names

# ---------------------------------------------------------
# 7. Predict with best estimator
# ---------------------------------------------------------
y_pred = rs.best_estimator_.predict(
    X_test)

# Metrics
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

acc
print(report)
cm

# ---------------------------------------------------------
# 8a. Confusiom Matrix to show broad view of model performance
# ---------------------------------------------------------
cmp = ConfusionMatrixDisplay.from_estimator(
    rs, X_test, y_test
    , normalize='true'
    , display_labels=class_names
);

# ---------------------------------------------------------
# 8b. Plot the Confusion Matrix
# ---------------------------------------------------------
# Create a Matplotlib figure and axes with the desired figsize
fig, ax = plt.subplots(figsize=(12, 9)) # Adjust width and height as needed

# Create a ConfusionMatrixDisplay object (note creating an obj then running the object still outputs a mini cm, which looks strange, so just adding '.plot' to the end of the unnamed obj;)

cmp.plot(ax=ax, cmap='Blues')

# Add a title (optional)
ax.set_title('Confusion Matrix')

# Display the plot
plt.show();


# ----------------------------------------------------------------------------------------------------- #
######### PART II.: Test/Train/Eval Loops (PyTorch Native) on best performing model (params) #########
# ----------------------------------------------------------------------------------------------------- #


