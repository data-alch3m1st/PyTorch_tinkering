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


# extract class_names (for plotting)
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

# ---------------------------------------------------------
# 0a. Functionize the train & test loop(s):
# ---------------------------------------------------------

def train_step(
    model: torch.nn.Module
    , data_loader: torch.utils.data.DataLoader
    , loss_fn: torch.nn.Module
    , optimizer: torch.optim.Optimizer
    , accuracy_fn
    , device: torch.device = device):
    
    train_loss, train_acc = 0, 0
    model.to(device)
    for batch, (X, y) in enumerate(data_loader):
        # Send data to GPU
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(
            y_true=y
            , y_pred=y_pred.argmax(dim=1)) # Go from logits -> pred labels

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

def test_step(
    data_loader: torch.utils.data.DataLoader
    , model: torch.nn.Module
    , loss_fn: torch.nn.Module
    , accuracy_fn
    , device: torch.device = device):
    
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval() # put model in eval mode
    # Turn on inference context manager
    with torch.inference_mode(): 
        for X, y in data_loader:
            # Send data to GPU
            X, y = X.to(device), y.to(device)
            
            # 1. Forward pass
            test_pred = model(X)
            
            # 2. Calculate loss and accuracy
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(
                y_true=y
                , y_pred=test_pred.argmax(dim=1) # Go from logits -> pred labels
            )
        
        # Adjust metrics and print out
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")


# ---------------------------------------------------------
# 0b. Import accuracy metric (custom or Torch native;)
# ---------------------------------------------------------

from helper_functions import accuracy_fn 
# Note: could also use torchmetrics.Accuracy(task = 'multiclass', num_classes=len(class_names)).to(device)

import torchmetrics

accuracy_torch_fn = torchmetrics.Accuracy(task = 'multiclass', num_classes=len(class_names)).to(device)

accuracy_fn
accuracy_torch_fn

# ---------------------------------------------------------
# 0c. Build & functionize a custom timer (for train/test loops;)
# ---------------------------------------------------------

from timeit import default_timer as timer 

def print_train_time(start: float, end: float, device: torch.device = None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time


# ---------------------------------------------------------
# 0d. Build & functionize a custom eval_mode
# ---------------------------------------------------------

# Move values to device
torch.manual_seed(27)
def eval_model(
    model: torch.nn.Module
    , data_loader: torch.utils.data.DataLoader 
    , loss_fn: torch.nn.Module
    , accuracy_fn
    , device: torch.device = device):   
    
    """Evaluates a given model on a given dataset.

    Args:
        model (torch.nn.Module): A PyTorch model capable of making predictions on data_loader.
        data_loader (torch.utils.data.DataLoader): The target dataset to predict on.
        loss_fn (torch.nn.Module): The loss function of model.
        accuracy_fn: An accuracy function to compare the models predictions to the truth labels.
        device (str, optional): Target device to compute on. Defaults to device.

    Returns:
        (dict): Results of model making predictions on data_loader.
    """
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            # Send data to the target device
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(
                y_true=y
                , y_pred=y_pred.argmax(dim=1))
        
        # Scale loss and acc
        loss /= len(data_loader)
        acc /= len(data_loader)
    return {"model_name": model.__class__.__name__ 
            # ^^^ only works when model was created with a class
            , "model_loss": loss.item()
            , "model_acc": acc}


# ---------------------------------------------------------
# 1. Extract best parameters from skorch RandomizedSearchCV
# ---------------------------------------------------------
best_params = rs.best_params_
print("Best parameters:", best_params)

# Example: {'module__channels': 64, 'module__dropout': 0.25, 'lr': 0.0003, 'optimizer': torch.optim.Adam, 'batch_size': 256}

# ---------------------------------------------------------
# 2. Rebuild the PyTorch CNN model with tuned params
# ---------------------------------------------------------
model = SimpleCNN(
    channels=best_params["module__channels"]
    , dropout=best_params["module__dropout"]
).to(device)

# ---------------------------------------------------------
# 3. Define loss and optimizer using tuned learning rate
# ---------------------------------------------------------
loss_fn = nn.CrossEntropyLoss()

optimizer = best_params["optimizer"](
    params=model.parameters()
    , lr=best_params["lr"]
)

# ---------------------------------------------------------
# 4. Prepare DataLoaders (with tuned batch size)
# ---------------------------------------------------------
train_dataloader = DataLoader(
    train_dataset
    , batch_size=best_params["batch_size"]
    , shuffle=True
)

test_dataloader = DataLoader(
    test_dataset
    , batch_size=best_params["batch_size"]
    , shuffle=False
)

# ---------------------------------------------------------
# 5. Training + Testing Loops
# ---------------------------------------------------------
torch.manual_seed(27)

from timeit import default_timer as timer
from tqdm.auto import tqdm

train_time_start = timer()

epochs = 20
for epoch in tqdm(range(epochs)):
    print(f"\nEpoch: {epoch+1}/{epochs}")
    
    # --- Training ---
    train_step(
        data_loader=train_dataloader
        , model=model
        , loss_fn=loss_fn
        , optimizer=optimizer
        , accuracy_fn=accuracy_fn
        , device=device
    )

    # --- Testing ---
    test_step(
        data_loader=test_dataloader
        , model=model
        , loss_fn=loss_fn
        , accuracy_fn=accuracy_fn
        , device=device
    )

train_time_end = timer()
total_train_time = print_train_time(
    start=train_time_start
    , end=train_time_end
    , device=device
)

print(f"Training complete in {total_train_time:.2f} seconds")


# ---------------------------------------------------------
# 6. Get Model Results
# ---------------------------------------------------------

model_0x_results = eval_model(
    model=model_0x
    , data_loader=test_dataloader
    , loss_fn=loss_fn
    , accuracy_fn=accuracy_fn
)
model_0x_results


# ---------------------------------------------------------
# 7a. Build a make_predictions function
# ---------------------------------------------------------

def make_predictions(model: torch.nn.Module, data: list, device: torch.device = device):
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data:
            # Prepare sample
            sample = torch.unsqueeze(sample, dim=0).to(device) # Add an extra dimension and send sample to device

            # Forward pass (model outputs raw logit)
            pred_logit = model(sample)

            # Get prediction probability (logit -> prediction probability)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0) # note: perform softmax on the "logits" dimension, not "batch" dimension (in this case we have a batch size of 1, so can perform on dim=0)

            # Get pred_prob off GPU for further calculations
            pred_probs.append(pred_prob.cpu())
            
    # Stack the pred_probs to turn list into a tensor
    return torch.stack(pred_probs)


# ---------------------------------------------------------
# 7b. Make actual predictions!
# ---------------------------------------------------------

# Make predictions on test samples with model 2
pred_probs= make_predictions(model=model_0x, 
                             data=test_samples)

# View first two prediction probabilities list
pred_probs[:2]


# Turn the prediction probabilities into prediction labels by taking the argmax()
pred_classes = pred_probs.argmax(dim=1)
pred_classes

# Are our predictions in the same form as our test labels? 
test_labels, pred_classes



# ---------------------------------------------------------
# 7c. Plot predictions visually
# ---------------------------------------------------------

# Plot predictions
plt.figure(figsize=(9, 9))
nrows = 3
ncols = 3
for i, sample in enumerate(test_samples):
  # Create a subplot
  plt.subplot(nrows, ncols, i+1)

  # Plot the target image
  plt.imshow(sample.squeeze(), cmap="gray")

  # Find the prediction label (in text form, e.g. "Sandal")
  pred_label = class_names[pred_classes[i]]

  # Get the truth label (in text form, e.g. "T-shirt")
  truth_label = class_names[test_labels[i]] 

  # Create the title text of the plot
  title_text = f"Pred: {pred_label} | Truth: {truth_label}"
  
  # Check for equality and change title colour accordingly
  if pred_label == truth_label:
      plt.title(title_text, fontsize=10, c="g") # green text if correct
  else:
      plt.title(title_text, fontsize=10, c="r") # red text if wrong
  plt.axis(False);

# ---------------------------------------------------------
# 8. Final Confusion Matrix
# ---------------------------------------------------------


from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

# Create confusion matrix
cm2 = ConfusionMatrix(
    num_classes=len(class_names)
    , task='multiclass'
#     , normalize="true"
)

cm2_tensor = cm2(
    preds=y_pred_tensor
    , target=test_dataset.targets)

# Convert to NumPy
cm2_numpy = cm2_tensor.numpy()

# Plot confusion matrix with formatted values
fig, ax = plot_confusion_matrix(
    conf_mat=cm2_numpy
    , class_names=class_names
    , figsize=(12, 9)
    , show_normed=True  # Normalized values
)


