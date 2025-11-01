# torch_helper_v2.py #
# Lightweight Model-Agnostic PyTorch training/evaluation helpers for notebooks & scripts.

from typing import List, Optional, Tuple
import torch

try:
    from tqdm.auto import tqdm  # optional, only used if show_progress=True
except Exception:
    tqdm = None  # fallback if tqdm isn't available


def _progress_iter(iterable, enable: bool, desc: Optional[str] = None):
    """
    Internal: wrap iterable with tqdm if available and requested.
    """
    if enable and tqdm is not None:
        return tqdm(iterable, desc=desc)
    return iterable


def train_epoch_loop(
    model: torch.nn.Module
    , train_loader: torch.utils.data.DataLoader
    , optimizer: torch.optim.Optimizer
    , criterion: torch.nn.Module
    , device: torch.device
    , scheduler: Optional[object] = None
    , epochs: int = 6
    , show_progress: bool = True
) -> Tuple[List[float], List[float]]:
    """
    Training-only loop over a specified number of epochs.
    Returns lists of train losses and accuracies for plotting later.

    Args:
        model:        torch.nn.Module (already moved to device)
        train_loader: DataLoader for training data
        optimizer:    torch optimizer
        criterion:    loss function
        device:       torch.device
        scheduler:    optional LR scheduler with .step()
        epochs:       number of epochs to run
        show_progress:show tqdm progress bars if True

    Returns:
        (train_losses, train_accuracies)
    """
    train_losses: List[float] = []
    train_accuracies: List[float] = []

    epoch_iter = _progress_iter(range(epochs), show_progress, desc="Training")

    for epoch in epoch_iter:
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        batch_iter = _progress_iter(train_loader, show_progress, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, labels in batch_iter:
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

        train_loss = (running_loss / total) if total > 0 else 0.0
        train_acc = (correct / total * 100) if total > 0 else 0.0

        if scheduler is not None:
            scheduler.step()

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Print epoch metrics and current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Acc: {train_acc:.4f}% | LR: {current_lr:.6f}")

    return train_losses, train_accuracies


def evaluate_model(
    model: torch.nn.Module
    , test_loader: torch.utils.data.DataLoader
    , device: torch.device
    , criterion: Optional[torch.nn.Module] = None
    , show_progress: bool = False
) -> Tuple[float, Optional[float]]:
    """
    Evaluates a model on a test/validation set and (optionally) returns loss.

    Args:
        model:        torch.nn.Module (already moved to device)
        test_loader:  DataLoader for test/validation data
        device:       torch.device ('cuda' or 'cpu')
        criterion:    optional loss function; if provided, returns avg test loss
        show_progress:wrap loop with tqdm if True

    Returns:
        (test_acc, test_loss) where test_loss may be None if criterion is None
    """
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0

    iterator = _progress_iter(test_loader, show_progress, desc="Testing")

    with torch.no_grad():
        for inputs, labels in iterator:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            correct += torch.sum(preds == labels).item()
            total += labels.size(0)

            if criterion is not None:
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)

    test_acc = (correct / total * 100) if total > 0 else 0.0
    test_loss = (running_loss / total) if (criterion is not None and total > 0) else None

    print(f"Test Accuracy: {test_acc:.4f}%")
    if test_loss is not None:
        print(f"Test Loss: {test_loss:.4f}")

    return test_acc, test_loss


__all__ = [
    "train_epoch_loop",
    , "evaluate_model"
]



# Including a separate eval function with params enabled to facilitate sklearn style metrics (confusion matrix, classification report, etc.)

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