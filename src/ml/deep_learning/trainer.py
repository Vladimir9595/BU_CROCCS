"""
Contains a more advanced training and evaluation loop for PyTorch models,
including per-epoch validation and early stopping.
"""
import torch
from tqdm import tqdm
import copy

def train_and_evaluate_cnn(model, train_loader, val_loader, epochs=50, learning_rate=0.001, patience=10):
    """
    Manages an advanced training and validation loop with early stopping.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_accuracy = 0.0
    epochs_no_improve = 0

    # This ensures that even if accuracy never improves, we have a valid state to load.
    best_model_state = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        # Training Phase
        model.train()
        running_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_pbar.set_postfix(loss=f"{running_loss / len(train_pbar):.4f}")

        # Validation Phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        epoch_accuracy = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} Summary: Val Loss: {avg_val_loss:.4f}, Val Acc: {epoch_accuracy:.2f}%")

        # Early Stopping Logic
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"  -> New best model saved with accuracy: {best_accuracy:.2f}%")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {patience} epochs with no improvement.")
            break

    # Evaluation using the BEST model state
    print(f"\nEvaluating with best model (Accuracy: {best_accuracy:.2f}%)")
    model.load_state_dict(best_model_state)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return best_accuracy, all_preds, all_labels