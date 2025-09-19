"""
Contains the machine learning pipeline for training and evaluating a classifier.
This version correctly implements Leave-One-Concentration-Out cross-validation
and supports both classic models and 1D/2D Deep Learning models.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from src.ml.deep_learning.dataset import SensorDataset
from src.ml.deep_learning.model import Enhanced1DCNN, SensorArrayCNN
from src.ml.deep_learning.trainer import train_and_evaluate_cnn

def run_deep_learning_pipeline(df: pd.DataFrame, epochs: int, use_full_array: bool):
    """
    Executes a Leave-One-Concentration-Out cross-validation for the PyTorch CNN model.
    Automatically selects 1D or 2D CNN based on the 'use_full_array' flag.
    """
    if df is None or df.empty:
        print("Dataset is empty. Aborting training.")
        return

    if use_full_array:
        # For the 2D CNN, each sample is one full experiment cycle (all 63 sensors)
        # The features are already concatenated into a long 1D vector by the data loader
        num_sensors = 7 * 9
        num_timesteps = df['features'].iloc[0].shape[0] // num_sensors

        # Reshape the features into a 2D "image" for each sample: (N, H, W)
        features_2d = [f.reshape(num_sensors, num_timesteps) for f in df['features']]
        X = np.array(features_2d)
        model_type = "2D CNN (Full Array)"
    else:
        # For the 1D CNN, each sample is a single sensor's signal
        X = np.vstack(df['features'].values)
        model_type = "1D CNN (Single Sensor)"

    y = df['label'].values
    groups = df['concentration'].values

    class_labels = sorted(df['label'].unique())
    num_classes = len(class_labels)
    global_label_map = {label: i for i, label in enumerate(class_labels)}
    reverse_global_label_map = {i: label for label, i in global_label_map.items()}

    print(f"\n--- Starting Deep Learning Pipeline ({model_type}) ---")
    print("--- Using Leave-One-CONCENTRATION-Out Cross-Validation ---")
    print(f"Feature matrix shape (X): {X.shape}")
    print(f"Number of classes: {num_classes}")

    all_predictions, all_true_labels = [], []
    logo = LeaveOneGroupOut()
    unique_groups = sorted(np.unique(groups))

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        test_concentration = groups[test_idx][0]
        print(f"\n--- Fold {fold + 1}/{len(unique_groups)}: Testing on Concentration {test_concentration}% ---")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Pass the is_2d flag to the dataset constructor
        train_dataset = SensorDataset(X_train, y_train, global_label_map, is_2d=use_full_array)
        val_dataset = SensorDataset(X_test, y_test, global_label_map, is_2d=use_full_array)

        train_loader = DataLoader(train_dataset, batch_size=16 if use_full_array else 32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16 if use_full_array else 32, shuffle=False)

        # Initialize the correct model based on the data format
        if use_full_array:
            model = SensorArrayCNN(num_sensors=X.shape[1], num_timesteps=X.shape[2], num_classes=num_classes)
        else:
            model = Enhanced1DCNN(num_features=X.shape[1], num_classes=num_classes)

        accuracy, preds, labels = train_and_evaluate_cnn(model, train_loader, val_loader, epochs=epochs)

        all_predictions.extend(preds)
        all_true_labels.extend(labels)
        print(f"Accuracy for this fold: {accuracy:.2f}%")

    # Final Evaluation
    print(f"\n--- Overall Cross-Validation Results for {model_type} ---")
    true_labels_orig = [reverse_global_label_map[int(l)] for l in all_true_labels]
    preds_orig = [reverse_global_label_map[int(p)] for p in all_predictions]

    print(classification_report(true_labels_orig, preds_orig, labels=class_labels, target_names=[f"{c}%" for c in class_labels], zero_division=0))

    cm = confusion_matrix(true_labels_orig, preds_orig, labels=class_labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f"{c}%" for c in class_labels],
                yticklabels=[f"{c}%" for c in class_labels])
    plt.title(f"Confusion Matrix for {df['gas'].iloc[0].capitalize()} ({model_type})")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()


def run_training_pipeline(df: pd.DataFrame):
    """
    Executes a Leave-One-Concentration-Out cross-validation for classic ML models.
    """
    if df is None or df.empty:
        print("Dataset is empty. Aborting training.")
        return

    X = np.vstack(df['features'].values)
    y = df['label'].values
    groups = df['concentration'].values

    print("\n--- Starting Classic ML Pipeline ---")
    print("--- Using Leave-One-CONCENTRATION-Out Cross-Validation ---")
    print(f"Feature matrix shape (X): {X.shape}")
    print(f"Label vector shape (y): {y.shape}")

    models = {
        "SVM": make_pipeline(StandardScaler(), SVC(kernel='linear', class_weight='balanced', random_state=42)),
        "Random Forest": make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
    }

    for model_name, model in models.items():
        print(f"\n===== Evaluating Model: {model_name} =====")
        all_predictions, all_true_labels = [], []
        logo = LeaveOneGroupOut()
        unique_groups = sorted(np.unique(groups))

        for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
            test_concentration = groups[test_idx][0]
            print(f"\n--- Fold {fold + 1}/{len(unique_groups)}: Testing on Concentration {test_concentration}% ---")

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            all_predictions.extend(y_pred)
            all_true_labels.extend(y_test)

            print(f"Accuracy for this fold: {accuracy_score(y_test, y_pred):.2%}")

        print(f"\n--- Overall Cross-Validation Results for {model_name} ---")
        class_labels = sorted(df['label'].unique())
        print(classification_report(all_true_labels, all_predictions, target_names=[f"{c}%" for c in class_labels], zero_division=0))

        cm = confusion_matrix(all_true_labels, all_predictions, labels=class_labels)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[f"{c}%" for c in class_labels],
                    yticklabels=[f"{c}%" for c in class_labels])
        plt.title(f"Confusion Matrix for {df['gas'].iloc[0].capitalize()} ({model_name})")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()