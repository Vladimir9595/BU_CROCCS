"""
Contains the machine learning pipeline for training and evaluating a classifier.
Includes a custom cross-validation strategy to test on one cycle per concentration.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def custom_cross_validator(df: pd.DataFrame):
    """
    A custom cross-validation generator.

    For each concentration group, this function iterates through the cycles,
    holding out one cycle at a time for testing and using the rest for training.

    Yields:
        tuple: A tuple of (train_indices, test_indices) for each fold.
    """
    df_indexed = df.reset_index() # Ensure we have a simple integer index

    # Find all unique concentrations and cycles in the dataset
    concentrations = sorted(df_indexed['concentration'].unique())
    cycles = sorted(df_indexed['cycle'].unique())

    # Iterate through each cycle, making it the hold-out test set for one fold
    for test_cycle in cycles:
        print(f"\nDefining fold: Testing on Cycle {test_cycle}, Training on others...")

        # Test indices are all rows where the cycle number matches the test_cycle
        test_indices = df_indexed[df_indexed['cycle'] == test_cycle].index.values

        # Train indices are all other rows
        train_indices = df_indexed[df_indexed['cycle'] != test_cycle].index.values

        yield train_indices, test_indices

def run_training_pipeline(df: pd.DataFrame):
    """
    Executes a custom cross-validation scheme to train and evaluate models.
    """
    if df is None or df.empty:
        print("Dataset is empty. Aborting training.")
        return

    X = np.vstack(df['features'].values)
    y = df['label'].values

    print("\n--- Starting Training Pipeline ---")
    print(f"Feature matrix shape (X): {X.shape}")
    print(f"Label vector shape (y): {y.shape}")

    models = {
        "SVM": make_pipeline(
            StandardScaler(),
            SVC(
                kernel='linear',
                class_weight='balanced',
                random_state=42
            )
        ),
        "Random Forest": make_pipeline(
            StandardScaler(),
            RandomForestClassifier(
                n_estimators=100,
                class_weight='balanced',
                random_state=42
            )
        )
    }

    for model_name, model in models.items():
        print(f"\n===== Evaluating Model: {model_name} =====")

        all_predictions = []
        all_true_labels = []

        # Use our new custom cross-validator
        for fold, (train_idx, test_idx) in enumerate(custom_cross_validator(df)):
            if len(train_idx) == 0 or len(test_idx) == 0:
                print(f"Skipping fold {fold+1} due to empty train/test set.")
                continue

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            all_predictions.extend(y_pred)
            all_true_labels.extend(y_test)

            print(f"Accuracy for this fold: {accuracy_score(y_test, y_pred):.2%}")

        # --- Final Evaluation ---
        print(f"\n--- Overall Cross-Validation Results for {model_name} ---")
        class_labels = sorted(df['label'].unique())

        # Handle cases where not all classes might be present in the predictions
        # This is important for robust reporting
        unique_preds_and_labels = sorted(list(set(all_true_labels) | set(all_predictions)))
        target_names = [f"{c}%" for c in unique_preds_and_labels]

        print(classification_report(all_true_labels, all_predictions, labels=unique_preds_and_labels, target_names=target_names, zero_division=0))

        cm = confusion_matrix(all_true_labels, all_predictions, labels=class_labels)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[f"{c}%" for c in class_labels],
                    yticklabels=[f"{c}%" for c in class_labels])
        plt.title(f"Confusion Matrix for {df['gas'].iloc[0].capitalize()} ({model_name})")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()