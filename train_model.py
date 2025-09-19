"""
Main script to run the machine learning training pipeline.
"""
import argparse
from src.ml.dataset import load_dataset_for_gas
from src.ml.training import run_training_pipeline, run_deep_learning_pipeline

def main():
    """
    Parses arguments and runs the training pipeline.
    """
    parser = argparse.ArgumentParser(description="Train a concentration classifier.")
    parser.add_argument("gas_name", type=str, help="The gas to train on (e.g., 'ammonia').")
    parser.add_argument(
        "--use-statistical-features",
        action='store_true',
        help="Use statistical summary features instead of the raw signal."
    )
    parser.add_argument(
        "--model-type", type=str, default="classic", choices=['classic', 'cnn'],
        help="Type of model to train: 'classic' (SVM, RF) or 'cnn' (PyTorch)."
    )
    parser.add_argument(
        "--use-full-array",
        action='store_true',
        help="Use the combined signal from all 63 sensors as a single feature vector."
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Number of epochs for CNN training."
    )
    args = parser.parse_args()

    gas_to_train = args.gas_name.lower()
    use_feature_engineering_flag = args.use_statistical_features
    if args.model_type == 'cnn':
        use_feature_engineering_flag = False

    dataset_df = load_dataset_for_gas(
        gas_to_train,
        use_feature_engineering=use_feature_engineering_flag,
        use_full_array=args.use_full_array
    )

    if dataset_df is not None:
        if args.model_type == 'classic':
            # Classic models can handle both single and full array features
            run_training_pipeline(dataset_df)
        elif args.model_type == 'cnn':
            # Pass the flag to the deep learning pipeline
            run_deep_learning_pipeline(dataset_df, epochs=args.epochs, use_full_array=args.use_full_array)
    else:
        print(f"Could not proceed with training for gas '{gas_to_train}'.")

if __name__ == "__main__":
    main()