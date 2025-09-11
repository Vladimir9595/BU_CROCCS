"""
Main script to run the machine learning training pipeline.
"""
import argparse
from src.ml.dataset import load_dataset_for_gas
from src.ml.training import run_training_pipeline

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
    args = parser.parse_args()

    gas_to_train = args.gas_name.lower()

    # Feature engineering is now default OFF (raw signal is better)
    use_feature_engineering_flag = args.use_statistical_features
    dataset_df = load_dataset_for_gas(gas_to_train, use_feature_engineering=use_feature_engineering_flag)

    if dataset_df is not None:
        run_training_pipeline(dataset_df)
    else:
        print(f"Could not proceed with training for gas '{gas_to_train}'.")

if __name__ == "__main__":
    main()