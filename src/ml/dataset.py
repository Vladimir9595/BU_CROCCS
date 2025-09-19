"""
Handles loading, resampling, feature extraction, and aggregation of the dataset.
"""
import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from src.ml.feature_engineering import extract_features

# resample_signal function remains the same
def resample_signal(signal: np.ndarray, target_length: int) -> np.ndarray:
    if len(signal) == target_length: return signal
    original_indices = np.linspace(0, 1, num=len(signal))
    interp_func = interp1d(original_indices, signal)
    resampled_indices = np.linspace(0, 1, num=target_length)
    return interp_func(resampled_indices)

def group_by_cycle(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates features from all 63 sensors for each unique (concentration, cycle)
    pair into a single "super-feature" vector.
    """
    print("Grouping data by cycle and concentration to create full-array feature vectors...")

    # Sort to ensure consistent sensor order
    df_sorted = df.sort_values(by=['concentration', 'cycle', 'sensor_id'])

    agg_funcs = {
        'gas': 'first',
        'label': 'first',
        'features': lambda feature_list: np.concatenate(feature_list.tolist())
    }

    # Group by the unique experiment run
    df_grouped = df_sorted.groupby(['concentration', 'cycle']).agg(agg_funcs).reset_index()
    return df_grouped

def load_dataset_for_gas(gas_name: str, use_feature_engineering: bool = True, use_full_array: bool = False):
    """
    Loads data, validates, resamples, engineers features, and optionally groups by cycle.
    """
    # ... (the first part of the function remains the same) ...
    base_path = os.path.join('extracted_data', gas_name)
    if not os.path.isdir(base_path):
        print(f"Error: Directory for '{gas_name}' not found.")
        return None
    all_records = []
    print(f"Loading extracted data for '{gas_name}'...")
    expected_sensors_per_cycle = 7 * 9
    for conc_folder in sorted(os.listdir(base_path)):
        if not conc_folder.startswith('concentration_'): continue
        concentration = int(conc_folder.split('_')[1])
        conc_path = os.path.join(base_path, conc_folder)
        for cycle_folder in sorted(os.listdir(conc_path)):
            if not cycle_folder.startswith('Cycle_'): continue
            cycle = int(cycle_folder.split('_')[1])
            cycle_path = os.path.join(conc_path, cycle_folder)
            num_files = len([f for f in os.listdir(cycle_path) if f.endswith('.csv')])
            if num_files != expected_sensors_per_cycle:
                print(f"Warning: Cycle {cycle} in {conc_folder} is incomplete. Skipping.")
                continue
            for sensor_file in sorted(os.listdir(cycle_path)):
                sensor_id = sensor_file.replace('.csv', '')
                file_path = os.path.join(cycle_path, sensor_file)
                try:
                    intensity_data = np.loadtxt(file_path, delimiter=',', skiprows=1, usecols=1)
                    if intensity_data.ndim == 0: continue
                    all_records.append({
                        "gas": gas_name, "concentration": concentration, "cycle": cycle,
                        "sensor_id": sensor_id, "features": intensity_data, "label": concentration
                    })
                except Exception as e:
                    print(f"Warning: Could not process {file_path}. Error: {e}")
    if not all_records:
        print(f"No complete data records found for gas '{gas_name}'.")
        return None
    df = pd.DataFrame(all_records)
    print(f"Successfully loaded {len(df)} individual sensor samples.")

    feature_lengths = df['features'].apply(len)
    if feature_lengths.nunique() > 1:
        target_length = int(feature_lengths.median())
        print(f"Resampling all signals to a fixed length of {target_length}.")
        df['features'] = df['features'].apply(lambda x: resample_signal(x, target_length))

    if use_feature_engineering:
        print("Applying feature engineering...")
        df['features'] = df['features'].apply(extract_features)
    else:
        print("Using raw resampled signal as features.")

    if use_full_array:
        df = group_by_cycle(df)
        print(f"Data aggregated into {len(df)} full-array samples.")

    return df