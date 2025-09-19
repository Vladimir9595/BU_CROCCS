"""
Contains functions to extract meaningful statistical features from raw time-series signals.
"""
import numpy as np
from scipy import stats

def extract_features(signal: np.ndarray) -> np.ndarray:
    """
    Extracts a feature vector from a single time-series signal.

    Args:
        signal (np.ndarray): A 1D numpy array representing a sensor's response over time.

    Returns:
        np.ndarray: A 1D numpy array of calculated features.
    """
    if signal is None or len(signal) < 2:
        # Return a zero vector of the expected feature length if signal is invalid
        return np.zeros(8)

    features = [
        np.mean(signal),          # 1. Mean value
        np.std(signal),           # 2. Standard deviation (volatility)
        np.min(signal),           # 3. Minimum value
        np.max(signal),           # 4. Maximum value
        np.max(signal) - np.min(signal), # 5. Peak-to-peak amplitude
        stats.skew(signal),       # 6. Skewness (asymmetry)
        stats.kurtosis(signal),   # 7. Kurtosis (tailedness)
        np.polyfit(np.arange(len(signal)), signal, 1)[0] # 8. Slope of the linear trend
    ]

    return np.array(features)