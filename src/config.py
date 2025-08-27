"""
Central configuration file for the VOC sensor analysis project.
Defines multiple datasets and their specific parameters.
"""
import os

# --- File Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Define All Available Datasets ---
DATASETS = {
    "10_30": {
        "name": "Data 10% and 30%",
        "data_dir": os.path.join(BASE_DIR, 'data_10_30'),
        "intervals": [
            665, 984, 1319, 1643, 1965, 2282, 2648, 2970, 3298, 3640,
            3970, 4296, 4631, 4956, 5292, 5618, 5948, 6276, 6608, 6945
        ],
        "concentrations": (10, 30) # Tuple of concentrations for cycles 1-5 and 6-10
    },
    "50_70": {
        "name": "Data 50% and 70%",
        "data_dir": os.path.join(BASE_DIR, 'data_50_70'),
        "intervals": [
            653, 977, 1315, 1646, 1972, 2265, 2637, 2960, 3296, 3632,
            3960, 4267, 4615, 4932, 5291, 5598, 5929, 6278, 6591, 6916
        ],
        "concentrations": (50, 70)
    },
    "100": {
        "name": "Data 100%",
        "data_dir": os.path.join(BASE_DIR, 'data_100'),
        "intervals": [
            669, 998, 1316, 1658, 1978, 2321, 2648, 2982, 3313, 3642
        ],
        "concentrations": (100,)  # Only one concentration for all cycles
    }
}

# --- Sensor Array Configuration ---
NUM_ROWS = 7
NUM_COLS = 9
