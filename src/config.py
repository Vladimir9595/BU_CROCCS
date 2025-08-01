"""
Central configuration file for the VOC sensor analysis project.
"""
import os

# --- File Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

RED_FILE = os.path.join(DATA_DIR, 'TestRed.csv')
GREEN_FILE = os.path.join(DATA_DIR, 'TestGreen.csv')
BLUE_FILE = os.path.join(DATA_DIR, 'TestBlue.csv')

# --- Experiment Parameters ---
# The start and end times for each exposure cycle
EXPOSURE_INTERVALS = [
    665, 984, 1319, 1643, 1965, 2282, 2648, 2970, 3298, 3640,
    3970, 4296, 4631, 4956, 5292, 5618, 5948, 6276, 6608, 6945
]

# --- Sensor Array Configuration ---
NUM_ROWS = 7
NUM_COLS = 9
