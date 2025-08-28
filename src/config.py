"""
Central configuration file for the VOC sensor analysis project.
Defines multiple datasets and their specific parameters.
"""
import os

# --- File Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAS_DIR = os.path.join(BASE_DIR, 'gas')

# --- Define All Experiments, Organized by Gas ---
EXPERIMENTS = {
    "ammonia": {
        "name": "Ammonia",
        "datasets": {
            "10_30": {
                "name": "Data 10% and 30%",
                "data_dir": os.path.join(GAS_DIR, 'ammonia', 'data_10_30'),
                "intervals": [665, 984, 1319, 1643, 1965, 2282, 2648, 2970, 3298, 3640,
                              3970, 4296, 4631, 4956, 5292, 5618, 5948, 6276, 6608, 6945],
                "concentrations": (10, 30)
            },
            "50_70": {
                "name": "Data 50% and 70%",
                "data_dir": os.path.join(GAS_DIR, 'ammonia', 'data_50_70'),
                "intervals": [653, 977, 1315, 1646, 1972, 2265, 2637, 2960, 3296, 3632,
                              3960, 4267, 4615, 4932, 5291, 5598, 5929, 6278, 6591, 6916],
                "concentrations": (50, 70)
            },
            "100": {
                "name": "Data 100%",
                "data_dir": os.path.join(GAS_DIR, 'ammonia', 'data_100'),
                "intervals": [669, 998, 1316, 1658, 1978, 2321, 2648, 2982, 3313, 3642],
                "concentrations": (100,)
            }
        }
    },
    "cadaverine": {
        "name": "Cadaverine",
        "datasets": {
            "10_30": {
                "name": "Data 10% and 30%",
                "data_dir": os.path.join(GAS_DIR, 'cadaverine', 'data_10_30'),
                "intervals": [730, 1055, 1380, 1705, 2030, 2355, 2680, 3005, 3330, 3655,
                              3980, 4305, 4641, 4956, 5302, 5632, 5929, 6296, 6624, 6950],
                "concentrations": (10, 30)
            },
            "50_70": {
                "name": "Data 50% and 70%",
                "data_dir": os.path.join(GAS_DIR, 'cadaverine', 'data_50_70'),
                "intervals": [658, 996, 1332, 1649, 1994, 2303, 2650, 2982, 3310, 3653,
                              3983, 4276, 4642, 4960, 5294, 5631, 5959, 6283, 6613, 6952],
                "concentrations": (50, 70)
            },
            "100": {
                "name": "Data 100%",
                "data_dir": os.path.join(GAS_DIR, 'cadaverine', 'data_100'),
                "intervals": [665, 983, 1331, 1658, 1993, 2315, 2646, 2990, 3308, 3653],
                "concentrations": (100,)
            }
        }
    },
    "methylamine": {
        "name": "Methylamine",
        "datasets": {
            "10_30": {
                "name": "Data 10% and 30%",
                "data_dir": os.path.join(GAS_DIR, 'methylamine', 'data_10_30'),
                "intervals": [678, 970, 1297, 1652, 1960, 2361, 2647, 3000, 3302, 3648,
                              3957, 4302, 4637, 4969, 5305, 5619, 5945, 6291, 6610, 6950],
                "concentrations": (10, 30)
            },
            "50_70": {
                "name": "Data 50% and 70%",
                "data_dir": os.path.join(GAS_DIR, 'methylamine', 'data_50_70'),
                "intervals": [667, 969, 1295, 1659, 1993, 2325, 2640, 2965, 3315, 3617,
                              3980, 4308, 4636, 4969, 5303, 5609, 5960, 6286, 6622, 6949],
                "concentrations": (50, 70)
            },
            "100": {
                "name": "Data 100%",
                "data_dir": os.path.join(GAS_DIR, 'methylamine', 'data_100'),
                "intervals": [666, 1003, 1330, 1664, 1978, 2329, 2655, 2991, 3314, 3652],
                "concentrations": (100,)
            }
        }
    },
    "putrescine": {
        "name": "Putrescine",
        "datasets": {
            "10_30": {
                "name": "Data 10% and 30%",
                "data_dir": os.path.join(GAS_DIR, 'putrescine', 'data_10_30'),
                "intervals": [674, 969, 1308, 1645, 1988, 2262, 2641, 2906, 3291, 3604,
                              3972, 4297, 4622, 4986, 5296, 5613, 5961, 6261, 6628, 6962],
                "concentrations": (10, 30)
            },
            "50_70": {
                "name": "Data 50% and 70%",
                "data_dir": os.path.join(GAS_DIR, 'putrescine', 'data_50_70'),
                "intervals": [678, 1018, 1347, 1684, 2019, 2347, 2678, 3005, 3343, 3673,
                              4004, 4336, 4668, 4986, 4668, 5662, 5992, 6271, 5992, 6271],
                "concentrations": (50, 70)
            },
            "100": {
                "name": "Data 100%",
                "data_dir": os.path.join(GAS_DIR, 'putrescine', 'data_100'),
                "intervals": [668, 1000, 1332, 1662, 1993, 2324, 2653, 2981, 3305, 3646],
                "concentrations": (100,)
            }
        }
    },
    "trimethylamine": {
        "name": "Trimethylamine",
        "datasets": {
            "10_30": {
                "name": "Data 10% and 30%",
                "data_dir": os.path.join(GAS_DIR, 'trimethylamine', 'data_10_30'),
                "intervals": [668, 1002, 1307, 1653, 1979, 2343, 2650, 2936, 3283, 3662,
                              3969, 4311, 4639, 4950, 5285, 5608, 5918, 6286, 6619, 6948],
                "concentrations": (10, 30)
            },
            "50_70": {
                "name": "Data 50% and 70%",
                "data_dir": os.path.join(GAS_DIR, 'trimethylamine', 'data_50_70'),
                "intervals": [658, 996, 1327, 1657, 1942, 2316, 2621, 2977, 3269, 3636,
                              3958, 4278, 4629, 4960, 5285, 5612, 5947, 6277, 6607, 6941],
                "concentrations": (50, 70)
            },
            "100": {
                "name": "Data 100%",
                "data_dir": os.path.join(GAS_DIR, 'trimethylamine', 'data_100'),
                "intervals": [667, 994, 1329, 1659, 1976, 2314, 2646, 2980, 3314, 3635],
                "concentrations": (100,)
            }
        }
    }
}

# --- Sensor Array Configuration ---
NUM_ROWS = 7
NUM_COLS = 9
