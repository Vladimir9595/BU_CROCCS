"""
Contains the SensorData class for loading, processing, and managing sensor data.
"""
import numpy as np

class SensorData:
    """A class to handle loading and processing of GMR sensor array data."""

    def __init__(self, red_path, green_path, blue_path, num_rows=7):
        """
        Initializes the SensorData object by loading and synchronizing data.
        """
        self.num_rows = num_rows
        self.time_vector = None
        self.red_data = None
        self.green_data = None
        self.blue_data = None
        self.summary_data = None

        self._load_and_process(red_path, green_path, blue_path)

    def _load_and_reshape(self, filepath):
        """Private method to load and reshape data from a single file."""
        try:
            raw_data = np.loadtxt(filepath, delimiter=',')
        except FileNotFoundError:
            print(f"Error: {filepath} not found.")
            return None, None

        time_vector = raw_data[:, 0]
        sensor_data = raw_data[:, 1:]
        reshaped = [sensor_data[:, i::self.num_rows] for i in range(self.num_rows)]
        return time_vector, reshaped

    def _load_and_process(self, red_path, green_path, blue_path):
        """
        Loads all RGB data, synchronizes, and calculates the summary signal.

        Args:
            red_path (str): Path to the red channel data file.
            green_path (str): Path to the green channel data file.
            blue_path (str): Path to the blue channel data file.
        """
        print("Loading all RGB data...")
        time_r, data_r = self._load_and_reshape(red_path)
        time_g, data_g = self._load_and_reshape(green_path)
        time_b, data_b = self._load_and_reshape(blue_path)

        if not all(d is not None for d in [data_r, data_g, data_b]):
            raise IOError("One or more data files could not be loaded.")

        # Synchronize data to the shortest length
        min_len = min(len(time_r), len(time_g), len(time_b))
        print(f"Data synchronized to shortest length: {min_len} points.")

        self.time_vector = time_g[:min_len]
        self.red_data = [row[:min_len, :] for row in data_r]
        self.green_data = [row[:min_len, :] for row in data_g]
        self.blue_data = [row[:min_len, :] for row in data_b]

        self._calculate_summary_signal()

    def _calculate_summary_signal(self):
        """Calculates the luminance summary signal."""
        print("Calculating summary (Luminance) signal...")
        self.summary_data = [
            0.299 * r + 0.587 * g + 0.114 * b
            for r, g, b in zip(self.red_data, self.green_data, self.blue_data)
        ]

    def get_data_for_row(self, row_idx, signal_type='summary'):
        """
        Returns the time vector and sensor data for a specific row.

        Args:
            row_idx (int): The index of the sensor row (0 for C1, etc.).
            signal_type (str): 'summary', 'green', 'red', or 'blue'.

        Returns:
            tuple: (time_vector, data_for_row)
        """
        if signal_type == 'summary':
            return self.time_vector, self.summary_data[row_idx]
        if signal_type == 'green':
            return self.time_vector, self.green_data[row_idx]
        if signal_type == 'red':
            return self.time_vector, self.red_data[row_idx]
        if signal_type == 'blue':
            return self.time_vector, self.blue_data[row_idx]

        raise ValueError("Invalid signal_type specified.")