# src/data_loader.py
"""
Contains the SensorData class for loading, processing, and managing sensor data.
This version pads all data to the required experiment end time to ensure a
complete time series visualization.
"""
import numpy as np

class SensorData:
    """A class to handle loading and processing of GMR sensor array data."""

    def __init__(self, red_path, green_path, blue_path, exposure_intervals, num_rows=7):
        """
        Initializes the SensorData object by loading and padding data.
        """
        self.num_rows = num_rows
        self.time_vector = None
        self.red_data = None
        self.green_data = None
        self.blue_data = None
        self.summary_data = None

        self._load_and_process(red_path, green_path, blue_path, exposure_intervals)

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

    def _pad_to_time(self, time_vec, data_rows, target_time_vec):
        """Pads a dataset to match the structure of the target time vector."""
        target_len = len(target_time_vec)
        current_len = len(time_vec)
        if current_len >= target_len:
            return [row[:target_len, :] for row in data_rows]

        print(f"Padding data from {current_len} to {target_len} points...")
        num_points_to_add = target_len - current_len

        # Pad the sensor data by repeating the last known value
        new_data_rows = []
        for row in data_rows:
            last_values = row[-1, :]
            padding = np.tile(last_values, (num_points_to_add, 1))
            padded_row = np.vstack([row, padding])
            new_data_rows.append(padded_row)

        return new_data_rows

    def _load_and_process(self, red_path, green_path, blue_path, exposure_intervals):
        """
        Loads all RGB data, pads them to the experiment end time, and calculates summary signal.

        Args:
            red_path (str): Path to the red channel data file.
            green_path (str): Path to the green channel data file.
            blue_path (str): Path to the blue channel data file.
            exposure_intervals (list): List of exposure intervals defining the experiment timeline.
        """
        print("Loading all RGB data...")
        time_r, data_r = self._load_and_reshape(red_path)
        time_g, data_g = self._load_and_reshape(green_path)
        time_b, data_b = self._load_and_reshape(blue_path)

        if not all(d is not None for d in [data_r, data_g, data_b]):
            raise IOError("One or more data files could not be loaded.")

        # Pad to experiment end time, not just longest file
        required_end_time = exposure_intervals[-1]

        # Find the longest of the actual recordings to use as our base
        all_times = [time_r, time_g, time_b]
        longest_time_idx = np.argmax([len(t) for t in all_times])
        base_time_vector = all_times[longest_time_idx]

        # Check if the longest recording already covers the experiment
        if base_time_vector[-1] < required_end_time:
            print(f"Extending time vector to required end time of {required_end_time}s...")
            # Extrapolate new time points
            avg_step = np.mean(np.diff(base_time_vector[-100:]))
            num_steps_to_add = int(np.ceil((required_end_time - base_time_vector[-1]) / avg_step)) + 1
            extra_time = np.arange(1, num_steps_to_add + 1) * avg_step + base_time_vector[-1]
            self.time_vector = np.concatenate([base_time_vector, extra_time])
        else:
            self.time_vector = base_time_vector

        self.red_data = self._pad_to_time(time_r, data_r, self.time_vector)
        self.green_data = self._pad_to_time(time_g, data_g, self.time_vector)
        self.blue_data = self._pad_to_time(time_b, data_b, self.time_vector)

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
            row_idx (int): The index of the row to retrieve data for (0-based).
            signal_type (str): The type of signal to retrieve ('summary', 'green')

        Returns:
            tuple: A tuple containing the time vector and the requested signal data for the specified row.
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