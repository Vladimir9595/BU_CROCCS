"""
This script visualizes the summary signal from a GMR sensor array,
processing RGB data from CSV files.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

def load_and_reshape_data(filepath, num_rows=7):
    """
    Load and reshape sensor data from a CSV file.
    The data is expected to have a time vector in the first column
    and sensor readings in later columns.
    Args:
        filepath (str): Path to the CSV file containing sensor data
        num_rows (int): Number of sensor rows in the array
    Returns:
        tuple: A tuple containing the time vector
        and a list of reshaped sensor data arrays.
    """
    try:
        raw_data = np.loadtxt(filepath, delimiter=',')
    except FileNotFoundError:
        print(f"Error: {filepath} not found. Please ensure the file is in the correct directory.")
        return None, None

    time_vector = raw_data[:, 0]
    sensor_data = raw_data[:, 1:]

    reshaped_data = []
    for i in range(num_rows):
        row_data = sensor_data[:, i::num_rows]
        reshaped_data.append(row_data)

    return time_vector, reshaped_data

def calculate_summary_signal(red_rows, green_rows, blue_rows):
    """
    Calculate the summary signal (luminance) from RGB sensor data.
    Args:
        red_rows (list): List of 2D arrays for red sensor data
        green_rows (list): List of 2D arrays for green sensor data
        blue_rows (list): List of 2D arrays for blue sensor data
    Returns:
        list: A list of 2D arrays containing the summary luminance signal
        for each sensor row.
    """
    summary_rows = []
    for r_row, g_row, b_row in zip(red_rows, green_rows, blue_rows):
        luminance_row = 0.299 * r_row + 0.587 * g_row + 0.114 * b_row
        summary_rows.append(luminance_row)
    return summary_rows

def pad_signals_to_time(time_vector, signal_rows, target_time, num_padding_points=50):
    """
    Pad the time vector and signal rows to ensure they cover the target time.
    Args:
        time_vector (np.ndarray): The original time vector
        signal_rows (list): List of 2D arrays containing sensor data
        target_time (float): The target time to pad the signals to
        num_padding_points (int): Number of points to pad at the end
    Returns:
        tuple: A tuple containing the padded time vector
        and the list of padded signal rows.
    """
    last_time = time_vector[-1]
    if last_time >= target_time:
        return time_vector, signal_rows  # No need to pad

    # Create equally spaced new time values from last_time to target_time
    padded_times = np.linspace(
        last_time,
        target_time,
        num_padding_points + 1)[1:]
    padded_time_vector = np.concatenate([time_vector, padded_times])

    padded_signal_rows = []
    for row in signal_rows:
        last_values = row[-1, :]

        padding = np.tile(last_values, (num_padding_points, 1))
        padded_row = np.vstack([row, padding])
        padded_signal_rows.append(padded_row)

    return padded_time_vector, padded_signal_rows

def plot_sensor_row_data(
        time_vector,
        sensor_data_rows,
        row_to_highlight,
        signal_name="Signal",
        exposure_intervals=None
):
    """
    Plot the sensor data for a specific row in the GMR sensor array.
    Args:
        time_vector (np.ndarray): The time vector for the sensor data
        sensor_data_rows (list): List of 2D arrays containing sensor data for each row
        row_to_highlight (int): The index of the row to highlight in the sensor array
        signal_name (str): Name of the signal being plotted
        exposure_intervals (list): List of exposure points for shading regions in the plot
    """
    if not 0 <= row_to_highlight < len(sensor_data_rows):
        print(f"Error: row_to_highlight must be between 0 and {len(sensor_data_rows)-1}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={'width_ratios': [1, 3]})
    fig.suptitle(f'Visualizing Sensor Array - Row C{row_to_highlight + 1}', fontsize=16)

    # Sensor board
    ax_board = axes[0]
    ax_board.imshow(np.ones((7, 9)), cmap='gray', vmin=0, vmax=1)
    ax_board.set_xticks(np.arange(9))
    ax_board.set_yticks(np.arange(7))
    ax_board.set_xticklabels([f'A{i+1}' for i in range(9)])
    ax_board.set_yticklabels([f'C{i+1}' for i in range(7)])
    for col in range(9):
        rect = patches.Rectangle((col - 0.5, row_to_highlight - 0.5), 1, 1,
                                 linewidth=3, edgecolor='lime', facecolor='lime', alpha=0.4)
        ax_board.add_patch(rect)
    ax_board.set_title('GMR Sensor Array')

    # Signal plot
    ax_signal = axes[1]
    signals_to_plot = sensor_data_rows[row_to_highlight]
    colors = plt.get_cmap('tab10').colors
    styles = ['-', '--', ':', '-.']

    for i in range(signals_to_plot.shape[1]):
        color = colors[i % len(colors)]
        style = styles[i % len(styles)]
        ax_signal.plot(time_vector, signals_to_plot[:, i],
                       label=f'A{i+1}', color=color, linestyle=style, linewidth=2)

    # Shaded exposure regions
    for i in range(0, len(exposure_intervals), 2):
        start_time = exposure_intervals[i]
        end_time = exposure_intervals[i+1]
        if i < 10:
            color = 'pink'
            label = '10% NH3 Exposure' if i == 0 else ''
        else:
            color = 'lightblue'
            label = '30% NH3 Exposure' if i == 10 else ''
        ax_signal.axvspan(start_time, end_time, color=color, alpha=0.3, zorder=0, label=label)

    ax_signal.set_title(f'{signal_name} Response for Row C{row_to_highlight + 1}')
    ax_signal.set_xlabel('Time (second)')
    ax_signal.set_ylabel(f'{signal_name} (Intensity)')
    ax_signal.legend(loc='upper right', bbox_to_anchor=(1.15, 1.02))
    ax_signal.grid(True, linestyle='--', alpha=0.6)
    ax_signal.set_xlim(0, exposure_intervals[-1] + 100)

    if signal_name == "G-Level":
        ax_signal.set_ylim(110, 195)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

if __name__ == '__main__':
    RED_FILE = 'TestRed.csv'
    GREEN_FILE = 'TestGreen.csv'
    BLUE_FILE = 'TestBlue.csv'

    dp_points = [
        665, 984, 1319, 1643, 1965, 2282, 2648, 2970, 3298, 3640,
        3970, 4296, 4631, 4956, 5292, 5618, 5948, 6276, 6608, 6945
    ]

    print("Loading all RGB data...")
    time_r, red_data = load_and_reshape_data(RED_FILE)
    time_g, green_data = load_and_reshape_data(GREEN_FILE)
    time_b, blue_data = load_and_reshape_data(BLUE_FILE)

    if all(data is not None for data in [time_r, red_data, green_data, blue_data]):
        REQUIRED_END_TIME = dp_points[-1]

        # Padding BEFORE trimming
        time_g, green_data = pad_signals_to_time(time_g, green_data, REQUIRED_END_TIME)
        time_r, red_data = pad_signals_to_time(time_r, red_data, REQUIRED_END_TIME)
        time_b, blue_data = pad_signals_to_time(time_b, blue_data, REQUIRED_END_TIME)

        # Find end index after padding
        end_index = np.searchsorted(time_g, REQUIRED_END_TIME, side='right')
        if end_index == len(time_g):
            print(f"✅ Time vector covers full exposure. "
                  f"Using up to index {end_index} (time: {time_g[-1]:.2f}s)")
        elif end_index < len(time_g):
            print(f"✅ Time vector covers full exposure. "
                  f"Using up to index {end_index} (time: {time_g[end_index-1]:.2f}s)")
        else:
            print(f"⚠️ Warning: Time vector ends before required exposure "
                  f"zone (max time: {time_g[-1]:.2f}s, required: {REQUIRED_END_TIME}s)")
            end_index = len(time_g)

        # Now trim safely
        time_vector_trimmed = time_g[:end_index]
        red_data_trimmed = [row[:end_index, :] for row in red_data]
        green_data_trimmed = [row[:end_index, :] for row in green_data]
        blue_data_trimmed = [row[:end_index, :] for row in blue_data]

        print("Calculating summary (Luminance) signal...")
        summary_data = calculate_summary_signal(
            red_data_trimmed,
            green_data_trimmed,
            blue_data_trimmed
        )

        # Pad signal if needed
        time_vector_trimmed, summary_data = pad_signals_to_time(
            time_vector_trimmed,
            summary_data,
            dp_points[-1]
        )

        print("Generating plots...")
        plot_sensor_row_data(time_vector_trimmed, summary_data, row_to_highlight=0,
                             signal_name="Summary Luminance", exposure_intervals=dp_points)

        plt.show()
