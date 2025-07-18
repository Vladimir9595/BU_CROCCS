import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def load_and_reshape_data(filepath, num_rows=7, num_cols=9):
    """
    Loads sensor data from a CSV file and reshapes it by sensor row.
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
    Combines RGB data into a single summary signal using the Luminance formula.
    """
    summary_rows = []
    for r_row, g_row, b_row in zip(red_rows, green_rows, blue_rows):
        luminance_row = 0.299 * r_row + 0.587 * g_row + 0.114 * b_row
        summary_rows.append(luminance_row)
    return summary_rows

def plot_sensor_row_data(time_vector, sensor_data_rows, row_to_highlight, signal_name="Signal"):
    """
    Generates a dashboard plot showing the sensor board and the corresponding
    signal traces for a specific row.
    """
    if not (0 <= row_to_highlight < len(sensor_data_rows)):
        print(f"Error: row_to_highlight must be between 0 and {len(sensor_data_rows)-1}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={'width_ratios': [1, 3]})
    fig.suptitle(f'Visualizing Sensor Array - Row C{row_to_highlight + 1}', fontsize=16)

    # --- Left Panel: The Sensor Board ---
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

    # --- Right Panel: The Signal Plot ---
    ax_signal = axes[1]
    signals_to_plot = sensor_data_rows[row_to_highlight]

    colors = plt.get_cmap('tab10').colors
    styles = ['-', '--', ':', '-.']

    for i in range(signals_to_plot.shape[1]):
        color = colors[i % len(colors)]
        style = styles[i % len(styles)]
        ax_signal.plot(time_vector, signals_to_plot[:, i],
                       label=f'A{i+1}', color=color, linestyle=style, linewidth=2)

    # --- Programmatically create shaded regions ---
    dp_points = [
        665, 984, 1319, 1643, 1965, 2282, 2648, 2970, 3298, 3640,
        3970, 4296, 4631, 4956, 5292, 5618, 5948, 6276, 6608, 6945
    ]

    for i in range(0, len(dp_points), 2):
        start_time = dp_points[i]
        end_time = dp_points[i+1]
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

    # --- CORRECTED LINE ---
    # Ensure the x-axis limit is wide enough to show the last shaded bar
    plot_end_time = max(max(time_vector), dp_points[-1])
    ax_signal.set_xlim(0, plot_end_time + 50) # Add 50 for padding

    if signal_name == "G-Level":
        ax_signal.set_ylim(110, 195)

    plt.tight_layout(rect=[0, 0, 1, 0.96])


if __name__ == '__main__':
    # --- Configuration ---
    RED_FILE = 'TestRed.csv'
    GREEN_FILE = 'TestGreen.csv'
    BLUE_FILE = 'TestBlue.csv'

# --- Load All RGB Data ---
    print("Loading all RGB data...")
    time_r, red_data = load_and_reshape_data(RED_FILE)
    time_g, green_data = load_and_reshape_data(GREEN_FILE)
    time_b, blue_data = load_and_reshape_data(BLUE_FILE)

    if all(data is not None for data in [time_r, red_data, green_data, blue_data]):
        # --- FIX: Synchronize data lengths ---
        min_length = min(len(time_r), len(time_g), len(time_b))
        print(f"Data length mismatch found. Trimming all data to the shortest length: {min_length}")

        time_vector_trimmed = time_g[:min_length]

        red_data_trimmed = [row[:min_length, :] for row in red_data]
        green_data_trimmed = [row[:min_length, :] for row in green_data]
        blue_data_trimmed = [row[:min_length, :] for row in blue_data]

        # --- Calculate the Summary Signal using the trimmed data ---
        print("Calculating summary (Luminance) signal...")
        summary_data = calculate_summary_signal(red_data_trimmed, green_data_trimmed, blue_data_trimmed)

        # --- Generate Plots using the Summary Signal ---
        print("Generating plots...")
        plot_sensor_row_data(time_vector_trimmed, summary_data, row_to_highlight=0, signal_name="Summary Luminance")

        plt.show()