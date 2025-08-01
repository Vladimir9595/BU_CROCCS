"""
Contains the InteractiveMultiRowAnalyzer class for creating an interactive GUI
to select a sensor row and analyze 300-second intervals of its data.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons

class InteractiveMultiRowAnalyzer:
    """
    A class to create a single interactive plot that allows selecting a sensor row (C1-C7)
    and then analyzing user-defined 300-second intervals on that row's data.
    """
    def __init__(self, time_vector, all_sensor_data_rows, exposure_intervals, signal_name="Signal"):
        self.time = time_vector
        self.all_signals = all_sensor_data_rows
        self.intervals = exposure_intervals
        self.signal_name = signal_name

        self.selected_row_idx = 0
        self.selected_sensor_idx = 0
        self.active_row_signals = self.all_signals[self.selected_row_idx]
        self.analysis_elements = []

        # Create a figure with a GridSpec for a dedicated legend area
        self.fig = plt.figure(figsize=(18, 8))
        self.fig.canvas.manager.set_window_title('Interactive Multi-Row Analyzer')
        # Create a grid: 1 row, 2 columns. Plot is 5x wider than the legend area.
        gs = self.fig.add_gridspec(1, 2, width_ratios=[5, 1])
        self.ax = self.fig.add_subplot(gs[0, 0])
        self.ax_legend = self.fig.add_subplot(gs[0, 1])
        self.ax_legend.axis('off')

        # Adjust subplot spacing to make room for widgets on the left
        plt.subplots_adjust(left=0.25)

        self._create_widgets()
        self._redraw_plot()
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)

        print("\n--- Interactive Multi-Row Analyzer Ready ---")
        self._print_instructions()

    def _redraw_plot(self):
        """Clears and redraws the entire plot for the currently selected row."""
        self._clear_previous_analysis()
        self.ax.clear()
        self.ax_legend.clear()
        self.ax_legend.axis('off')

        colors = plt.get_cmap('tab10').colors
        styles = ['-', '--', ':', '-.']

        for i in range(self.active_row_signals.shape[1]):
            self.ax.plot(self.time, self.active_row_signals[:, i],
                         label=f'A{i+1}',
                         color=colors[i % len(colors)],
                         linestyle=styles[i % len(styles)],
                         alpha=0.4,
                         zorder=1)

        # Handle legend placement
        handles, labels = self.ax.get_legend_handles_labels()
        # Place the legend in its dedicated axes
        self.ax_legend.legend(handles, labels, loc='center left', fontsize='large')

        for i in range(0, len(self.intervals), 2):
            color = 'pink' if i < 10 else 'lightblue'
            self.ax.axvspan(self.intervals[i], self.intervals[i+1], color=color, alpha=0.2, zorder=0)

        self.ax.set_title(f"Analyze Row C{self.selected_row_idx + 1} ({self.signal_name})")
        self.ax.set_xlabel("Time (second)")
        self.ax.set_ylabel(f"{self.signal_name} (Intensity)")
        self.ax.grid(True, linestyle='--')

        plot_end_time = max(max(self.time), self.intervals[-1])
        self.ax.set_xlim(0, plot_end_time + 100)
        self.ax.set_ylim(95, 155)
        self.fig.canvas.draw_idle()

    def _create_widgets(self):
        """Creates the radio buttons for row and sensor selection."""
        # Row Selector
        rax_rows = self.fig.add_axes((0.05, 0.6, 0.15, 0.3))
        row_labels = [f"Row C{i+1}" for i in range(7)]
        self.radio_rows = RadioButtons(rax_rows, row_labels, active=0)
        self.radio_rows.on_clicked(self._on_row_change)

        # Sensor Selector
        rax_sensors = self.fig.add_axes((0.05, 0.15, 0.15, 0.4))
        sensor_labels = [f"Sensor A{i+1}" for i in range(9)]
        self.radio_sensors = RadioButtons(rax_sensors, sensor_labels, active=0)
        self.radio_sensors.on_clicked(self._on_sensor_change)

    def _on_row_change(self, label):
        """Handles when the user selects a new row."""
        self.selected_row_idx = int(label.split('C')[1]) - 1
        self.active_row_signals = self.all_signals[self.selected_row_idx]
        print(f"\n--- Switched to Row C{self.selected_row_idx + 1} ---")
        self._redraw_plot()
        self._print_instructions()

    def _on_sensor_change(self, label):
        """Handles when the user selects a new sensor."""
        self.selected_sensor_idx = int(label.split('A')[1]) - 1
        print(f"\nActive Sensor changed to: A{self.selected_sensor_idx + 1}")
        self._print_instructions()

    def _print_instructions(self):
        print("Click on the plot to define the START of a 300-second analysis window.")

    def _clear_previous_analysis(self):
        """Removes the previous analysis bar and markers from the plot."""
        for element in self.analysis_elements:
            element.remove()
        self.analysis_elements.clear()

    def _on_click(self, event):
        """Handles a single click to define and analyze an interval."""
        if event.inaxes != self.ax:
            return

        self._clear_previous_analysis()
        start_time = event.xdata
        end_time = start_time + 300

        start_idx = np.searchsorted(self.time, start_time, side='left')
        end_idx = np.searchsorted(self.time, end_time, side='left')

        if end_idx >= len(self.time):
            print("Selected interval extends beyond data. Analysis truncated.")
            end_idx = len(self.time) - 1

        signal_trace = self.active_row_signals[:, self.selected_sensor_idx]
        color = plt.get_cmap('tab10').colors[self.selected_sensor_idx]

        i1_value, i2_value = signal_trace[start_idx], signal_trace[end_idx]
        delta_i = i2_value - i1_value

        print(f"\n--- Analysis for Row C{self.selected_row_idx+1} "
              f"| Sensor A{self.selected_sensor_idx + 1} ---")
        print(f"  Time Window: {self.time[start_idx]:.2f}s to {self.time[end_idx]:.2f}s")
        print(f"  I1 (start value): {i1_value:.2f}")
        print(f"  I2 (end value):   {i2_value:.2f}")
        print(f"  ΔI (Signal Change): {delta_i: .2f}")

        bar = self.ax.axvspan(
            self.time[start_idx],
            self.time[end_idx],
            color='yellow',
            alpha=0.4,
            zorder=3
        )
        marker1, = self.ax.plot(
            self.time[start_idx],
            i1_value,
            'o',
            ms=8,
            color=color,
            mec='black',
            zorder=10
        )
        marker2, = self.ax.plot(
            self.time[end_idx],
            i2_value,
            'x',
            ms=10,
            mew=2,
            color=color,
            mec='black',
            zorder=10
        )

        self.analysis_elements.extend([bar, marker1, marker2])
        self.fig.canvas.draw()

    def show(self):
        """Displays the plot."""
        plt.show()