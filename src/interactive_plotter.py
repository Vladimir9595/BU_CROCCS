"""
Contains the InteractiveDraggableAnalyzer class for creating an interactive GUI
to dynamically select and analyze sensor data intervals by dragging.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons

class InteractiveDraggableAnalyzer:
    """
    A class for an interactive plot where users can click and drag to define
    a custom analysis window and calculate the signal change (ΔI) within it.
    """
    def __init__(self, time_vector, all_sensor_data_rows, exposure_intervals, signal_name="Signal"):
        self.time = time_vector
        self.all_signals = all_sensor_data_rows
        self.intervals = exposure_intervals
        self.signal_name = signal_name

        self.selected_row_idx = 0
        self.selected_sensor_idx = 0
        self.active_row_signals = self.all_signals[self.selected_row_idx]

        # State management for dragging
        self.is_dragging = False
        self.start_point_data = None
        self.analysis_elements = []

        self.fig = plt.figure(figsize=(18, 8))
        self.fig.canvas.manager.set_window_title('Interactive Draggable Analyzer')
        gs = self.fig.add_gridspec(1, 2, width_ratios=[5, 1])
        self.ax = self.fig.add_subplot(gs[0, 0])
        self.ax_legend = self.fig.add_subplot(gs[0, 1])
        self.ax_legend.axis('off')

        plt.subplots_adjust(left=0.25)

        self._create_widgets()
        self._redraw_plot()
        # Connect the new set of event handlers for dragging
        self.fig.canvas.mpl_connect('button_press_event', self._on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self._on_release)

        print("\n--- Interactive Draggable Analyzer Ready ---")
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
                         label=f'A{i+1}', color=colors[i % len(colors)],
                         linestyle=styles[i % len(styles)], alpha=0.4, zorder=1)

        handles, labels = self.ax.get_legend_handles_labels()
        self.ax_legend.legend(handles, labels, loc='center left', fontsize='large')

        for i in range(0, len(self.intervals), 2):
            color = 'pink' if i < 10 else 'lightblue'
            self.ax.axvspan(
                self.intervals[i],
                self.intervals[i+1],
                color=color,
                alpha=0.2,
                zorder=0
            )

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
        self.selected_row_idx = int(label.split('C')[1]) - 1
        self.active_row_signals = self.all_signals[self.selected_row_idx]
        print(f"\n--- Switched to Row C{self.selected_row_idx + 1} ---")
        self._redraw_plot()
        self._print_instructions()

    def _on_sensor_change(self, label):
        self.selected_sensor_idx = int(label.split('A')[1]) - 1
        print(f"\nActive Sensor changed to: A{self.selected_sensor_idx + 1}")
        self._print_instructions()

    def _print_instructions(self):
        print("Click and drag on the plot to define a custom analysis window.")

    def _clear_previous_analysis(self):
        for element in self.analysis_elements:
            element.remove()
        self.analysis_elements.clear()

    def _on_press(self, event):
        """Handles the mouse button press: starts the dragging process."""
        if event.inaxes != self.ax:
            return
        self.is_dragging = True
        self._clear_previous_analysis()

        start_time = event.xdata
        start_idx = np.searchsorted(self.time, start_time)

        if start_idx >= len(self.time):
            return

        signal_trace = self.active_row_signals[:, self.selected_sensor_idx]
        start_value = signal_trace[start_idx]
        self.start_point_data = (self.time[start_idx], start_value)

        color = plt.get_cmap('tab10').colors[self.selected_sensor_idx]
        marker1, = self.ax.plot(self.start_point_data[0], self.start_point_data[1], 'o',
                                ms=8, color=color, mec='black', zorder=10)
        self.analysis_elements.append(marker1)
        self.fig.canvas.draw()

    def _on_motion(self, event):
        """Handles mouse movement while dragging: provides a live preview."""
        if not self.is_dragging or event.inaxes != self.ax:
            return

        # Remove old preview elements (bar and end marker)
        if len(self.analysis_elements) > 1:
            for element in self.analysis_elements[1:]:
                element.remove()
            self.analysis_elements = self.analysis_elements[:1]

        end_time = event.xdata
        end_idx = np.searchsorted(self.time, end_time)
        if end_idx >= len(self.time):
            end_idx = len(self.time) - 1

        signal_trace = self.active_row_signals[:, self.selected_sensor_idx]
        end_value = signal_trace[end_idx]
        color = plt.get_cmap('tab10').colors[self.selected_sensor_idx]

        bar = self.ax.axvspan(self.start_point_data[0], self.time[end_idx],
                              color='yellow', alpha=0.4, zorder=3)
        marker2, = self.ax.plot(self.time[end_idx], end_value, 'x',
                                ms=10, mew=2, color=color, mec='black', zorder=10)

        self.analysis_elements.extend([bar, marker2])
        self.fig.canvas.draw()

    def _on_release(self, event):
        """
        Handles mouse button release: finalizes the analysis for ALL sensors in the
        selected window and prints their results.
        """
        if not self.is_dragging or event.inaxes != self.ax:
            self.is_dragging = False
            return

        self.is_dragging = False
        end_time = event.xdata

        # Get start and end indices from the drag action
        start_idx = np.searchsorted(self.time, self.start_point_data[0])
        end_idx = np.searchsorted(self.time, end_time)
        if end_idx >= len(self.time): end_idx = len(self.time) - 1

        # Loop through all 9 sensors to get their data
        print(f"\n--- Final Analysis for Row C{self.selected_row_idx+1} ---")
        print(f"--- Time Window: {self.time[start_idx]:.2f}s to {self.time[end_idx]:.2f}s ---")

        for sensor_idx in range(self.active_row_signals.shape[1]):
            signal_trace = self.active_row_signals[:, sensor_idx]

            # Get I1 and I2 for THIS sensor
            i1_value = signal_trace[start_idx]
            i2_value = signal_trace[end_idx]
            delta_i = i2_value - i1_value

            print(f"  Sensor A{sensor_idx + 1}:  "
                  f"I1={i1_value:<7.2f} | "
                  f"I2={i2_value:<7.2f} | "
                  f"ΔI = {delta_i: .2f}")

        self._print_instructions()

    def show(self):
        """Displays the plot."""
        plt.show()