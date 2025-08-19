"""
Contains the InteractiveDraggableAnalyzer class for creating an interactive GUI
to dynamically select and analyze sensor data intervals by dragging, and to
extract the full time-series data from all predefined exposure cycles
within that selected window.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Button

class InteractiveDraggableAnalyzer:
    """
    A class for an interactive plot where users can analyze data by dragging,
    and can also extract and save the ΔI from all predefined cycles
    that fall within the selected window.
    """
    def __init__(self, time_vector, all_sensor_data_rows, exposure_intervals, signal_name="Signal"):
        self.time = time_vector
        self.all_signals = all_sensor_data_rows
        self.intervals = exposure_intervals
        self.signal_name = signal_name

        self.selected_row_idx = 0
        self.selected_sensor_idx = 0
        self.active_row_signals = self.all_signals[self.selected_row_idx]

        self.is_dragging = False
        self.start_point_data = None
        self.analysis_elements = []

        self.fig = plt.figure(figsize=(18, 8))
        self.fig.canvas.manager.set_window_title('Interactive Draggable Analyzer')
        gs = self.fig.add_gridspec(1, 2, width_ratios=[5, 1])
        self.ax = self.fig.add_subplot(gs[0, 0])
        self.ax_legend = self.fig.add_subplot(gs[0, 1])
        self.ax_legend.axis('off')

        plt.subplots_adjust(left=0.25, bottom=0.15)

        self._create_widgets()
        self._redraw_plot()
        self.fig.canvas.mpl_connect('button_press_event', self._on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self._on_release)

        print("\n--- Interactive Draggable Analyzer Ready ---")
        self._print_instructions()

    def _redraw_plot(self):
        """Redraws the plot with the current active row signals."""
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

        ax_extract = self.fig.add_axes((0.05, 0.1, 0.15, 0.05))
        self.button_extract = Button(ax_extract, 'Extract Cycles in Window')
        self.button_extract.on_clicked(self._extract_and_save_all_data)

    def _on_row_change(self, label):
        """Handles row selection changes."""
        self.selected_row_idx = int(label.split('C')[1]) - 1
        self.active_row_signals = self.all_signals[self.selected_row_idx]
        print(f"\n--- Switched to Row C{self.selected_row_idx + 1} ---")
        self._redraw_plot()
        self._print_instructions()

    def _on_sensor_change(self, label):
        """Handles sensor selection changes."""
        self.selected_sensor_idx = int(label.split('A')[1]) - 1
        print(f"\nActive Sensor changed to: A{self.selected_sensor_idx + 1}")
        self._print_instructions()

    def _print_instructions(self):
        print("Click and drag to analyze a window, or click 'Extract Cycles in Window' to save.")

    def _clear_previous_analysis(self):
        for element in self.analysis_elements:
            element.remove()
        self.analysis_elements.clear()

    def _on_press(self, event):
        """Handles mouse press events to start dragging."""
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
        """Handles mouse motion events to update the analysis window."""
        if not self.is_dragging or event.inaxes != self.ax:
            return
        if len(self.analysis_elements) > 1:
            for element in self.analysis_elements[1:]:
                element.remove()
            self.analysis_elements = self.analysis_elements[:1]
        end_time = event.xdata
        end_idx = np.searchsorted(self.time, end_time)
        if end_idx >= len(self.time): end_idx = len(self.time) - 1
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
        if not self.is_dragging or event.inaxes != self.ax:
            self.is_dragging = False
            return
        self.is_dragging = False
        end_time = event.xdata
        start_idx = np.searchsorted(self.time, self.start_point_data[0])
        end_idx = np.searchsorted(self.time, end_time)
        if end_idx >= len(self.time): end_idx = len(self.time) - 1

        print(f"\n--- Visual Analysis for Row C{self.selected_row_idx+1} ---")
        print(f"--- Time Window: {self.time[start_idx]:.2f}s to {self.time[end_idx]:.2f}s ---")
        for sensor_idx in range(self.active_row_signals.shape[1]):
            signal_trace = self.active_row_signals[:, sensor_idx]
            i1_value = signal_trace[start_idx]
            i2_value = signal_trace[end_idx]
            delta_i = i2_value - i1_value
            print(f"  Sensor A{sensor_idx + 1}:  "
                  f"I1={i1_value:<7.2f} | "
                  f"I2={i2_value:<7.2f} | "
                  f"ΔI = {delta_i: .2f}")
        self._print_instructions()

    def _extract_and_save_all_data(self, event):
        """
        Extracts the full time-series data segment for all 63 sensors for each
        cycle contained within the user-defined window. Saves each segment to its
        own CSV file in a structured directory format.
        """
        print("\n--- Starting Full Time-Series Extraction for Cycles in Window ---")

        if not self.start_point_data or len(self.analysis_elements) < 3:
            print("Extraction failed: Please define an analysis window by clicking and dragging first.")
            return

        user_start_time = self.start_point_data[0]
        user_end_time = self.analysis_elements[-1].get_xdata()[0]

        print(f"Filtering cycles within user-defined window: {user_start_time:.2f}s to {user_end_time:.2f}s")

        output_base_dir = 'extracted_data'
        os.makedirs(output_base_dir, exist_ok=True)
        print(f"Data will be saved in the '{output_base_dir}' directory.")

        extracted_files_count = 0

        for cycle_idx in range(len(self.intervals) // 2):
            cycle_start_time = self.intervals[cycle_idx * 2]
            cycle_end_time = self.intervals[cycle_idx * 2 + 1]

            if cycle_start_time >= user_start_time and cycle_end_time <= user_end_time:
                cycle_num = cycle_idx + 1
                print(f"  -> Processing Cycle {cycle_num}...")

                cycle_dir = os.path.join(output_base_dir, f"Cycle_{cycle_num:02d}")
                os.makedirs(cycle_dir, exist_ok=True)

                start_idx = np.searchsorted(self.time, cycle_start_time)
                end_idx = np.searchsorted(self.time, cycle_end_time)

                if start_idx >= end_idx:
                    continue

                for row_idx in range(len(self.all_signals)):
                    current_row_data = self.all_signals[row_idx]
                    for sensor_idx in range(current_row_data.shape[1]):
                        time_segment = self.time[start_idx:end_idx]
                        signal_segment = current_row_data[start_idx:end_idx, sensor_idx]

                        sensor_name = f"C{row_idx+1}_A{sensor_idx+1}"
                        output_filename = os.path.join(cycle_dir, f"{sensor_name}.csv")

                        data_to_save = np.vstack((time_segment, signal_segment)).T

                        np.savetxt(output_filename, data_to_save, delimiter=',', header='time,intensity', comments='')
                        extracted_files_count += 1

        if extracted_files_count > 0:
            print(f"\nSUCCESS: Extracted and saved {extracted_files_count} individual sensor segments.")
        else:
            print("\nExtraction complete: No full cycles were found within your selected window.")

    def show(self):
        """Displays the plot."""
        plt.show()