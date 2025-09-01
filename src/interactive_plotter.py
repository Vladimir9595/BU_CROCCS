"""
Contains the InteractiveDraggableAnalyzer class which acts as a "view"
to be embedded within a larger Tkinter application window.
"""
import os
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import RadioButtons, Button

class InteractiveDraggableAnalyzer:
    """
    This class now manages the PLOTTING VIEW within a main application window.
    """
    def __init__(self, parent_frame, app_controller, time_vector, all_sensor_data_rows, dataset_config, gas_name, signal_name="Signal"):
        # Store references to the main components
        self.parent_frame = parent_frame
        self.app_controller = app_controller
        self.time = time_vector
        self.all_signals = all_sensor_data_rows
        self.config = dataset_config
        self.gas_name = gas_name
        self.intervals = self.config['intervals']
        self.concentrations = self.config['concentrations']
        self.signal_name = signal_name

        # State management
        self.selected_row_idx, self.selected_sensor_idx = 0, 0
        self.active_row_signals = self.all_signals[self.selected_row_idx]
        self.is_dragging, self.start_point_data = False, None
        self.analysis_elements = []

        # Create the Matplotlib Figure and embed it in the Tkinter frame
        self.fig = Figure(figsize=(18, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.parent_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Create layout within the figure
        gs = self.fig.add_gridspec(1, 2, width_ratios=[5, 1])
        self.ax = self.fig.add_subplot(gs[0, 0])
        self.ax_legend = self.fig.add_subplot(gs[0, 1])
        self.ax_legend.axis('off')

        self.fig.subplots_adjust(left=0.25, bottom=0.15)
        self._create_widgets()
        self._redraw_plot()

        # Connect events to the canvas
        self.canvas.mpl_connect('button_press_event', self._on_press)
        self.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.canvas.mpl_connect('button_release_event', self._on_release)
        print("\n--- Interactive Draggable Analyzer Ready ---")
        self._print_instructions()

    def _redraw_plot(self):
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
        num_cycles_per_conc = len(self.intervals) // (2 * len(self.concentrations))
        for i in range(0, len(self.intervals), 2):
            conc_index = i // (2 * num_cycles_per_conc)
            color = ['pink', 'lightblue', 'lightgreen', 'lightcoral'][conc_index % 4]
            self.ax.axvspan(self.intervals[i], self.intervals[i+1], color=color, alpha=0.2, zorder=0)
        self.ax.set_title(f"Analyze Row C{self.selected_row_idx + 1} ({self.signal_name})")
        self.ax.set_xlabel("Time (second)")
        self.ax.set_ylabel(f"{self.signal_name} (Intensity)")
        self.ax.grid(True, linestyle='--')
        plot_end_time = max(max(self.time), self.intervals[-1])
        self.ax.set_xlim(0, plot_end_time + 100)
        self.ax.set_ylim(95, 155)
        self.canvas.draw_idle()

    def _create_widgets(self):
        rax_rows = self.fig.add_axes((0.05, 0.6, 0.15, 0.3))
        self.radio_rows = RadioButtons(rax_rows, [f"Row C{i+1}" for i in range(7)], active=0)
        self.radio_rows.on_clicked(self._on_row_change)
        rax_sensors = self.fig.add_axes((0.05, 0.25, 0.15, 0.3))
        self.radio_sensors = RadioButtons(rax_sensors, [f"Sensor A{i+1}" for i in range(9)], active=0)
        self.radio_sensors.on_clicked(self._on_sensor_change)
        ax_extract = self.fig.add_axes((0.05, 0.1, 0.15, 0.05))
        self.button_extract = Button(ax_extract, 'Extract Cycles in Window')
        self.button_extract.on_clicked(self._extract_and_save_all_data)
        ax_back_dataset = self.fig.add_axes((0.8, 0.9, 0.15, 0.05))
        self.button_back_dataset = Button(ax_back_dataset, 'Back to concentration')
        self.button_back_dataset.on_clicked(self._go_back_to_dataset_choice)
        ax_back_gas = self.fig.add_axes((0.8, 0.82, 0.15, 0.05))
        self.button_back_gas = Button(ax_back_gas, 'Back to gas choice')
        self.button_back_gas.on_clicked(self._go_back_to_gas_choice)

    def _go_back_to_dataset_choice(self, event):
        """Closes this window and tells the controller to show the dataset selector."""
        print("\nNavigating back to dataset selection...")
        self.app_controller.launch_dataset_selector(self.gas_name)

    def _go_back_to_gas_choice(self, event):
        """Closes this window and tells the controller to show the gas selector."""
        print("\nNavigating back to gas selection...")
        self.app_controller.launch_gas_selector()

    def _on_row_change(self, label):
        """Handles changing the active sensor row."""
        self.selected_row_idx = int(label.split('C')[1]) - 1
        self.active_row_signals = self.all_signals[self.selected_row_idx]
        print(f"\n--- Switched to Row C{self.selected_row_idx + 1} ---")
        self._redraw_plot()
        self._print_instructions()

    def _on_sensor_change(self, label):
        """Handles changing the active sensor within the current row."""
        self.selected_sensor_idx = int(label.split('A')[1]) - 1
        print(f"\nActive Sensor changed to: A{self.selected_sensor_idx + 1}")
        self._print_instructions()

    def _print_instructions(self):
        print("Click and drag to analyze a window, then click 'Extract Cycles in Window' to save.")

    def _clear_previous_analysis(self):
        for element in self.analysis_elements:
            element.remove()
        self.analysis_elements.clear()

    def _on_press(self, event):
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
        self.canvas.draw()

    def _on_motion(self, event):
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
        self.canvas.draw()

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
            i1_value, i2_value = signal_trace[start_idx], signal_trace[end_idx]
            delta_i = i2_value - i1_value
            print(f"  Sensor A{sensor_idx + 1}:  I1={i1_value:<7.2f} | I2={i2_value:<7.2f} | ΔI = {delta_i: .2f}")
        self._print_instructions()

    def _extract_and_save_all_data(self, event):
        print("\n--- Starting Full Time-Series Extraction for Cycles in Window ---")
        if not self.start_point_data or len(self.analysis_elements) < 3:
            print("Extraction failed: Please define an analysis window by clicking and dragging first.")
            return
        user_start_time, user_end_time = self.start_point_data[0], self.analysis_elements[-1].get_xdata()[0]
        print(f"Filtering cycles within user-defined window: {user_start_time:.2f}s to {user_end_time:.2f}s")
        output_base_dir = os.path.join('extracted_data', self.gas_name)
        os.makedirs(output_base_dir, exist_ok=True)
        print(f"Data will be saved in the '{output_base_dir}' directory.")
        extracted_files_count = 0
        num_cycles = len(self.intervals) // 2
        num_concentrations = len(self.concentrations)
        num_cycles_per_conc = num_cycles // num_concentrations

        for cycle_idx in range(num_cycles):
            cycle_start_time, cycle_end_time = self.intervals[cycle_idx * 2], self.intervals[cycle_idx * 2 + 1]
            if cycle_start_time >= user_start_time and cycle_end_time <= user_end_time:
                cycle_num = cycle_idx + 1
                conc_index = cycle_idx // num_cycles_per_conc
                concentration = self.concentrations[conc_index]
                concentration_dir = os.path.join(output_base_dir, f"concentration_{concentration}")
                os.makedirs(concentration_dir, exist_ok=True)
                print(f"  -> Processing Cycle {cycle_num} (Concentration: {concentration}%)...")
                cycle_dir = os.path.join(concentration_dir, f"Cycle_{cycle_num:02d}")
                os.makedirs(cycle_dir, exist_ok=True)
                start_idx, end_idx = np.searchsorted(self.time, [cycle_start_time, cycle_end_time])
                if start_idx >= end_idx:
                    continue
                for row_idx in range(len(self.all_signals)):
                    current_row_data = self.all_signals[row_idx]
                    for sensor_idx in range(current_row_data.shape[1]):
                        time_segment, signal_segment = self.time[start_idx:end_idx], current_row_data[start_idx:end_idx, sensor_idx]
                        sensor_name = f"C{row_idx+1}_A{sensor_idx+1}"
                        output_filename = os.path.join(cycle_dir, f"{sensor_name}.csv")
                        np.savetxt(output_filename, np.vstack((time_segment, signal_segment)).T, delimiter=',', header='time,intensity', comments='')
                        extracted_files_count += 1

        if extracted_files_count > 0:
            print(f"\nSUCCESS: Extracted and saved {extracted_files_count} segments.")
        else: print("\nExtraction complete: No full cycles found in window.")