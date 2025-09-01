"""
Main script to launch the VOC sensor data analysis tool.
Creates a single main window with a resizable, embedded console log and manages
the application flow: Gas -> Dataset -> Analyzer.
"""
import tkinter as tk
from tkinter import scrolledtext
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.widgets import Button

from src.config import EXPERIMENTS
from src.data_loader import SensorData
from src.interactive_plotter import InteractiveDraggableAnalyzer
from src.logger import LOGGER

matplotlib.use('TkAgg')

class AppController:
    """
    Manages the main Tkinter window and switches between views.
    Uses a PanedWindow to create a resizable layout.
    """
    def __init__(self, root, experiments_config):
        self.root = root
        self.root.title("VOC Sensor Analysis Tool")
        self.root.geometry("1400x900")
        self.experiments = experiments_config
        self.current_view_frame = None

        # Create a vertical paned window
        paned_window = tk.PanedWindow(root, orient=tk.VERTICAL, sashwidth=8, sashrelief=tk.RAISED)
        paned_window.pack(fill=tk.BOTH, expand=True)

        # Top frame for main content (will be added to the paned window)
        self.content_frame = tk.Frame(paned_window)
        paned_window.add(self.content_frame, minsize=600)

        # Bottom frame for the logger (will also be added to the paned window)
        log_frame = tk.Frame(paned_window, height=250)
        paned_window.add(log_frame, minsize=100)

        # Setup and start the logger in the bottom frame
        log_widget = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, font=("Courier New", 9))
        log_widget.pack(expand=True, fill='both')
        LOGGER.set_widget(log_widget)

        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _clear_content_frame(self):
        """Destroys all widgets in the content frame to prepare for a new view."""
        # This check is important because the old matplotlib canvas needs to be destroyed
        if self.current_view_frame:
            for widget in self.current_view_frame.winfo_children():
                widget.destroy()
        else: # First time setup
            self.current_view_frame = tk.Frame(self.content_frame)
            self.current_view_frame.pack(fill=tk.BOTH, expand=True)

        # Clear any old matplotlib figures that might be lingering
        plt.close('all')

    def launch_gas_selector(self):
        """Displays the initial gas selection view."""
        self._clear_content_frame()

        # New figure and canvas for the buttons
        fig = Figure(figsize=(5, 4))
        canvas = FigureCanvasTkAgg(fig, master=self.current_view_frame)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        ax = fig.add_subplot(111)

        ax.set_title("Choose the gas", fontsize=16, pad=20)
        ax.axis('off')

        labels_and_callbacks = [
            (config['name'], lambda event, k=key: self.launch_dataset_selector(k))
            for key, config in self.experiments.items()
        ]

        button_height, spacing, num_buttons = 0.12, 0.05, len(labels_and_callbacks)
        total_height = num_buttons * button_height + (num_buttons - 1) * spacing
        start_y = (1 - total_height) / 2

        self.buttons = []
        for i, (label, callback) in enumerate(labels_and_callbacks):
            y_pos = start_y + (num_buttons - 1 - i) * (button_height + spacing)
            ax_button = fig.add_axes([0.15, y_pos, 0.7, button_height])
            button = Button(ax_button, label)
            button.on_clicked(callback)
            self.buttons.append(button)
        canvas.draw()

    def launch_dataset_selector(self, gas_key):
        """Displays the dataset selection view for the chosen gas."""
        self._clear_content_frame()
        gas_config = self.experiments[gas_key]

        fig = Figure(figsize=(5, 4))
        canvas = FigureCanvasTkAgg(fig, master=self.current_view_frame)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        ax = fig.add_subplot(111)

        ax.set_title(f"Choose Dataset for {gas_config['name']}", fontsize=16)
        ax.axis('off')

        labels_and_callbacks = [
            (config['name'], lambda event, k=key: self.launch_analyzer(gas_key, k))
            for key, config in gas_config['datasets'].items()
        ]

        if not labels_and_callbacks:
            print(f"No datasets defined for {gas_config['name']}. Please check config.py.")
            return

        button_height, spacing, num_buttons = 0.12, 0.05, len(labels_and_callbacks)
        total_height = num_buttons * button_height + (num_buttons - 1) * spacing
        start_y = (1 - total_height) / 2

        self.buttons = []
        for i, (label, callback) in enumerate(labels_and_callbacks):
            y_pos = start_y + (num_buttons - 1 - i) * (button_height + spacing)
            ax_button = fig.add_axes([0.15, y_pos, 0.7, button_height])
            button = Button(ax_button, label)
            button.on_clicked(callback)
            self.buttons.append(button)
        canvas.draw()

    def launch_analyzer(self, gas_key, dataset_key):
        """Loads data and launches the main matplotlib analyzer view."""
        self._clear_content_frame()
        gas_config = self.experiments[gas_key]
        dataset_config = gas_config['datasets'][dataset_key]
        print(f"\n--- Loading Gas: {gas_config['name']}, Dataset: {dataset_config['name']} ---")
        try:
            sensor_data = SensorData(data_dir=dataset_config['data_dir'], exposure_intervals=dataset_config['intervals'])
            # The analyzer class now takes the parent frame as its first argument
            InteractiveDraggableAnalyzer(
                parent_frame=self.current_view_frame,
                app_controller=self,
                time_vector=sensor_data.time_vector,
                all_sensor_data_rows=sensor_data.summary_data,
                dataset_config=dataset_config,
                gas_name=gas_key,
                signal_name='Summary Luminance'
            )
        except (IOError, ValueError) as e:
            print(f"\nAn error occurred while loading this dataset: {e}")

    def _on_closing(self):
        """Handles the main window closing event."""
        print("\n--- Main window closed. Application shutting down. ---")
        LOGGER.stop()
        self.root.quit()
        self.root.destroy()

def main():
    """Main function to run the application."""
    root = tk.Tk()
    app = AppController(root, EXPERIMENTS)
    LOGGER.start() # Start logger after the GUI is built
    try:
        print("--- Application Started ---")
        app.launch_gas_selector()
        root.mainloop()
    finally:
        LOGGER.stop()

if __name__ == "__main__":
    main()