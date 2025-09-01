"""
Main script to launch the VOC sensor data analysis tool.
Provides a cascading selection menu: Gas -> Dataset -> Analyzer.
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from src.config import EXPERIMENTS
from src.data_loader import SensorData
from src.interactive_plotter import InteractiveDraggableAnalyzer
from src.logger import LOGGER

matplotlib.use('TkAgg')

class AppController:
    """
    Manages the application flow, switching between the Gas selector,
    Dataset selector, and the main Analyzer window.
    """
    def __init__(self, experiments_config):
        self.experiments = experiments_config
        self.fig = None
        self.ax = None
        self.buttons = []

    def _create_button_window(self, title, labels_and_callbacks):
        """Generic function to create a window with a list of buttons."""
        plt.close('all')
        self.fig, self.ax = plt.subplots(figsize=(5, max(3, len(labels_and_callbacks) * 0.8)))
        self.fig.canvas.manager.set_window_title("Experiment Selector")
        self.ax.set_title(title, fontsize=16)
        self.ax.axis('off')

        button_height = 0.12
        spacing = 0.05
        num_buttons = len(labels_and_callbacks)
        total_height = num_buttons * button_height + (num_buttons - 1) * spacing
        start_y = (1 - total_height) / 2

        self.buttons.clear()
        for i, (label, callback) in enumerate(labels_and_callbacks):
            y_pos = start_y + (num_buttons - 1 - i) * (button_height + spacing)
            ax_button = self.fig.add_axes((0.15, y_pos, 0.7, button_height))
            button = Button(ax_button, label)
            button.on_clicked(callback)
            self.buttons.append(button)
        self.fig.show()

    def launch_gas_selector(self, event=None):
        """Creates the initial window for selecting a gas."""
        labels_and_callbacks = [
            (config['name'], lambda event, k=key: self.launch_dataset_selector(k))
            for key, config in self.experiments.items()
        ]
        self._create_button_window("Choose the gas", labels_and_callbacks)

    def launch_dataset_selector(self, gas_key):
        """Creates the window for selecting a dataset for the chosen gas."""
        gas_config = self.experiments[gas_key]

        labels_and_callbacks = [
            (config['name'], lambda event, k=key: self.launch_analyzer(gas_key, k))
            for key, config in gas_config['datasets'].items()
        ]

        if not labels_and_callbacks:
            print(f"No datasets defined for {gas_config['name']}. Please check config.py.")
            return

        self._create_button_window(f"Choose Dataset for {gas_config['name']}", labels_and_callbacks)

    def launch_analyzer(self, gas_key, dataset_key):
        """Loads data and launches the main analyzer window."""
        plt.close(self.fig)
        gas_config = self.experiments[gas_key]
        dataset_config = gas_config['datasets'][dataset_key]

        print(f"\n--- Loading Gas: {gas_config['name']}, Dataset: {dataset_config['name']} ---")

        try:
            sensor_data = SensorData(
                data_dir=dataset_config['data_dir'],
                exposure_intervals=dataset_config['intervals'])
            analyzer = InteractiveDraggableAnalyzer(
                time_vector=sensor_data.time_vector,
                all_sensor_data_rows=sensor_data.summary_data,
                dataset_config=dataset_config,
                gas_name=gas_key,
                app_controller=self,
                signal_name='Summary Luminance'
            )
            analyzer.show()
        except (IOError, ValueError) as e:
            print(f"\nAn error occurred while loading this dataset: {e}")

def main():
    """Main function to run the application."""
    LOGGER.start()
    try:
        print("--- Application Started ---")
        controller = AppController(EXPERIMENTS)
        controller.launch_gas_selector()
        plt.show()
    finally:
        print("\n--- Analysis complete. Closing application. ---")
        LOGGER.stop()
        LOGGER.destroy()

if __name__ == "__main__":
    main()
