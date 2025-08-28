"""
Main script to launch the interactive VOC sensor data analyzer.
This version allows for the analysis of user-defined draggable intervals
on any selected sensor row.
"""
import matplotlib

import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from src.config import DATASETS
from src.data_loader import SensorData
from src.interactive_plotter import InteractiveDraggableAnalyzer
from src.logger import LOGGER

matplotlib.use('TkAgg')

class DatasetLauncher:
    """A simple GUI to select which dataset to load and analyze."""
    def __init__(self, datasets_config):
        self.datasets = datasets_config
        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.fig.canvas.manager.set_window_title('Dataset Selector')
        self.ax.set_title("Select a Dataset to Analyze")
        self.ax.axis('off')

        self._create_buttons()

    def _create_buttons(self):
        """Creates a button for each dataset defined in the config."""
        button_height = 0.15
        spacing = 0.1
        num_buttons = len(self.datasets)
        total_height = num_buttons * button_height + (num_buttons - 1) * spacing
        start_y = (1 - total_height) / 2

        for i, (key, config) in enumerate(self.datasets.items()):
            y_pos = start_y + (num_buttons - 1 - i) * (button_height + spacing)
            ax_button = self.fig.add_axes((0.2, y_pos, 0.6, button_height))
            button = Button(ax_button, config['name'])
            # Use a lambda to capture the key for the callback
            button.on_clicked(lambda event, k=key: self.launch_analyzer(k))
            # Store button to prevent garbage collection
            if not hasattr(self, 'buttons'):
                self.buttons = []
            self.buttons.append(button)

    def launch_analyzer(self, dataset_key):
        """Callback function to load data and launch the main analyzer window."""
        print(f"\n--- Loading dataset: {self.datasets[dataset_key]['name']} ---")

        # Close the launcher window
        plt.close(self.fig)

        config = self.datasets[dataset_key]
        try:
            sensor_data = SensorData(
                data_dir=config['data_dir'],
                exposure_intervals=config['intervals']
            )

            analyzer = InteractiveDraggableAnalyzer(
                time_vector=sensor_data.time_vector,
                all_sensor_data_rows=sensor_data.summary_data,
                dataset_config=config,
                signal_name='Summary Luminance'
            )
            analyzer.show()

        except (IOError, ValueError) as e:
            print(f"\nAn error occurred while loading this dataset: {e}")

    def show(self):
        plt.show()

def main():
    """Main function to run the application."""
    LOGGER.start()
    try:
        print("--- Application Started ---")
        print("Please select a dataset to begin analysis.")
        launcher = DatasetLauncher(DATASETS)
        launcher.show()
    finally:
        # --- This block ensures cleanup happens no matter how the app closes ---
        print("\n--- Analysis complete. Closing application. ---")
        LOGGER.stop()
        LOGGER.destroy()

if __name__ == "__main__":
    main()