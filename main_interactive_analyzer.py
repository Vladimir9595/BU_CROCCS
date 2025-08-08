"""
Main script to launch the interactive VOC sensor data analyzer.
This version allows for the analysis of user-defined draggable intervals
on any selected sensor row.
"""
import matplotlib

from src.config import RED_FILE, GREEN_FILE, BLUE_FILE, EXPOSURE_INTERVALS
from src.data_loader import SensorData
from src.interactive_plotter import InteractiveDraggableAnalyzer

matplotlib.use('TkAgg')

def main():
    """Main function to run the application."""
    try:
        # 1. Load and process all sensor data
        sensor_data = SensorData(
            RED_FILE,
            GREEN_FILE,
            BLUE_FILE,
            exposure_intervals=EXPOSURE_INTERVALS
        )

        # 2. Select which signal type to analyze
        signal_type_to_analyze = 'summary'

        if signal_type_to_analyze == 'summary':
            all_rows_data = sensor_data.summary_data
        elif signal_type_to_analyze == 'green':
            all_rows_data = sensor_data.green_data
        else:
            raise ValueError(f"Unknown signal type: {signal_type_to_analyze}")

        signal_name = {'summary': 'Summary Luminance', 'green': 'G-Level'}.get(signal_type_to_analyze)

        # 3. Launch the interactive draggable analyzer
        analyzer = InteractiveDraggableAnalyzer(
            time_vector=sensor_data.time_vector,
            all_sensor_data_rows=all_rows_data,
            exposure_intervals=EXPOSURE_INTERVALS,
            signal_name=signal_name
        )
        analyzer.show()

    except (IOError, ValueError) as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure your data files are present and correctly formatted.")

if __name__ == "__main__":
    main()