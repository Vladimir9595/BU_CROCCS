"""
Defines the 1D Convolutional Neural Network (CNN) architecture.
"""
from torch import nn

class Enhanced1DCNN(nn.Module):
    """
    An enhanced 1D CNN with more layers and batch normalization for better performance.
    """
    def __init__(self, num_features: int, num_classes: int):
        super().__init__()
        # Deeper convolutional base to learn more complex features
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.flatten = nn.Flatten()

        # Calculate the flattened size dynamically
        # Input length is divided by 2 four times -> // 16
        flattened_size = (num_features // 16) * 256

        self.fc_block = nn.Sequential(
            nn.Linear(flattened_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.flatten(x)
        x = self.fc_block(x)
        return x

class SensorArrayCNN(nn.Module):
    """
    A 2D CNN designed to process the entire 63-sensor array as an 'image'.
    The dimensions are (channels, height, width) = (1, 63 sensors, 359 time points).
    """
    def __init__(self, num_sensors: int, num_timesteps: int, num_classes: int):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 7), padding=(1, 3)),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 5), padding=(1, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        self.flatten = nn.Flatten()

        # Calculate flattened size dynamically
        h = num_sensors // 8  # 63 -> 31 -> 15 -> 7
        w = num_timesteps // 8 # 359 -> 179 -> 89 -> 44
        flattened_size = h * w * 64

        self.fc_block = nn.Sequential(
            nn.Linear(flattened_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.flatten(x)
        x = self.fc_block(x)
        return x