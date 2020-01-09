import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3),
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super(ConvAutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, stride=1), # b, 16, 28, 28
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2), # b, 16, 14, 14,
            nn.Conv2d(16, 8, kernel_size=3, padding=1, stride=1), # b, 8, 14, 14   
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2), # b, 8, 7, 7
            nn.Conv2d(8, 8, kernel_size=3, padding=1, stride=1), # b, 8, 7, 7
            nn.ReLU(),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.MaxPool2d(2, stride=2)) # b, 8, 4, 4

        self.decoder = nn.Sequential(             
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),  # b, 8, 4, 4
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2), # b, 8, 8, 8
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),  # b, 8, 8, 8
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2), # b, 8, 16, 16
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0),  # b, 16, 14, 14
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2), # b, 16, 28, 28
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),  # b, 1, 28, 28
            nn.Sigmoid())

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return torch.squeeze(encoded, 3), decoded


class ConvAutoEncoder2(nn.Module):
    def __init__(self):
        super(ConvAutoEncoder2, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
