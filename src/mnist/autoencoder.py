import torch
import torch.nn as nn


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
            nn.Conv2d(1, 8, kernel_size=2, padding=0, stride=1), # b, 8, 27, 27
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2), # b, 8, 16, 16,
            nn.Conv2d(8, 16, kernel_size=2, padding=0, stride=1), # b, 16, 13, 13
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2), # b, 16, 6, 6
            nn.Conv2d(16, 20, kernel_size=3, padding=1, stride=2), # b, 20, 4, 4
            nn.ReLU(),
            nn.MaxPool2d(4, stride=4)) # b, 20, 1, 1 

        self.decoder = nn.Sequential(             
            nn.ConvTranspose2d(20, 16, kernel_size=4, stride=4),  # b, 20, 4, 4
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=5, stride=3, padding=0),  # b, 8, 14, 14
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=0),  # b, 1, 28, 28  
            nn.Sigmoid())

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return torch.squeeze(encoded, 3), decoded