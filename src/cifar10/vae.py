import torch
import torch.nn as nn
import torch.nn.functional as F


class UnFlatten(nn.Module):
    def __init__(self, nb_filters, size):
        super(UnFlatten, self).__init__()
        self.nb_filters = nb_filters
        self.size = size

    def forward(self, input):
        return input.view(input.size(0), self.nb_filters, self.size, self.size)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class ConvLargeVAE(nn.Module):
    def __init__(self, z_dim, image_channels=3, h_dim=4096):
        super(ConvLargeVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False), # b, 16, 32, 32
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False), # b, 32, 16, 16
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False), # b, 64, 16, 16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False), # b, 64, 8, 8
            nn.BatchNorm2d(64),
            nn.ReLU(),
            Flatten())

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc1_bn = nn.BatchNorm1d(z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc2_bn = nn.BatchNorm1d(z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        self.fc3_bn = nn.BatchNorm1d(h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(nb_filters=64, size=8), # b, 64, 8, 8
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False), # b, 64, 16, 16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1, output_padding=0, bias=False), # b, 32, 16, 16
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False), # b, 16, 32, 32
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1, bias=False), # b, 3, 32, 32
            nn.Sigmoid())
    
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size())
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1_bn(self.fc1(h)), self.fc2_bn(self.fc2(h))
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3_bn(self.fc3(z))
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        generated_x = self.decode(z)
        return generated_x, mu, logvar, z
    
    def save_weights(self, path):
        torch.save(self.state_dict(), path)

