import torch
import torch.nn as nn
import torch.nn.functional as F


class VAencoder(nn.Module):
    ''' This the encoder part of VAE

    '''
    def __init__(self, input_dim, hidden_dim, latent_dim):
        '''
        Args:
            input_dim: A integer indicating the size of input (in case of MNIST 28 * 28).
            hidden_dim: A integer indicating the size of hidden dimension.
            latent_dim: A integer indicating the latent size.
            n_classes: A integer indicating the number of classes. (dimension of one-hot representation of labels)
        '''
        super().__init__()

        self.linear = nn.Linear(input_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        # x is of shape [batch_size, input_dim + n_classes]

        hidden = F.relu(self.linear(x))
        # hidden is of shape [batch_size, hidden_dim]

        # latent parameters
        mean = self.mu(hidden)
        # mean is of shape [batch_size, latent_dim]
        log_var = self.var(hidden)
        # log_var is of shape [batch_size, latent_dim]

        return mean, log_var


class VAdecoder(nn.Module):
    ''' This the decoder part of VAE

    '''
    def __init__(self, latent_dim, hidden_dim, output_dim):
        '''
        Args:
            latent_dim: A integer indicating the latent size.
            hidden_dim: A integer indicating the size of hidden dimension.
            output_dim: A integer indicating the size of output (in case of MNIST 28 * 28).
            n_classes: A integer indicating the number of classes. (dimension of one-hot representation of labels)
        '''
        super().__init__()

        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.hidden_to_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x is of shape [batch_size, latent_dim + num_classes]
        x = F.relu(self.latent_to_hidden(x))
        # x is of shape [batch_size, hidden_dim]
        generated_x = F.sigmoid(self.hidden_to_out(x))
        # x is of shape [batch_size, output_dim]

        return generated_x


class VariationalAE(nn.Module):
    ''' This the VAE, which takes a encoder and decoder.

    '''
    def __init__(self, input_dim, hidden_dim, latent_dim):
        '''
        Args:
            input_dim: A integer indicating the size of input (in case of MNIST 28 * 28).
            hidden_dim: A integer indicating the size of hidden dimension.
            latent_dim: A integer indicating the latent size.
            n_classes: A integer indicating the number of classes. (dimension of one-hot representation of labels)
        '''
        super().__init__()

        self.encoder = VAencoder(input_dim, hidden_dim, latent_dim)
        self.decoder = VAdecoder(latent_dim, hidden_dim, input_dim)

    def forward(self, x):

        # encode
        z_mu, z_logvar = self.encoder(x)

        # sample from the distribution having latent parameters z_mu, z_var
        # reparameterize
        std = torch.exp(z_logvar / 2)
        eps = torch.randn_like(std)
        x_sample = z_mu + std * eps
        z = x_sample

        # decode
        generated_x = self.decoder(z)

        return generated_x, z_mu, z_logvar, z


class ConvVAE(nn.Module):
    def __init__(self, z_dim, image_channels=1, h_dim=128):
        super(ConvVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 16, kernel_size=3, padding=1,
                      stride=1),  # b, 16, 28, 28
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # b, 16, 14, 14,
            nn.Conv2d(16, 8, kernel_size=3, padding=1,
                      stride=1),  # b, 8, 14, 14
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # b, 8, 7, 7
            nn.Conv2d(8, 8, kernel_size=3, padding=1, stride=1),  # b, 8, 7, 7
            nn.ReLU(),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.MaxPool2d(2, stride=2),  # b, 8, 4, 4
            Flatten())

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(nb_filters=8, size=4),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),  # b, 8, 4, 4
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),  # b, 8, 8, 8
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),  # b, 8, 8, 8
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),  # b, 8, 16, 16
            nn.Conv2d(8, 16, kernel_size=3, stride=1,
                      padding=0),  # b, 16, 14, 14
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),  # b, 16, 28, 28
            nn.Conv2d(16, 1, kernel_size=3, stride=1,
                      padding=1),  # b, 1, 28, 28
            nn.Sigmoid())

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size())
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        generated_x = self.decode(z)
        return generated_x, mu, logvar, z


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
    def __init__(self, z_dim, image_channels=1, h_dim=1024):
        super(ConvLargeVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 16, kernel_size=3, padding=1,
                      stride=1),  # b, 16, 28, 28
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # b, 16, 14, 14,
            nn.Conv2d(16, 32, kernel_size=3, padding=1,
                      stride=1),  # b, 32, 14, 14
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # b, 32, 7, 7
            nn.Conv2d(32, 64, kernel_size=3, padding=1,
                      stride=1),  # b, 64, 7, 7
            nn.ReLU(),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.MaxPool2d(2, stride=2),  # b, 64, 4, 4
            Flatten())

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(nb_filters=64, size=4),
            nn.Conv2d(64, 64, kernel_size=3, stride=1,
                      padding=1),  # b, 64, 4, 4
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),  # b, 64, 8, 8
            nn.Conv2d(64, 32, kernel_size=3, stride=1,
                      padding=1),  # b, 64, 8, 8
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),  # b, 32, 16, 16
            nn.Conv2d(32, 16, kernel_size=3, stride=1,
                      padding=0),  # b, 16, 14, 14
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),  # b, 16, 28, 28
            nn.Conv2d(16, 1, kernel_size=3, stride=1,
                      padding=1),  # b, 1, 28, 28
            nn.Sigmoid())

    def reparameterize(self, mu, logvar, device):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size())
        std = std.to(device)
        esp = esp.to(device)
        z = mu + std * esp
        return z

    def bottleneck(self, h, device):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar, device)
        return z, mu, logvar

    def encode(self, x, device):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h, device)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x, device):
        z, mu, logvar = self.encode(x, device)
        generated_x = self.decode(z)
        return generated_x, mu, logvar, z

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

