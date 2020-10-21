import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torch.hub import load_state_dict_from_url


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


class CarsConvVAE(nn.Module):
    def __init__(self, z_dim, image_channels=3, h_dim=6272):
        super(CarsConvVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False), # b, 16, 224, 224
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False), # b, 32, 112, 112
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False), # b, 64, 56, 56
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False), # b, 128, 28, 28
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False), # b, 256, 14, 14
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1, bias=False), # b, 128, 7, 7
            nn.BatchNorm2d(128),
            nn.ReLU(),
            Flatten())

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc1_bn = nn.BatchNorm1d(z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc2_bn = nn.BatchNorm1d(z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        self.fc3_bn = nn.BatchNorm1d(h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(nb_filters=128, size=7), # b, 128, 7, 7
            nn.ConvTranspose2d(128, 256, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False), # b, 256, 14, 14
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False), # b, 128, 28, 28
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False), # b, 64, 56, 56
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False), # b, 32, 112, 112
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False), # b, 16, 224, 224
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1, bias=False), # b, 3, 224, 224
            nn.Sigmoid())

    def reparameterize(self, mu, logvar, gpu):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size())
        if gpu:
            std = std.cuda()
            esp = esp.cuda()
        z = mu + std * esp
        return z

    def bottleneck(self, h, gpu):
        mu, logvar = self.fc1_bn(self.fc1(h)), self.fc2_bn(self.fc2(h))
        z = self.reparameterize(mu, logvar, gpu)
        return z, mu, logvar

    def encode(self, x, gpu):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h, gpu)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3_bn(self.fc3(z))
        z = self.decoder(z)
        return z

    def forward(self, x, gpu=False):
        z, mu, logvar = self.encode(x, gpu)
        generated_x = self.decode(z)
        return generated_x, mu, logvar, z

    def save_weights(self, path):
        torch.save(self.state_dict(), path)


class SmallCarsConvVAE(nn.Module):
    def __init__(self, z_dim, image_channels=3, h_dim=12544):
        super(SmallCarsConvVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False), # b, 16, 224, 224
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False), # b, 32, 112, 112
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False), # b, 64, 56, 56
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 16, kernel_size=3, stride=2, padding=1, bias=False), # b, 16, 28, 28
            nn.BatchNorm2d(16),
            nn.ReLU(),
            Flatten())

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc1_bn = nn.BatchNorm1d(z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc2_bn = nn.BatchNorm1d(z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        self.fc3_bn = nn.BatchNorm1d(h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(nb_filters=16, size=28), # b, 16, 28, 28
            nn.ConvTranspose2d(16, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False), # b, 64, 56, 56
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False), # b, 32, 112, 112
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False), # b, 16, 224, 224
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1, bias=False), # b, 3, 224, 224
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
        mu, logvar = self.fc1_bn(self.fc1(h)), self.fc2_bn(self.fc2(h))
        z = self.reparameterize(mu, logvar, device)
        return z, mu, logvar

    def encode(self, x, device):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h, device)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3_bn(self.fc3(z))
        z = self.decoder(z)
        return z

    def forward(self, x, device):
        z, mu, logvar = self.encode(x, device)
        generated_x = self.decode(z)
        return generated_x, mu, logvar, z

    def save_weights(self, path):
        torch.save(self.state_dict(), path)


class SmallCarsConvVAE128(nn.Module):
    def __init__(self, z_dim, image_channels=3, h_dim=131072):
        super(SmallCarsConvVAE128, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False), # b, 16, 128, 128
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 128, kernel_size=3, stride=2, padding=1, bias=False), # b, 32, 64, 64
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False), # b, 64, 32, 32
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False), # b, 16, 16, 16
            nn.BatchNorm2d(512),
            nn.ReLU(),
            Flatten())

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc1_bn = nn.BatchNorm1d(z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc2_bn = nn.BatchNorm1d(z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        self.fc3_bn = nn.BatchNorm1d(h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(nb_filters=512, size=16), # b, 512, 16, 16
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False), # b, 64, 56, 56
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False), # b, 32, 112, 112
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False), # b, 16, 224, 224
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1, bias=False), # b, 3, 224, 224
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
        mu, logvar = self.fc1_bn(self.fc1(h)), self.fc2_bn(self.fc2(h))
        z = self.reparameterize(mu, logvar, device)
        return z, mu, logvar

    def encode(self, x, device):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h, device)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3_bn(self.fc3(z))
        z = self.decoder(z)
        return z

    def forward(self, x, device):
        z, mu, logvar = self.encode(x, device)
        generated_x = self.decode(z)
        return generated_x, mu, logvar, z

    def save_weights(self, path):
        torch.save(self.state_dict(), path)


class VAE(nn.Module):
    def __init__(self, nc, ngf, ndf, latent_variable_size, device):
        super(VAE, self).__init__()

        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf
        self.latent_variable_size = latent_variable_size
        self.device = device

        # encoder
        self.e1 = nn.Conv2d(nc, ndf, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(ndf)

        self.e2 = nn.Conv2d(ndf, ndf*2, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(ndf*2)

        self.e3 = nn.Conv2d(ndf*2, ndf*4, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(ndf*4)

        self.e4 = nn.Conv2d(ndf*4, ndf*8, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(ndf*8)

        self.e5 = nn.Conv2d(ndf*8, ndf*8, 4, 2, 1)
        self.bn5 = nn.BatchNorm2d(ndf*8)

        self.fc1 = nn.Linear(ndf*8*4*4, latent_variable_size)
        self.fc2 = nn.Linear(ndf*8*4*4, latent_variable_size)

        # decoder
        self.d1 = nn.Linear(latent_variable_size, ngf*8*2*4*4)

        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd1 = nn.ReplicationPad2d(1)
        self.d2 = nn.Conv2d(ngf*8*2, ngf*8, 3, 1)
        self.bn6 = nn.BatchNorm2d(ngf*8, 1.e-3)

        self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd2 = nn.ReplicationPad2d(1)
        self.d3 = nn.Conv2d(ngf*8, ngf*4, 3, 1)
        self.bn7 = nn.BatchNorm2d(ngf*4, 1.e-3)

        self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd3 = nn.ReplicationPad2d(1)
        self.d4 = nn.Conv2d(ngf*4, ngf*2, 3, 1)
        self.bn8 = nn.BatchNorm2d(ngf*2, 1.e-3)

        self.up4 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd4 = nn.ReplicationPad2d(1)
        self.d5 = nn.Conv2d(ngf*2, ngf, 3, 1)
        self.bn9 = nn.BatchNorm2d(ngf, 1.e-3)

        self.up5 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd5 = nn.ReplicationPad2d(1)
        self.d6 = nn.Conv2d(ngf, nc, 3, 1)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.leakyrelu(self.bn1(self.e1(x)))
        h2 = self.leakyrelu(self.bn2(self.e2(h1)))
        h3 = self.leakyrelu(self.bn3(self.e3(h2)))
        h4 = self.leakyrelu(self.bn4(self.e4(h3)))
        h5 = self.leakyrelu(self.bn5(self.e5(h4)))
        h5 = h5.view(-1, self.ndf*8*4*4)

        return self.fc1(h5), self.fc2(h5)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(self.device)
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h1 = self.relu(self.d1(z))
        h1 = h1.view(-1, self.ngf*8*2, 4, 4)
        h2 = self.leakyrelu(self.bn6(self.d2(self.pd1(self.up1(h1)))))
        h3 = self.leakyrelu(self.bn7(self.d3(self.pd2(self.up2(h2)))))
        h4 = self.leakyrelu(self.bn8(self.d4(self.pd3(self.up3(h3)))))
        h5 = self.leakyrelu(self.bn9(self.d5(self.pd4(self.up4(h4)))))

        return self.sigmoid(self.d6(self.pd5(self.up5(h5))))

    def get_latent_var(self, x):
        mu, logvar = self.encode(x.view(-1, self.nc, self.ndf, self.ngf))
        z = self.reparametrize(mu, logvar)
        return z

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.nc, self.ndf, self.ngf))
        z = self.reparametrize(mu, logvar)
        res = self.decode(z)
        return res, mu, logvar


class AlexNetVAE(nn.Module):
    def __init__(self, z_dim, image_channels=3, h_dim=256*6*6, pretrained=True):
        super(AlexNetVAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout())

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc1_bn = nn.BatchNorm1d(z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc2_bn = nn.BatchNorm1d(z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        self.fc3_bn = nn.BatchNorm1d(h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(nb_filters=512, size=16), # b, 512, 16, 16
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False), # b, 64, 56, 56
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False), # b, 32, 112, 112
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False), # b, 16, 224, 224
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1, bias=False), # b, 3, 224, 224
            nn.Sigmoid())

        if pretrained:
            model_url = 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'
            state_dict = load_state_dict_from_url(model_url,
                                                  progress=True)
            self.encoder[0].weight

    def reparameterize(self, mu, logvar, device):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size())
        std = std.to(device)
        esp = esp.to(device)

        z = mu + std * esp
        return z

    def bottleneck(self, h, device):
        mu, logvar = self.fc1_bn(self.fc1(h)), self.fc2_bn(self.fc2(h))
        z = self.reparameterize(mu, logvar, device)
        return z, mu, logvar

    def encode(self, x, device):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h, device)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3_bn(self.fc3(z))
        z = self.decoder(z)
        return z

    def forward(self, x, device):
        z, mu, logvar = self.encode(x, device)
        generated_x = self.decode(z)
        return generated_x, mu, logvar, z

    def save_weights(self, path):
        torch.save(self.state_dict(), path)


class CarsConvAE(nn.Module):
    def __init__(self):
        super(CarsConvAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False), # b, 16, 64, 64
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False), # b, 32, 32, 32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False), # b, 64, 16, 16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False), # b, 128, 8, 8
            nn.BatchNorm2d(128),
            nn.ReLU())

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False), # b, 64, 16, 16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False), # b, 32, 32, 32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False), # b, 16, 64, 64
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False), # b, 3, 128, 128
            nn.Sigmoid())

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def save_weights(self, path):
        torch.save(self.state_dict(), path)
