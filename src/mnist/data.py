# Load MNIST data

import os
import torchvision
from torchvision import transforms

from torchvision.datasets.mnist import MNIST
from torchvision.datasets import FashionMNIST

from six.moves import urllib


def load_mnist(path, download=False):

    # Path for torchvision bug: https://github.com/pytorch/vision/issues/1938
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

    img_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = MNIST(path,
                          train=True,
                          download=download,
                          transform=img_transform)
    test_dataset = MNIST(path,
                         train=False,
                         download=download,
                         transform=img_transform)
    return train_dataset, test_dataset


def load_mnist_fashion(path, download=False):

    # Path for torchvision bug: https://github.com/pytorch/vision/issues/1938
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

    img_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = FashionMNIST(path,
                          train=True,
                          download=download,
                          transform=img_transform)
    test_dataset = FashionMNIST(path,
                         train=False,
                         download=download,
                         transform=img_transform)
    return train_dataset, test_dataset


if __name__ == '__main__':
    train_data, test_data = load_mnist(path='/Users/stephanecaron/Downloads/mnist',
                             download=True)
