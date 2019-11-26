# Load MNIST data

import os
import torchvision
from torchvision import transforms

from torchvision.datasets.mnist import MNIST


def load_mnist(path, download=False):

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


if __name__ == '__main__':
    train_data, test_data = load_mnist(path='/Users/stephanecaron/Downloads/mnist',
                             download=True)
