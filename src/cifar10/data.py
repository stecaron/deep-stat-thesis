# Load MNIST data

import os
import torchvision
from torchvision import transforms

from torchvision.datasets.cifar import CIFAR10


def load_cifar10(path, download=False):

    img_transform_train = transforms.Compose([
        transforms.RandomRotation(degrees=25),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    img_transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CIFAR10(path,
                            train=True,
                            download=download,
                            transform=img_transform_train)
    test_dataset = CIFAR10(path,
                            train=False,
                            download=download,
                            transform=img_transform_test)
    return train_dataset, test_dataset


if __name__ == '__main__':
    train_data, test_data = load_cifar10(path='/Users/stephanecaron/Downloads/cifar10',
                             download=True)


