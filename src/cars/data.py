import os
import numpy
import torchvision
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from skimage import transform


def load_image(file):
    return Image.open(file).convert('RGB')


class DataGenerator(Dataset):
    def __init__(self, path_dogs, path_cars, size, noise_perc, transform,
                 image_size):
        """
        Majority class is the cars
        Minority calss is the dogs
        """

        self.path_cars = path_cars
        self.path_dogs = path_dogs
        self.transform = transform
        self.image_size = image_size

        all_dogs_files = numpy.array(os.listdir(path_dogs))
        all_cars_files = numpy.array(os.listdir(path_cars))

        id_maj = numpy.random.choice(numpy.arange(0,
                                                  len(all_cars_files),
                                                  step=1),
                                     int((1 - noise_perc) * size),
                                     replace=False)

        id_min = numpy.random.choice(numpy.arange(0,
                                                  len(all_dogs_files),
                                                  step=1),
                                     int(noise_perc * size),
                                     replace=False)

        cars_files = all_cars_files[id_maj]
        dogs_files = all_dogs_files[id_min]

        self.images = numpy.concatenate((cars_files, dogs_files))
        self.labels = numpy.concatenate(
            (numpy.repeat(0,
                          len(cars_files)), numpy.repeat(1, len(dogs_files))))

    def __getitem__(self, index):
        filename = self.images[index]
        label = self.labels[index]

        if label == 0:
            path = self.path_cars
        else:
            path = self.path_dogs

        with open(os.path.join(path, filename), 'rb') as f:
            image = numpy.array(load_image(f))[..., :3]

        image = transform.resize(image, self.image_size)
        img_tensor = self.transform(image).type(torch.FloatTensor)
        img_tensor.transpose_(1, 2)

        return img_tensor, label

    def __len__(self):
        return len(self.images)
