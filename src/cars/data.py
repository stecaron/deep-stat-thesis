import os
import numpy
import torchvision
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from skimage import transform
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa


def load_image(file):
    return Image.open(file).convert('RGB')


def define_filenames(path_dogs, path_cars, train_size, test_size,
                     train_noise_perc, test_noise_perc):

    all_dogs_files = numpy.array(os.listdir(path_dogs))
    all_cars_files = numpy.array(os.listdir(path_cars))

    train_cars_size = int(train_size * (1 - train_noise_perc))
    train_dogs_size = int(train_size * train_noise_perc)
    test_cars_size = int(test_size * (1 - test_noise_perc))
    test_dogs_size = int(test_size * test_noise_perc)

    train_cars, test_cars, _, _ = train_test_split(all_cars_files,
                                                   numpy.repeat(
                                                       1, len(all_cars_files)),
                                                   train_size=train_cars_size,
                                                   test_size=test_cars_size)

    train_dogs, test_dogs, _, _ = train_test_split(all_dogs_files,
                                                   numpy.repeat(
                                                       1, len(all_dogs_files)),
                                                   train_size=train_dogs_size,
                                                   test_size=test_dogs_size)
    # add back the directory
    train_cars = numpy.array([os.path.join(path_cars, x) for x in train_cars])
    test_cars = numpy.array([os.path.join(path_cars, x) for x in test_cars])
    train_dogs = numpy.array([os.path.join(path_dogs, x) for x in train_dogs])
    test_dogs = numpy.array([os.path.join(path_dogs, x) for x in test_dogs])


    x_train = numpy.concatenate((train_cars, train_dogs))
    x_test = numpy.concatenate((test_cars, test_dogs))
    y_train = numpy.concatenate((numpy.repeat(0, len(train_cars)), numpy.repeat(1, len(train_dogs))))
    y_test = numpy.concatenate((numpy.repeat(0, len(test_cars)), numpy.repeat(1, len(test_dogs))))

    return x_train, x_test, y_train, y_test


class DataGenerator(Dataset):
    def __init__(self, x_files, labels, transform, image_size):
        """
        Majority class is the cars
        Minority calss is the dogs
        """

        self.images = x_files
        self.labels = labels
        self.transform = transform
        self.image_size = image_size

    def __getitem__(self, index):
        filename = self.images[index]
        label = self.labels[index]

        with open(filename, 'rb') as f:
            image = numpy.array(load_image(f))[..., :3]

        #image = numpy.swapaxes(image, 0, 2)
        #image = numpy.swapaxes(image, 1, 2)

        #seq = iaa.Sequential([
        #    iaa.Resize((224, 224))
        #])

        image = transform.resize(image, self.image_size)
        #image = seq(images=image)
        img_tensor = self.transform(image).type(torch.FloatTensor)
        img_tensor.transpose_(1, 2)

        return img_tensor, label

    def __len__(self):
        return len(self.images)


if __name__ == "__main__":
    define_filenames(
        os.path.join(os.path.expanduser("~"), 'Downloads/stanford_dogs'),
        os.path.join(os.path.expanduser("~"), 'Downloads/stanford_cars'), 4000,
        2000, 0.01, 0.1)
