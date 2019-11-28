import matplotlib.pyplot as plt
import numpy


def plot_n_images(dt_mnist, target, nb_images=16, shape=(4, 4)):

    fig, ax = plt.subplots(shape[0], shape[1])
    ax = ax.flatten()

    # Sample images to plot
    img_idx = numpy.random.choice(numpy.argwhere(dt_mnist.targets == target)[0],
                                      nb_images,
                                      replace=False)

    i=0
    for id in img_idx:
        plottable_image = numpy.reshape(dt_mnist.data[id], (28, 28))
        ax[i].imshow(plottable_image, cmap='gray_r')
        ax[i].axis('off')
        i+=1
    
    plt.show()


def plot_outliers_idx(dt, idx_in, idx_out, shape = (2, 5)):

    fig, ax = plt.subplots(shape[0], shape[1])
    ax = ax.flatten()
    
    i=0
    for id in idx_in:
        plottable_image = numpy.reshape(dt[id], (28, 28))
        ax[i].imshow(plottable_image, cmap='gray_r')
        ax[i].axis('off')
        ax[i].set_title('Inliers')
        i+=1

    for id in idx_out:
        plottable_image = numpy.reshape(dt[id], (28, 28))
        ax[i].imshow(plottable_image, cmap='gray_r')
        ax[i].axis('off')
        ax[i].set_title('Outliers')
        i+=1
    
    plt.show()


if __name__ == '__main__':
    from src.mnist.data import load_mnist
    train_data, test_data = load_mnist('/Users/stephanecaron/Downloads/mnist')
    plot_n_images(train_data, 4)