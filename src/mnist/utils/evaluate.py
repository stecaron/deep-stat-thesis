import numpy
import torch
import matplotlib.pyplot as plt

from matplotlib.pyplot import savefig


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x
    

def plot_comparisons(true, preds):

    graph_height = 2
    graph_width = preds.shape[0]

    fig, ax = plt.subplots(graph_height, graph_width)

    i=0
    for id in range(true.shape[0]):
        plottable_image = numpy.reshape(true[id], (28, 28))
        ax[0, i].imshow(plottable_image, cmap='gray_r')
        ax[0, i].axis('off')
        plt.gray()
        i+=1
    
    i=0
    for id in range(preds.shape[0]):
        plottable_image = numpy.reshape(preds[id].detach(), (28, 28))
        ax[1, i].imshow(plottable_image, cmap='gray_r')
        ax[1, i].axis('off')
        plt.gray()
        i+=1
    
    plt.show()
    return fig
