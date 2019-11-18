import numpy
import matplotlib.pyplot as plt


def plot_comparisons(true, preds):

    graph_height = 2
    graph_width = preds.shape[0]

    fig, ax = plt.subplots(graph_height, graph_width)

    i=0
    for id in range(true.shape[0]):
        plottable_image = numpy.reshape(true[id], (28, 28))
        ax[0, i].imshow(plottable_image, cmap='gray_r')
        ax[0, i].axis('off')
        i+=1
    
    i=0
    for id in range(preds.shape[0]):
        plottable_image = numpy.reshape(preds[id].detach(), (28, 28))
        ax[1, i].imshow(plottable_image, cmap='gray_r')
        ax[1, i].axis('off')
        i+=1
    
    plt.show()
