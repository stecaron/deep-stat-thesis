import pandas

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

# Data source: https://github.com/deric/clustering-benchmark/blob/master/src/main/resources/datasets/artificial/chainlink.arff
dt = pandas.read_csv('src/chainlink/chainlink.csv')


def plot_chainlink(dt, labels, show=False):

    dt["group"] = labels
    groups = dt["group"].unique().tolist()
    colors = ("red", "blue")
    group_names = ("group1", "group2")

    fig = plt.figure()  
    ax = fig.add_subplot(111, projection='3d')
    for group, color, group_name in zip(groups, colors, group_names):
        dt_temp = dt.loc[dt["group"] == group]
        ax.scatter(dt_temp["x"], dt_temp["y"], dt_temp["z"], alpha=0.8, c=color, label=group_name)

    ax.legend()

    if show:
        plt.show()

    return(fig)