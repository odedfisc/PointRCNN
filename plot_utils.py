import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors
import numpy as np


def map_clusters_to_colors(coloring):
    cmap = ListedColormap(
                        ['w', 'magenta', 'orange', 'mediumspringgreen', 'deepskyblue', 'pink', 'y', 'g', 'r', 'purple',
                         'lime', 'crimson', 'aqua'])
    coloring[coloring != 0] = np.mod(coloring[coloring != 0], len(cmap.colors) - 1) + 1
    c = cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=len(cmap.colors) - 1))
    color = c.to_rgba(coloring)
    return color