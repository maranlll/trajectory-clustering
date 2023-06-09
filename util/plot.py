import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import colorsys
import numpy as np


def figure(num):
    plt.figure(num)

def subplot(nrows, ncols, index):
    plt.subplot(nrows, ncols, index)

def subplot(args):
    plt.subplot(args)
    
def plot(x, y, *args, **kwargs):
    plt.plot(x, y, *args, **kwargs)
    
def xlim(xmin, xmax):
    plt.xlim(xmin, xmax)
    
def ylim(ymin, ymax):
    plt.ylim(ymin, ymax)

def show():
    plt.show()
    
def scatter(x, y, *args, **kwargs):
    plt.scatter(x, y, *args, **kwargs)

def plot_trajectory(trajectory, color=(np.random.random(), np.random.random(), np.random.random()), alpha=1.0):
    plt.plot([p[0] for p in trajectory], [p[1] for p in trajectory], c=color, alpha=alpha)

def get_rand_color_map(nlabels, type='bright', first_color_black=False, last_color_black=False):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :param verbose: Prints the number of labels and shows the colormap. True or False
    :return: colormap for matplotlib
    """

    if type not in ('bright', 'soft'):
        print ('Please choose "bright" or "soft" for type')
        return

    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.2, high=1),
                          np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in range(nlabels)]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    return random_colormap

def get_color_map(n, name='hsv'):
    return plt.cm.get_cmap(name, n)