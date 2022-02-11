"""
Assorted plotting utilities.

https://github.com/ClandininLab/visanalysis
mhturner@stanford.edu
"""
import numpy as np
import matplotlib.pyplot as plt


def addLine(ax, x, y, line_name='', color='k', linestyle='-', marker='None'):
    ax.plot(x, y, linestyle=linestyle, marker=marker,
            linewidth=1, label=line_name, color=color)


def addErrorBars(ax, xdata, ydata, line_name='',
                 stat='sem',
                 mode='sticks',
                 color='k'):

    if len(xdata.shape) == 2 and len(ydata.shape) == 2:  # x and y error
        err_x = _calcError(xdata, stat)
        err_y = _calcError(ydata, stat)
        mean_x = np.mean(xdata, axis=0)
        mean_y = np.mean(ydata, axis=0)

        _addYError(ax, mean_x, mean_y, err_y, line_name, mode, stat, color)
        _addXError(ax, mean_x, mean_y, err_x, line_name, mode, stat, color)

    elif len(xdata.shape) == 1 and len(ydata.shape) == 2:  # y error
        err_y = _calcError(ydata, stat)
        mean_y = np.mean(ydata, axis=0)

        _addYError(ax, xdata, mean_y, err_y, line_name, mode, stat, color)
    elif len(xdata.shape) == 2 and len(ydata.shape) == 1:  # x error
        err_x = _calcError(xdata, stat)
        mean_x = np.mean(xdata, axis=0)

        _addXError(ax, mean_x, ydata, err_x, line_name, mode, stat, color)
    else:
        raise Exception('no population data to compute errors')


def addScaleBars(axis, dT, dF, T_value=-0.1, F_value=-0.4):
    axis.plot(T_value * np.ones((2)), np.array([F_value, F_value + dF]), 'k-', alpha=0.9)
    axis.plot(np.array([T_value, dT + T_value]), F_value * np.ones((2)), 'k-', alpha=0.9)


def _addXError(ax, x, y, err_x, line_name, mode, stat, color):
    xx = ax.plot([x - err_x, x + err_x], [y, y], linestyle='--', marker=None, linewidth=1, label=line_name + '_errX')


def _addYError(ax, x, y, err_y, line_name, mode, stat, color):
    yp = ax.plot(x, y - err_y, linestyle='--', marker=None,
                 linewidth=1, label=line_name + '_errY_plus', color=color)
    ym = ax.plot(x, y + err_y, linestyle='--', marker=None,
                 linewidth=1, label=line_name + '_errY_minus', color=color)
    ym[0].tag = mode
    yp[0].tag = 'hide'


def _calcError(data, stat):
    if stat == 'sem':
        err = np.std(data, axis=0) / np.sqrt(data.shape[0])
    elif stat == 'std':
        err = np.std(data, axis=0)
    return err


# tools for images:
def addImageScaleBar(ax, image, scale_bar_length, microns_per_pixel, location):
    dim_x = image.shape[1]
    dim_y = image.shape[0]
    dx = scale_bar_length / microns_per_pixel  # pixels
    if location[0] == 'l':
        start_y = 0.9 * dim_y
    elif location[0] == 'u':
        start_y = 0.1 * dim_y

    if location[1] == 'l':
        start_x = 0.1 * dim_x
        end_x = start_x + dx
    elif location[1] == 'r':
        start_x = 0.9 * dim_x
        end_x = start_x - dx
    ax.plot([start_x, end_x], [start_y, start_y], 'w')


def overlayImage(im, mask, alpha, colors=None, z=0):
    im = im / np.max(im)
    if len(im.shape) < 3:
        imRGB = np.tile(im[..., np.newaxis], 3)
    else:
        imRGB = im

    overlayComponent = 0
    origImageComponent = 0
    if len(mask[0].shape) == 2:
        compositeMask = np.tile(mask[0][..., np.newaxis], 3)
    else:
        compositeMask = np.tile(mask[0][:, :, z, np.newaxis], 3)
    for ind, currentRoi in enumerate(mask):
        if len(mask[0].shape) == 2:
            maskRGB = np.tile(currentRoi[..., np.newaxis], 3)
        else:
            maskRGB = np.tile(currentRoi[:, :, z, np.newaxis], 3)
        if colors is None:
            newColor = (1, 0, 0)
        else:
            newColor = colors[ind]

        compositeMask = compositeMask + maskRGB
        overlayComponent += alpha * np.array(newColor) * maskRGB
        origImageComponent += (1 - alpha) * maskRGB * imRGB

    untouched = (compositeMask == False) * imRGB

    im_out = untouched + overlayComponent + origImageComponent
    im_out = (im_out * 255).astype(np.uint8)
    return im_out


def cleanAxes(ax):
    ax.set_axis_off()
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
