# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 12:46:07 2018

@author: mhturner
"""
import numpy as np
import matplotlib.colors as mcolors
import h5py
import os
import datetime
import inspect
import yaml

import visanalysis


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


def makeIgorStructure(ax, file_name=None, directory=None):
    if file_name is None:
        file_name = datetime.datetime.now().isoformat()
    if directory is None:
        directory = os.getcwd()
    axis_structure = getAxisStructure(ax)
    file_path = os.path.join(directory, file_name + '.h5')
    if os.path.isfile(file_path):
        os.remove(file_path)
        print('Overwriting existing figure file')
        print('New igor export: ' + file_name)

    figure_file = h5py.File(file_path, 'w-')

    for k, v in axis_structure.items():
        figure_file[file_name + '/' + k] = v

    figure_file.close()


def getAxisStructure(ax):
    axis_structure = {}
    axis_structure['Xlabel'] = ax.get_xlabel()
    axis_structure['Ylabel'] = ax.get_ylabel()
    axis_structure['Xlim'] = ax.get_xlim()
    axis_structure['Ylim'] = ax.get_ylim()
    axis_structure['Xscale'] = ax.get_xscale()
    axis_structure['Yscale'] = ax.get_yscale()

    for line in ax.lines:
        base_name = line.get_label()
        if hasattr(line, 'tag'):
            axis_structure[base_name + '_tag'] = line.tag

        # line data:
        axis_structure[base_name + '_X'] = line.get_xdata()
        axis_structure[base_name + '_Y'] = line.get_ydata()
        # colors:
        axis_structure[base_name + '_color'] = mcolors.to_rgb(line.get_color())
        axis_structure[base_name + '_markercolor'] = mcolors.to_rgb(line.get_markerfacecolor())
        # styles:
        axis_structure[base_name + '_marker'] = convertMarkerStyleToIgor(line.get_marker())
        axis_structure[base_name + '_linestyle'] = convertLineStyleToIgor(line.get_linestyle())

    # remove entries where item is None
    axis_structure = {k: v for k, v in axis_structure.items() if v is not None}
    return axis_structure


def convertMarkerStyleToIgor(marker_code):
    conversion_dictionary = {'.': 19, 'o': 8, 'v': 23, '^': 17, '<': 46, '>': 49,
                             's': 16, '+': 3, 'd': 29, 'D': 18, 'None': None}
    m = conversion_dictionary.get(marker_code)

    return m


def convertLineStyleToIgor(line_code):
    conversion_dictionary = {':': 1, '-.': 4, '--': 3,'-': 0, 'None': None}
    igor_code = conversion_dictionary.get(line_code)

    return igor_code


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
    compositeMask = np.tile(mask[0][:, :, z, np.newaxis], 3)
    for ind, currentRoi in enumerate(mask):
        maskRGB = np.tile(currentRoi[:, :, z, np.newaxis], 3)
        if colors is None:
            newColor = (1, 1, 1)
        else:
            newColor = colors[ind]

        compositeMask = compositeMask + maskRGB
        overlayComponent += alpha * np.array(newColor) * maskRGB
        origImageComponent += (1 - alpha) * maskRGB * imRGB

    untouched = (compositeMask == False) * imRGB

    im_out = untouched + overlayComponent + origImageComponent
    im_out = (im_out * 255).astype(np.uint8)
    return im_out
