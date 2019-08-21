#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 10:51:42 2018

@author: mhturner
"""
import sys
from pyqtgraph.widgets.PlotWidget import PlotWidget
from matplotlib import path
import seaborn as sns
from pyqtgraph.widgets.MatplotlibWidget import MatplotlibWidget
from matplotlib.widgets import LassoSelector, EllipseSelector
import matplotlib.cm as cm
import pyqtgraph as pg
from PyQt5.QtWidgets import (QPushButton, QWidget, QLabel, QGridLayout,
                             QApplication, QComboBox, QLineEdit, QFileDialog,
                             QTableWidget, QTableWidgetItem, QToolBar, QSlider)
import PyQt5.QtCore as QtCore
import PyQt5.QtGui as QtGui
import numpy as np
import os
from lazy5.inspect import get_hierarchy, get_attrs_group
from lazy5 import alter

from visanalysis import roi, plot_tools, plugin


class DataGUI(QWidget):

    def __init__(self, roi_type='freehand', roi_radius=None):
        super().__init__()

        self.experiment_file_name = None
        self.experiment_file_directory = None
        self.data_directory = None
        self.max_rois = 12
        self.roi_type = roi_type
        self.roi_radius = roi_radius

        self.colors = sns.color_palette("deep",n_colors = 20)

        self.initUI()

    def initUI(self):
        self.grid = QGridLayout(self)

        # Grid for file selection and attriute table
        self.file_control_grid = QGridLayout()
        self.file_control_grid.setSpacing(3)
        self.grid.addLayout(self.file_control_grid, 0, 0)

        self.attribute_grid = QGridLayout()
        self.attribute_grid.setSpacing(3)
        self.grid.addLayout(self.attribute_grid, 1, 0)

        self.roi_control_grid = QGridLayout()
        self.roi_control_grid.setSpacing(3)
        self.grid.addLayout(self.roi_control_grid, 0, 1)

        self.plot_grid = QGridLayout()
        self.plot_grid.setSpacing(3)
        self.grid.addLayout(self.plot_grid, 1, 1)

        # # # # File control browser: # # # # # # # #
        loadButton = QPushButton("Load expt. file", self)
        loadButton.clicked.connect(self.selectDataFile)
        # Label with current expt file
        self.currentExperimentLabel = QLabel('')
        self.file_control_grid.addWidget(loadButton, 0, 0)
        self.file_control_grid.addWidget(self.currentExperimentLabel, 1, 0)

        directoryButton = QPushButton("Select data directory", self)
        directoryButton.clicked.connect(self.selectDataDirectory)
        self.file_control_grid.addWidget(directoryButton, 0, 1)
        self.data_directory_display = QLabel('')
        self.data_directory_display.setFont(QtGui.QFont('SansSerif', 8))
        self.file_control_grid.addWidget(self.data_directory_display, 1, 1)

        registerButton = QPushButton("Register image series", self)
        registerButton.clicked.connect(self.registerStacks)
        self.file_control_grid.addWidget(registerButton, 2, 0)
        attachDatabutton = QPushButton("Attach data to file", self)
        attachDatabutton.clicked.connect(self.attachData)
        self.file_control_grid.addWidget(attachDatabutton, 2, 1)


        # # # # Attribute browser: # # # # # # # #
        # Heavily based on QtHdfLoad from LazyHDF5
        # Group selection combobox
        self.comboBoxGroupSelect = QComboBox()
        self.comboBoxGroupSelect.currentTextChanged.connect(self.groupChange)
        self.file_control_grid.addWidget(self.comboBoxGroupSelect, 3, 0, 1, 2)

        # Attribute table
        self.tableAttributes = QTableWidget()
        self.tableAttributes.setStyleSheet("")
        self.tableAttributes.setColumnCount(2)
        self.tableAttributes.setObjectName("tableAttributes")
        self.tableAttributes.setRowCount(0)
        item = QTableWidgetItem()
        font = QtGui.QFont()
        font.setPointSize(10)
        item.setFont(font)
        item.setBackground(QtGui.QColor(121, 121, 121))
        brush = QtGui.QBrush(QtGui.QColor(91, 91, 91))
        brush.setStyle(QtCore.Qt.SolidPattern)
        item.setForeground(brush)
        self.tableAttributes.setHorizontalHeaderItem(0, item)
        item = QTableWidgetItem()
        item.setBackground(QtGui.QColor(123, 123, 123))
        brush = QtGui.QBrush(QtGui.QColor(91, 91, 91))
        brush.setStyle(QtCore.Qt.SolidPattern)
        item.setForeground(brush)
        self.tableAttributes.setHorizontalHeaderItem(1, item)
        self.tableAttributes.horizontalHeader().setCascadingSectionResizes(True)
        self.tableAttributes.horizontalHeader().setHighlightSections(False)
        self.tableAttributes.horizontalHeader().setSortIndicatorShown(True)
        self.tableAttributes.horizontalHeader().setStretchLastSection(True)
        self.tableAttributes.verticalHeader().setVisible(False)
        self.tableAttributes.verticalHeader().setHighlightSections(False)
        item = self.tableAttributes.horizontalHeaderItem(0)
        item.setText("Attribute")
        item = self.tableAttributes.horizontalHeaderItem(1)
        item.setText("Value")

        self.tableAttributes.itemChanged.connect(self.update_attrs_to_file)
        self.attribute_grid.addWidget(self.tableAttributes, 3, 0, 1, 8)

        # Roi control buttons
        # ROI type drop-down
        self.RoiTypeComboBox = QComboBox(self)
        self.RoiTypeComboBox.addItem("freehand")
        radii = [1, 2, 3, 4, 6, 8]
        for radius in radii:
            self.RoiTypeComboBox.addItem("circle:"+str(radius))
        self.RoiTypeComboBox.activated.connect(self.selectRoiType)
        self.roi_control_grid.addWidget(self.RoiTypeComboBox, 0, 0)

        # Clear all ROIs button
        self.clearROIsButton = QPushButton("Clear ROIs", self)
        self.clearROIsButton.clicked.connect(self.clearRois)
        self.roi_control_grid.addWidget(self.clearROIsButton, 0, 2)

        # Available ROIs combobox
        self.RoiComboBox = QComboBox(self)
        self.RoiComboBox.activated.connect(self.loadRois)
        self.RoiComboBox.addItem("(select an ROI set)")
        # for roi_set in roi.getAvailableROIsets():
        #     self.RoiComboBox.addItem(roi_set)
        self.roi_control_grid.addWidget(self.RoiComboBox, 1, 0)

        # ROIset file name line edit box
        self.defaultRoiSetName = "roi_set_name"
        self.le_roiSetName = QLineEdit(self.defaultRoiSetName)
        self.roi_control_grid.addWidget(self.le_roiSetName, 1, 1)

        # Save ROIs button
        self.saveROIsButton = QPushButton("Save ROIs", self)
        self.saveROIsButton.clicked.connect(self.saveRois)
        self.roi_control_grid.addWidget(self.saveROIsButton, 1, 2)

        # TODO: slider for roi / poi index
        self.roiSlider = QSlider(QtCore.Qt.Horizontal, self)
        self.roi_control_grid.addWidget(self.roiSlider, 2, 1, 1, 2)



        self.responsePlot = PlotWidget()
        self.plot_grid.addWidget(self.responsePlot, 0, 0)

        # Image canvas for image and lasso widget
        self.roi_canvas = MatplotlibWidget()
        toolbar = self.roi_canvas.findChild(QToolBar)
        toolbar.setVisible(False)
        self.roi_fig = self.roi_canvas.getFigure()
        self.roi_ax = self.roi_fig.add_subplot(1, 1, 1)
        self.roi_ax.set_aspect('equal')
        self.roi_ax.set_axis_off()
        self.plot_grid.addWidget(self.roi_canvas, 1, 0)
        self.plot_grid.setRowStretch(0, 1)
        self.plot_grid.setRowStretch(1, 3)

        self.setWindowTitle('Visanalysis')
        self.setGeometry(200, 200, 1200, 600)
        self.show()

    def selectDataFile(self):
        filePath, _ = QFileDialog.getOpenFileName(self, "Open file")
        self.experiment_file_name = os.path.split(filePath)[1].split('.')[0]
        self.experiment_file_directory = os.path.split(filePath)[0]

        if self.experiment_file_name is not '':
            self.currentExperimentLabel.setText(self.experiment_file_name)
            self.initializeDataAnalysis()
            self.populateGroups()

    def selectDataDirectory(self):
        filePath = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.data_directory = filePath
        self.data_directory_display.setText('..' + self.data_directory[-24:])

    def initializeDataAnalysis(self):
        file_path = os.path.join(self.experiment_file_directory, self.experiment_file_name + '.hdf5')
        data_type = plugin.base.getDataType(file_path)
        # Load plugin based on Rig name in hdf5 file
        if data_type == 'Bruker':
            self.plugin = plugin.bruker.BrukerPlugin()
        elif data_type == 'AODscope':
            pass #TODO: make plugin
        else:
            self.plugin = plugin.base.BasePlugin()

    def registerStacks(self):
        # TODO: spin this up in a separate thread maybe
        file_path = os.path.join(self.experiment_file_directory, self.experiment_file_name + '.hdf5')
        self.plugin.registerAndSaveStacks(self.experiment_file_name, file_path, self.data_directory)

    def attachData(self):
        file_path = os.path.join(self.experiment_file_directory, self.experiment_file_name + '.hdf5')
        self.plugin.attachData(self.experiment_file_name, file_path, self.data_directory)
        print('Data attached')

    def populateGroups(self):  # Qt-related pylint: disable=C0103
        """ Populate dropdown box of group comboBoxGroupSelect """
        file_path = os.path.join(self.experiment_file_directory, self.experiment_file_name + '.hdf5')
        self.group_dset_dict = get_hierarchy(file_path)
        # Load Group dropdown box
        self.comboBoxGroupSelect.clear()
        for key in self.group_dset_dict:
            if 'epochs' in key:
                pass
            elif 'stimulus_timing' in key:
                pass
            else:
                self.comboBoxGroupSelect.addItem(key)
        return [file_path]

    def populate_attrs(self, attr_dict=None, editable_values = False):
        """ Populate attribute for currently selected group """
        self.tableAttributes.blockSignals(True) #block udpate signals for auto-filled forms
        self.tableAttributes.setRowCount(0)
        self.tableAttributes.setColumnCount(2)
        self.tableAttributes.setSortingEnabled(False)

        if attr_dict:
            for num, key in enumerate(attr_dict):
                self.tableAttributes.insertRow(self.tableAttributes.rowCount())
                key_item = QTableWidgetItem(key)
                key_item.setFlags( QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled )
                self.tableAttributes.setItem(num, 0, key_item)

                val_item = QTableWidgetItem(str(attr_dict[key]))
                if editable_values:
                    val_item.setFlags( QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsEnabled )
                else:
                    val_item.setFlags( QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled )
                self.tableAttributes.setItem(num, 1, val_item)

        self.tableAttributes.blockSignals(False)

    def update_attrs_to_file(self, item):
        file_path = os.path.join(self.experiment_file_directory, self.experiment_file_name + '.hdf5')
        group_path = self.comboBoxGroupSelect.currentText()

        attr_key = self.tableAttributes.item(item.row(),0).text()
        attr_val = item.text()

        # update attr in file
        alter.alter_attr(group_path, attr_key, attr_val, file=file_path)
        print('Changed attr {} to = {}'.format(attr_key, attr_val))

    def groupChange(self):  # Qt-related pylint: disable=C0103
        group_path = self.comboBoxGroupSelect.currentText()
        if group_path != '':
            file_path = os.path.join(self.experiment_file_directory, self.experiment_file_name + '.hdf5')

            attr_dict = get_attrs_group(file_path, group_path)
            if 'series' in group_path.split('/')[-1]:
                editable_values = False  # don't let user edit epoch parameters
            else:
                editable_values = True
            self.populate_attrs(attr_dict = attr_dict, editable_values = editable_values)

        levels = len(group_path.split('/'))
        if levels > 2:
            parent = group_path.split('/')[-2]
        else:
            parent = ''

        if parent == 'rois':  # selected node is an existing roi set
            print('roi set')
            self.loadRois()
            self.refreshLassoWidget(self.roi_image)
            self.redrawRoiTraces()
        elif 'series_' in group_path:  # selected node is within a series group
            print('selected series')
            series_number = int(group_path.split('series_')[-1].split('/')[0])
            if self.data_directory is not None:  # user has selected a raw data directory
                self.current_series = self.plugin.loadImageSeries(self.experiment_file_name, self.data_directory, series_number)
                self.roi_image = np.mean(self.current_series, axis=2)
                self.refreshLassoWidget(self.roi_image)

            else:
                print('Select a data directory first')

# %% # # # # # # # # ROI SELECTOR WIDGET # # # # # # # # # # # # # # # # # # #

    def refreshLassoWidget(self, image):
        self.roi_ax.imshow(image, cmap=cm.gray)
        self.roi_canvas.draw()

        # Pixel coordinates of lasso selector
        pixX = np.arange(image.shape[1])
        pixY = np.arange(image.shape[0])
        yv, xv = np.meshgrid(pixX, pixY)
        self.roi_pix = np.vstack((yv.flatten(), xv.flatten())).T
        if self.roi_type == 'circle':
            self.lasso = EllipseSelector(self.roi_ax, self.onselectEllipse)
        elif self.roi_type == 'freehand':
            self.lasso = LassoSelector(self.roi_ax, self.onselectFreehand)
        else:
            print('Warning ROI type not recognized. Choose circle or freehand')

    def onselectFreehand(self, verts):
        new_roi_path = path.Path(verts)
        ind = new_roi_path.contains_points(self.roi_pix, radius=1)
        self.updateRoiSelection(ind, new_roi_path)

    def onselectEllipse(self, pos1, pos2, definedRadius=None):
        x1 = np.round(pos1.xdata)
        x2 = np.round(pos2.xdata)
        y1 = np.round(pos1.ydata)
        y2 = np.round(pos2.ydata)

        radiusX = np.sqrt((x1 - x2)**2)/2
        radiusY = np.sqrt((y1 - y2)**2)/2
        if self.roiRadius is not None:
            radiusX = self.roiRadius

        center = (np.round((x1 + x2)/2), np.round((y1 + y2)/2))
        new_roi_path = path.Path.circle(center = center, radius = radiusX)
        ind = new_roi_path.contains_points(self.roi_pix, radius=0.5)

        self.updateRoiSelection(ind, new_roi_path)

    def updateRoiSelection(self, ind, path):
        mask = roi.getRoiMask(self.roi_image, ind)
        self.new_roi_resp = roi.getRoiDataFromMask(current_series, mask)

        #update list of roi data
        self.roi_mask.append(mask)
        self.roi_path.append(path)
        self.roi_response.append(self.new_roi_resp)

        # Update figures
        self.redrawRoiTraces()

    def redrawRoiTraces(self):
        roiIndex = 0
        self.responsePlot.clear()
        penStyle = pg.mkPen(color = tuple([255*x for x in self.colors[roiIndex]]))
        self.responsePlot.plot(np.squeeze(self.roi_response[roiIndex].T), pen=penStyle )

        if len(self.roi_mask) > 0:
            newImage = plot_tools.overlayImage(self.roi_image, self.roi_mask , 0.5, self.colors)
        else:
            newImage = self.roi_image
        # self.show()


# %% # # # # # # # # LOADING / SAVING / COMPUTING ROIS # # # # # # # # # # # # # # # # # # #
    def loadRois(self):
        file_path = os.path.join(self.experiment_file_directory, self.experiment_file_name + '.hdf5')
        roi_set_path = self.comboBoxGroupSelect.currentText()
        self.roi_response, self.roi_image, self.roi_path, self.roi_mask = roi.loadRoiSet(file_path, roi_set_path)

    def saveRois(self):
        file_path = os.path.join(self.experiment_file_directory, self.experiment_file_name + '.hdf5')
        roi_set_name = self.le_roiSetName.text()

        roi.saveRoiSet(file_path, roi_set_path, roi_response, roi_image, roi_path, roi_mask)

    def deleteRoi(self):
        pass

    def clearRois(self):
        pass

    def selectRoiType(self):
        pass






if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DataGUI()
    sys.exit(app.exec_())
