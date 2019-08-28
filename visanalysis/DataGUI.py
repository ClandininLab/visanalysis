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
                             QTableWidget, QTableWidgetItem, QToolBar, QSlider,
                             QMessageBox, QTreeWidget, QTreeWidgetItem)
import PyQt5.QtCore as QtCore
from PyQt5.QtCore import QThread
import PyQt5.QtGui as QtGui
import numpy as np
import os
from lazy5.inspect import get_attrs_group
from lazy5 import alter
import inspect
import skimage.io as io

from visanalysis import roi, plot_tools, plugin
import visanalysis


class DataGUI(QWidget):

    def __init__(self):
        super().__init__()

        self.experiment_file_name = None
        self.experiment_file_directory = None
        self.data_directory = None
        self.max_rois = 12
        self.roi_type = 'freehand'
        self.roi_radius = None

        self.current_roi_index = 0
        self.roi_response = []
        self.roi_mask = []
        self.roi_path = []
        self.roi_image = []

        img_path = os.path.join(inspect.getfile(visanalysis).split('visanalysis')[0],
                                'visanalysis',
                                'resources',
                                'clandinin_stainedglass.png')
        self.clandinin_logo = io.imread(img_path)

        self.colors = sns.color_palette("deep", n_colors = 20)

        self.initUI()

    def initUI(self):
        self.grid = QGridLayout(self)

        self.file_control_grid = QGridLayout()
        self.file_control_grid.setSpacing(3)
        self.grid.addLayout(self.file_control_grid, 0, 0)

        self.file_tree_grid = QGridLayout()
        self.file_tree_grid.setSpacing(3)
        self.grid.addLayout(self.file_tree_grid, 1, 0)

        self.group_control_grid = QGridLayout()
        self.group_control_grid.setSpacing(3)
        self.grid.addLayout(self.group_control_grid, 0, 1)

        self.attribute_grid = QGridLayout()
        self.attribute_grid.setSpacing(3)
        self.grid.addLayout(self.attribute_grid, 1, 1)

        self.roi_control_grid = QGridLayout()
        self.roi_control_grid.setSpacing(3)
        self.grid.addLayout(self.roi_control_grid, 0, 2)

        self.plot_grid = QGridLayout()
        self.plot_grid.setSpacing(3)
        self.grid.addLayout(self.plot_grid, 1, 2)

        # # # # File control browser: # # # # # # # # (0,0)
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

        # # # # File tree: # # # # # # # #  (1,0)
        self.groupTree = QTreeWidget(self)
        self.groupTree.setHeaderHidden(True)
        self.groupTree.itemClicked.connect(self.onTreeItemClicked)
        self.file_tree_grid.addWidget(self.groupTree, 3, 0, 2, 7)

        # # # # Group control: # # # # # # # # (0, 1)
        deleteGroupButton = QPushButton("Delete selected group", self)
        deleteGroupButton.clicked.connect(self.deleteSelectedGroup)
        self.group_control_grid.addWidget(deleteGroupButton, 7, 0, 1, 2)

        # # # # Attribute table: # # # # # # # # (1, 1)
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

        # # # # Roi control # # # # # # # # (0, 2)
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

        # ROIset file name line edit box
        self.defaultRoiSetName = "roi_set_name"
        self.le_roiSetName = QLineEdit(self.defaultRoiSetName)
        self.roi_control_grid.addWidget(self.le_roiSetName, 1, 1)

        # Save ROIs button
        self.saveROIsButton = QPushButton("Save ROIs", self)
        self.saveROIsButton.clicked.connect(self.saveRois)
        self.roi_control_grid.addWidget(self.saveROIsButton, 1, 0)

        # Delete current roi button
        self.deleteROIButton = QPushButton("Delete ROI", self)
        self.deleteROIButton.clicked.connect(self.deleteRoi)
        self.roi_control_grid.addWidget(self.deleteROIButton, 2, 0)

        # Current roi slider
        self.roiSlider = QSlider(QtCore.Qt.Horizontal, self)
        self.roiSlider.setMinimum(0)
        self.roiSlider.setMaximum(self.max_rois)
        self.roiSlider.valueChanged.connect(self.sliderUpdated)
        self.roi_control_grid.addWidget(self.roiSlider, 2, 1, 1, 2)

        self.responsePlot = PlotWidget()
        self.plot_grid.addWidget(self.responsePlot, 0, 0)

        # # # # Image canvas # # # # # # # # (1, 2)
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

    def _populateTree(self, widget, dict):
        widget.clear()
        self.fill_item(widget.invisibleRootItem(), dict)

    def fill_item(self, item, value):
        item.setExpanded(True)
        if type(value) is dict:
            for key, val in sorted(value.items()):
                child = QTreeWidgetItem()
                child.setText(0, key)
                item.addChild(child)
                self.fill_item(child, val)
        elif type(value) is list:
            for val in value:
                child = QTreeWidgetItem()
                item.addChild(child)
                if type(val) is dict:
                    child.setText(0, '[dict]')
                    self.fill_item(child, val)
                elif type(val) is list:
                    child.setText(0, '[list]')
                    self.fill_item(child, val)
                else:
                    child.setText(0, val)
                child.setExpanded(True)
        else:
            child = QTreeWidgetItem()
            child.setText(0, value)
            item.addChild(child)

    def onTreeItemClicked(self, item, column):
        file_path = os.path.join(self.experiment_file_directory, self.experiment_file_name + '.hdf5')
        group_path = self.plugin.getPathFromTreeItem(self.groupTree.selectedItems()[0])
        if 'series_' in group_path:
            self.series_number = int(group_path.split('series_')[-1].split('/')[0])
            print('selected series {}'.format(self.series_number))

        if group_path != '':
            attr_dict = get_attrs_group(file_path, group_path)
            if 'series' in group_path.split('/')[-1]:
                editable_values = False  # don't let user edit epoch parameters
            else:
                editable_values = True
            self.populate_attrs(attr_dict = attr_dict, editable_values = editable_values)

        if item.parent() is not None:
            if item.parent().text(column) == 'rois': # selected existing roi group
                roi_set_name = item.text(column)
                print('Selected roi set {} from series {}'.format(roi_set_name, self.series_number))
                self.le_roiSetName.setText(roi_set_name)
                self.loadRois()
                self.redrawRoiTraces()

        # show roi image
        if self.data_directory is not None:  # user has selected a raw data directory
            kwargs = {'data_directory': self.data_directory,
                      'series_number': self.series_number,
                      'experiment_file_name': self.experiment_file_name,
                      'file_path': file_path,
                      'pmt': 1}
            self.roi_image = self.plugin.getRoiImage(**kwargs)
            self.redrawRoiTraces()
        else:
            print('Select a data directory before drawing rois')

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
            self.plugin = plugin.aodscope.AodScopePlugin()
        else:
            self.plugin = plugin.base.BasePlugin()

    def registerStacks(self):
        file_path = os.path.join(self.experiment_file_directory, self.experiment_file_name + '.hdf5')
        self.registerStacksThread = registerStacksThread(plugin=self.plugin,
                                                         experiment_file_name=self.experiment_file_name,
                                                         file_path=file_path,
                                                         data_directory=self.data_directory)

        self.registerStacksThread.start()

    def attachData(self):
        if self.data_directory is not None:
            file_path = os.path.join(self.experiment_file_directory, self.experiment_file_name + '.hdf5')
            self.plugin.attachData(self.experiment_file_name, file_path, self.data_directory)
            print('Data attached')
        else:
            print('Select a data directory before attaching new data')

    def deleteSelectedGroup(self):
        file_path = os.path.join(self.experiment_file_directory, self.experiment_file_name + '.hdf5')
        group_path = self.plugin.getPathFromTreeItem(self.groupTree.selectedItems()[0])
        group_name = group_path.split('/')[-1]

        buttonReply = QMessageBox.question(self,
                                           'Delete series',
                                           "Are you sure you want to delete group {}?".format(group_name),
                                           QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if buttonReply == QMessageBox.Yes:
            self.plugin.deleteGroup(file_path=file_path,
                                    group_path=group_path)
            print('Deleted group {}'.format(group_name))
            self.populateGroups()
        else:
            print('Delete aborted')

    def populateGroups(self):
        file_path = os.path.join(self.experiment_file_directory, self.experiment_file_name + '.hdf5')
        self.group_dset_dict = self.plugin.getHierarchy(file_path)
        self._populateTree(self.groupTree, self.group_dset_dict)

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
        group_path = self.plugin.getPathFromTreeItem(self.groupTree.selectedItems()[0])

        attr_key = self.tableAttributes.item(item.row(),0).text()
        attr_val = item.text()

        # update attr in file
        alter.alter_attr(group_path, attr_key, attr_val, file=file_path)
        print('Changed attr {} to = {}'.format(attr_key, attr_val))

# %% # # # # # # # # ROI SELECTOR WIDGET # # # # # # # # # # # # # # # # # # #

    def refreshLassoWidget(self):
        self.roi_ax.clear()
        init_lasso = False
        if len(self.roi_image) > 0:
            if len(self.roi_mask) > 0:
                newImage = plot_tools.overlayImage(self.roi_image, self.roi_mask , 0.5, self.colors)
            else:
                newImage = self.roi_image
            self.roi_ax.imshow(newImage, cmap=cm.gray)
            init_lasso = True
        else:
            self.roi_ax.imshow(self.clandinin_logo)

        self.roi_canvas.draw()

        if init_lasso:
            pixX = np.arange(self.roi_image.shape[1])
            pixY = np.arange(self.roi_image.shape[0])
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
        if self.roi_radius is not None:
            radiusX = self.roi_radius

        center = (np.round((x1 + x2)/2), np.round((y1 + y2)/2))
        new_roi_path = path.Path.circle(center = center, radius = radiusX)
        ind = new_roi_path.contains_points(self.roi_pix, radius=0.5)

        self.updateRoiSelection(ind, new_roi_path)

    def updateRoiSelection(self, ind, path):
        mask = roi.getRoiMask(self.roi_image, ind)
        self.new_roi_resp = self.plugin.getRoiDataFromMask(mask=mask,
                                                           data_directory=self.data_directory,
                                                           series_number=self.series_number,
                                                           experiment_file_name=self.experiment_file_name)

        # update list of roi data
        self.roi_mask.append(mask)
        self.roi_path.append(path)
        self.roi_response.append(self.new_roi_resp)
        # update slider to show most recently drawn roi response
        self.current_roi_index = len(self.roi_response)-1
        self.roiSlider.setValue(self.current_roi_index)

        # Update figures
        self.redrawRoiTraces()

    def sliderUpdated(self):
        self.current_roi_index = self.roiSlider.value()
        self.redrawRoiTraces()

    def redrawRoiTraces(self):
        self.responsePlot.clear()
        if self.current_roi_index < len(self.roi_response):
            penStyle = pg.mkPen(color = tuple([255*x for x in self.colors[self.current_roi_index]]))
            self.responsePlot.plot(np.squeeze(self.roi_response[self.current_roi_index].T), pen=penStyle)

        self.refreshLassoWidget()

# %% # # # # # # # # LOADING / SAVING / COMPUTING ROIS # # # # # # # # # # # # # # # # # # #

    def loadRois(self):
        file_path = os.path.join(self.experiment_file_directory, self.experiment_file_name + '.hdf5')
        roi_set_path = self.plugin.getPathFromTreeItem(self.groupTree.selectedItems()[0])
        self.roi_response, self.roi_image, self.roi_path, self.roi_mask = roi.loadRoiSet(file_path, roi_set_path)

    def saveRois(self):
        file_path = os.path.join(self.experiment_file_directory, self.experiment_file_name + '.hdf5')
        roi_set_name = self.le_roiSetName.text()
        roi.saveRoiSet(file_path, series_number=self.series_number,
                     roi_set_name=roi_set_name,
                     roi_mask=self.roi_mask,
                     roi_response=self.roi_response,
                     roi_image=self.roi_image,
                     roi_path=self.roi_path)
        print('Saved roi set {} to series {}'.format(roi_set_name, self.series_number))
        self.populateGroups()

    def deleteRoi(self):
        if self.current_roi_index < len(self.roi_response):
            self.roi_mask.pop(self.current_roi_index)
            self.roi_response.pop(self.current_roi_index)
            self.roi_path.pop(self.current_roi_index)
            self.roiSlider.setValue(self.current_roi_index-1)
            self.redrawRoiTraces()

    def clearRois(self):
        self.roi_mask = []
        self.roi_response = []
        self.roi_path = []
        self.roi_image = []
        self.responsePlot.clear()
        self.redrawRoiTraces()
        self.roi_ax.clear()

    def selectRoiType(self):
        self.roi_type = self.RoiTypeComboBox.currentText().split(':')[0]
        if 'circle' in self.RoiTypeComboBox.currentText():
            self.roi_radius = int(self.RoiTypeComboBox.currentText().split(':')[1])
        else:
            self.roi_radius = None
        self.redrawRoiTraces()


class registerStacksThread(QThread):
    # https://nikolak.com/pyqt-threading-tutorial/
    # https://stackoverflow.com/questions/41848769/pyqt5-object-has-no-attribute-connect
    def __init__(self, plugin, experiment_file_name, file_path, data_directory):
        QThread.__init__(self)
        self.plugin = plugin
        self.experiment_file_name = experiment_file_name
        self.file_path = file_path
        self.data_directory = data_directory

    def __del__(self):
        self.wait()

    def _startReg(self):
        self.plugin.registerAndSaveStacks(self.experiment_file_name, self.file_path, self.data_directory)

    def run(self):
        self._startReg()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DataGUI()
    sys.exit(app.exec_())
