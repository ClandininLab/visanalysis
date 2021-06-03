#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data file GUI and ROI drawer for visanalysis.

https://github.com/ClandininLab/visanalysis
mhturner@stanford.edu
"""
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib import path
import seaborn as sns
from matplotlib.widgets import LassoSelector, EllipseSelector
import matplotlib.cm as cm
from PyQt5.QtWidgets import (QPushButton, QWidget, QLabel, QGridLayout,
                             QApplication, QComboBox, QLineEdit, QFileDialog,
                             QTableWidget, QTableWidgetItem, QSlider,
                             QMessageBox, QTreeWidget, QTreeWidgetItem)
import PyQt5.QtCore as QtCore
import PyQt5.QtGui as QtGui
import numpy as np
import os

from visanalysis import plot_tools, plugin

import psutil


class DataGUI(QWidget):

    def __init__(self):
        super().__init__()

        self.experiment_file_name = None
        self.experiment_file_directory = None
        self.data_directory = None
        self.max_rois = 50
        self.roi_type = 'freehand'
        self.roi_radius = None
        self.existing_roi_set_paths = {}

        self.current_roi_index = 0
        self.current_z_slice = 0
        self.current_channel = 1 # index
        self.image_series_name = ''
        self.series_number = None
        self.roi_response = []
        self.roi_mask = []
        self.roi_path = []
        self.roi_image = None
        self.roi_path_list = []
        self.roi_z_list =[]

        self.blank_image = np.zeros((1, 1))

        self.colors = sns.color_palette("deep", n_colors=20)

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

        # Attach metadata to file
        attachDatabutton = QPushButton("Attach metadata to file", self)
        attachDatabutton.clicked.connect(self.attachData)
        self.file_control_grid.addWidget(attachDatabutton, 2, 0, 1, 2)

        # Select image data file
        selectImageDataFileButton = QPushButton("Select image data file", self)
        selectImageDataFileButton.clicked.connect(self.selectImageDataFile)
        self.file_control_grid.addWidget(selectImageDataFileButton, 3, 0, 1, 2)

        # # # # File tree: # # # # # # # #  (1,0)
        self.groupTree = QTreeWidget(self)
        self.groupTree.setHeaderHidden(True)
        self.groupTree.itemClicked.connect(self.onTreeItemClicked)
        self.file_tree_grid.addWidget(self.groupTree, 3, 0, 2, 7)

        # # # # Group control: # # # # # # # # (0, 1)
        deleteGroupButton = QPushButton("Delete selected group", self)
        deleteGroupButton.clicked.connect(self.deleteSelectedGroup)
        self.group_control_grid.addWidget(deleteGroupButton, 0, 0, 1, 2)

        # File name display
        self.currentImageFileNameLabel = QLabel('')
        self.group_control_grid.addWidget(self.currentImageFileNameLabel, 1, 0)

        # Channel drop down
        ch_label = QLabel('Channel:')
        self.ChannelComboBox = QComboBox(self)
        self.ChannelComboBox.addItem("1")
        self.ChannelComboBox.addItem("0")
        self.ChannelComboBox.activated.connect(self.selectChannel)
        self.group_control_grid.addWidget(ch_label, 2, 0)
        self.group_control_grid.addWidget(self.ChannelComboBox, 2, 1)

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

        # Response display type dropdown
        self.RoiResponseTypeComboBox = QComboBox(self)

        self.RoiResponseTypeComboBox.addItem("RawTrace")
        self.RoiResponseTypeComboBox.addItem("TrialAverage")
        self.RoiResponseTypeComboBox.addItem("TrialResponses")
        self.RoiResponseTypeComboBox.addItem("TrialAverageDFF")
        self.roi_control_grid.addWidget(self.RoiResponseTypeComboBox, 2, 2)

        # ROIset file name line edit box
        self.defaultRoiSetName = "roi_set_name"
        self.le_roiSetName = QLineEdit(self.defaultRoiSetName)
        self.roi_control_grid.addWidget(self.le_roiSetName, 1, 1)

        # Save ROIs button
        self.saveROIsButton = QPushButton("Save ROIs", self)
        self.saveROIsButton.clicked.connect(self.saveRois)
        self.roi_control_grid.addWidget(self.saveROIsButton, 1, 0)

        # Load ROI set combobox
        self.loadROIsComboBox = QComboBox(self)
        self.loadROIsComboBox.addItem("(load existing ROI set)")
        self.loadROIsComboBox.activated.connect(self.selectedExistingRoiSet)
        self.roi_control_grid.addWidget(self.loadROIsComboBox, 1, 2)
        self.updateExistingRoiSetList()

        # Delete current roi button
        self.deleteROIButton = QPushButton("Delete ROI", self)
        self.deleteROIButton.clicked.connect(self.deleteRoi)
        self.roi_control_grid.addWidget(self.deleteROIButton, 2, 0)

        # Current roi slider
        self.roiSlider = QSlider(QtCore.Qt.Horizontal, self)
        self.roiSlider.setMinimum(0)
        self.roiSlider.setMaximum(self.max_rois)
        self.roiSlider.valueChanged.connect(self.sliderUpdated)
        self.roi_control_grid.addWidget(self.roiSlider, 2, 1, 1, 1)

        plt.rc_context({'axes.edgecolor': 'white',
                        'xtick.color': 'white',
                        'ytick.color': 'white',
                        'figure.facecolor': 'black',
                        'axes.facecolor': 'black'})
        self.responseFig = plt.figure()
        self.responsePlot = self.responseFig.add_subplot(111)
        self.responseFig.subplots_adjust(left=0.05, bottom=0.20, top=0.95, right=0.98)
        self.responseCanvas = FigureCanvas(self.responseFig)
        self.responseCanvas.draw_idle()
        self.plot_grid.addWidget(self.responseCanvas, 0, 0)

        # # # # Image canvas # # # # # # # # (1, 2)
        self.roi_fig = plt.figure()
        self.roi_ax = self.roi_fig.add_subplot(111)
        self.roi_canvas = FigureCanvas(self.roi_fig)
        self.toolbar = NavigationToolbar(self.roi_canvas, self)
        self.roi_ax.set_aspect('equal')
        self.roi_ax.set_axis_off()
        self.plot_grid.addWidget(self.toolbar, 1, 0)
        self.plot_grid.addWidget(self.roi_canvas, 2, 0)
        self.plot_grid.setRowStretch(0, 1)
        self.plot_grid.setRowStretch(1, 3)
        self.plot_grid.setRowStretch(2, 3)

        # Current z slice slider
        self.zSlider = QSlider(QtCore.Qt.Horizontal, self)
        self.zSlider.setMinimum(0)
        self.zSlider.setMaximum(50)
        self.zSlider.setValue(0)
        self.zSlider.valueChanged.connect(self.zSliderUpdated)
        self.plot_grid.addWidget(self.zSlider, 3, 0)

        self.roi_fig.tight_layout()

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
        group_path = plugin.base.getPathFromTreeItem(self.groupTree.selectedItems()[0])
        self.clearRois()
        self.series_number = None
        if 'series_' in group_path:
            self.series_number = int(group_path.split('series_')[-1].split('/')[0])
            if self.plugin.dataIsAttached(file_path, self.series_number):
                self.plugin.updateImagingDataObject(self.experiment_file_directory, self.experiment_file_name, self.series_number)
            # look for image_file_name or ask user to select it
            if self.data_directory is not None:
                image_file_name = plugin.base.readImageFileName(file_path, self.series_number)
                if image_file_name is None or image_file_name == '':
                    image_file_path, _ = QFileDialog.getOpenFileName(self, "Select image file")
                    print('User selected image file at {}'.format(image_file_path))
                    image_file_name = os.path.split(image_file_path)[-1]
                    self.data_directory = os.path.split(image_file_path)[:-1][0]
                    plugin.base.attachImageFileName(file_path, self.series_number, image_file_name)
                    print('Attached image_file_name {} to series {}'.format(image_file_name, self.series_number))
                    print('Data directory is {}'.format(self.data_directory))

                self.image_file_name = image_file_name
                self.currentImageFileNameLabel.setText(self.image_file_name)

        if item.parent() is not None:
            if item.parent().text(column) == 'rois': # selected existing roi group
                roi_set_name = item.text(column)
                # print('Selected roi set {} from series {}'.format(roi_set_name, self.series_number))
                self.le_roiSetName.setText(roi_set_name)
                roi_set_path = plugin.base.getPathFromTreeItem(self.groupTree.selectedItems()[0])
                self.loadRois(roi_set_path)
                self.redrawRoiTraces()

        if group_path != '':
            attr_dict = plugin.base.getAttributesFromGroup(file_path, group_path)
            if 'series' in group_path.split('/')[-1]:
                editable_values = False  # don't let user edit epoch parameters
            else:
                editable_values = True
            self.populate_attrs(attr_dict=attr_dict, editable_values=editable_values)

        # show roi image
        if self.series_number is not None:
            if self.data_directory is not None:  # user has selected a raw data directory
                self.plugin.updateImageSeries(data_directory=self.data_directory,
                                              image_file_name=self.image_file_name,
                                              series_number=self.series_number,
                                              channel=self.current_channel)
                self.roi_image = self.plugin.mean_brain
                self.zSlider.setValue(0)
                self.zSlider.setMaximum(self.roi_image.shape[2]-1)
                self.redrawRoiTraces()

            else:
                print('Select a data directory before drawing rois')

        # # # TEST # # #
        memory_usage = psutil.Process(os.getpid()).memory_info().rss*10**-9
        print('Current Memory Usage: {:.2f}GB'.format(memory_usage))
        sys.stdout.flush()
        # # # TEST # # #

    def updateExistingRoiSetList(self):
        if self.experiment_file_name is not None:
            file_path = os.path.join(self.experiment_file_directory, self.experiment_file_name + '.hdf5')
            self.existing_roi_set_paths = self.plugin.getRoiSetPaths(file_path)  # dictionary of name: full path
            self.loadROIsComboBox.clear()
            for r_path in self.existing_roi_set_paths:
                self.loadROIsComboBox.addItem(r_path)

            self.show()

    def selectedExistingRoiSet(self):
        file_path = os.path.join(self.experiment_file_directory, self.experiment_file_name + '.hdf5')
        roi_set_key = self.loadROIsComboBox.currentText()
        roi_set_path = self.existing_roi_set_paths[roi_set_key]

        _, _, self.roi_path, self.roi_mask = self.plugin.loadRoiSet(file_path, roi_set_path)

        if self.series_number is not None:
            self.roi_response = []
            for new_path in self.roi_path:
                new_roi_resp = self.plugin.getRoiDataFromPath(roi_path=new_path)
                self.roi_response.append(new_roi_resp)

            # update slider to show most recently drawn roi response
            self.current_roi_index = len(self.roi_response)-1
            self.roiSlider.setValue(self.current_roi_index)

            # Update figures
            self.redrawRoiTraces()

    def selectDataFile(self):
        filePath, _ = QFileDialog.getOpenFileName(self, "Open experiment (hdf5) file")
        self.experiment_file_name = os.path.split(filePath)[1].split('.')[0]
        self.experiment_file_directory = os.path.split(filePath)[0]

        if self.experiment_file_name != '':
            self.currentExperimentLabel.setText(self.experiment_file_name)
            self.initializeDataAnalysis()
            self.populateGroups()
            self.updateExistingRoiSetList()

    def selectDataDirectory(self):
        filePath = str(QFileDialog.getExistingDirectory(self, "Select data directory"))
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

        self.plugin.parent_gui = self

        # # # TEST # # #
        memory_usage = psutil.Process(os.getpid()).memory_info().rss*10**-9
        print('Current memory usage: {:.2f}GB'.format(memory_usage))
        sys.stdout.flush()
        # # # TEST # # #

    def attachData(self):
        if self.data_directory is not None:
            file_path = os.path.join(self.experiment_file_directory, self.experiment_file_name + '.hdf5')
            self.plugin.attachData(self.experiment_file_name, file_path, self.data_directory)
            print('Data attached')
        else:
            print('Select a data directory before attaching new data')

    def selectImageDataFile(self):
        file_path = os.path.join(self.experiment_file_directory, self.experiment_file_name + '.hdf5')

        image_file_path, _ = QFileDialog.getOpenFileName(self, "Select image file")
        print('User selected image file at {}'.format(image_file_path))
        self.image_file_name = os.path.split(image_file_path)[-1]
        self.data_directory = os.path.split(image_file_path)[:-1][0]
        plugin.base.attachImageFileName(file_path, self.series_number, self.image_file_name)
        print('Attached image_file_name {} to series {}'.format(self.image_file_name, self.series_number))
        print('Data directory is {}'.format(self.data_directory))

        self.currentImageFileNameLabel.setText(self.image_file_name)

        # show roi image
        if self.series_number is not None:
            if self.data_directory is not None:  # user has selected a raw data directory
                self.plugin.updateImageSeries(data_directory=self.data_directory,
                                              image_file_name=self.image_file_name,
                                              series_number=self.series_number,
                                              channel=self.current_channel)
                self.roi_image = self.plugin.mean_brain
                self.zSlider.setValue(0)
                self.zSlider.setMaximum(self.roi_image.shape[2]-1)
                self.redrawRoiTraces()
            else:
                print('Select a data directory before drawing rois')

    def deleteSelectedGroup(self):
        file_path = os.path.join(self.experiment_file_directory, self.experiment_file_name + '.hdf5')
        group_path = plugin.base.getPathFromTreeItem(self.groupTree.selectedItems()[0])
        group_name = group_path.split('/')[-1]

        buttonReply = QMessageBox.question(self,
                                           'Delete series',
                                           "Are you sure you want to delete group {}?".format(group_name),
                                           QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if buttonReply == QMessageBox.Yes:
            plugin.base.deleteGroup(file_path=file_path,
                                    group_path=group_path)
            print('Deleted group {}'.format(group_name))
            self.updateExistingRoiSetList()
            self.populateGroups()
        else:
            print('Delete aborted')

    def populateGroups(self):
        file_path = os.path.join(self.experiment_file_directory, self.experiment_file_name + '.hdf5')
        self.group_dset_dict = plugin.base.getHierarchy(file_path)
        self._populateTree(self.groupTree, self.group_dset_dict)

    def populate_attrs(self, attr_dict=None, editable_values=False):
        """Populate attribute for currently selected group."""
        self.tableAttributes.blockSignals(True) # block udpate signals for auto-filled forms
        self.tableAttributes.setRowCount(0)
        self.tableAttributes.setColumnCount(2)
        self.tableAttributes.setSortingEnabled(False)

        if attr_dict:
            for num, key in enumerate(attr_dict):
                self.tableAttributes.insertRow(self.tableAttributes.rowCount())
                key_item = QTableWidgetItem(key)
                key_item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                self.tableAttributes.setItem(num, 0, key_item)

                val_item = QTableWidgetItem(str(attr_dict[key]))
                if editable_values:
                    val_item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsEnabled)
                else:
                    val_item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                self.tableAttributes.setItem(num, 1, val_item)

        self.tableAttributes.blockSignals(False)

    def update_attrs_to_file(self, item):
        file_path = os.path.join(self.experiment_file_directory, self.experiment_file_name + '.hdf5')
        group_path = plugin.base.getPathFromTreeItem(self.groupTree.selectedItems()[0])

        attr_key = self.tableAttributes.item(item.row(), 0).text()
        attr_val = item.text()

        # update attr in file
        plugin.base.changeAttribute(file_path=file_path,
                                    group_path=group_path,
                                    attr_key=attr_key,
                                    attr_val=attr_val)
        print('Changed attr {} to = {}'.format(attr_key, attr_val))

# %% # # # # # # # # ROI SELECTOR WIDGET # # # # # # # # # # # # # # # # # # #

    def refreshLassoWidget(self, keep_paths=False):
        self.roi_ax.clear()
        init_lasso = False
        if self.roi_image is not None:
            if len(self.roi_mask) > 0:
                newImage = plot_tools.overlayImage(self.roi_image[:, :, self.current_z_slice], self.roi_mask, 0.5, self.colors, z=self.current_z_slice)
            else:
                newImage = self.roi_image[:, :, self.current_z_slice]
            self.roi_ax.imshow(newImage, cmap=cm.gray)
            init_lasso = True
        else:
            self.roi_ax.imshow(self.blank_image)

        self.roi_canvas.draw()

        if not keep_paths:
            self.roi_path_list = []
            self.roi_z_list = []

        if init_lasso:
            if self.roi_type == 'circle':
                self.lasso_1 = EllipseSelector(self.roi_ax, onselect=self.newEllipse, button=1)
            elif self.roi_type == 'freehand':
                self.lasso_1 = LassoSelector(self.roi_ax, onselect=self.newFreehand, button=1)
                self.lasso_2 = LassoSelector(self.roi_ax, onselect=self.appendFreehand, button=3)
            else:
                print('Warning ROI type not recognized. Choose circle or freehand')

    def newFreehand(self, verts):
        new_roi_path = path.Path(verts)
        new_roi_path.z_level = self.zSlider.value()
        new_roi_path.channel = self.current_channel
        self.updateRoiSelection([new_roi_path])

    def appendFreehand(self, verts):
        print('Appending rois, hit Enter/Return to finish')
        new_roi_path = path.Path(verts)
        new_roi_path.z_level = self.zSlider.value()
        new_roi_path.channel = self.current_channel
        self.roi_path_list.append(new_roi_path)

    def keyPressEvent(self, event):
        if type(event) == QtGui.QKeyEvent:
            if np.any([event.key() == QtCore.Qt.Key_Return, event.key() == QtCore.Qt.Key_Enter]):
                if len(self.roi_path_list) > 0:
                    event.accept()
                    self.updateRoiSelection(self.roi_path_list)
                else:
                    event.ignore()
            else:
                event.ignore()
        else:
            event.ignore()

    def newEllipse(self, pos1, pos2, definedRadius=None):
        x1 = np.round(pos1.xdata)
        x2 = np.round(pos2.xdata)
        y1 = np.round(pos1.ydata)
        y2 = np.round(pos2.ydata)

        radiusX = np.sqrt((x1 - x2)**2)/2
        radiusY = np.sqrt((y1 - y2)**2)/2
        if self.roi_radius is not None:
            radiusX = self.roi_radius

        center = (np.round((x1 + x2)/2), np.round((y1 + y2)/2))
        new_roi_path = path.Path.circle(center=center, radius=radiusX)
        new_roi_path.z_level = self.zSlider.value()
        new_roi_path.channel = self.current_channel
        self.updateRoiSelection([new_roi_path])

    def updateRoiSelection(self, new_roi_path):
        mask = self.plugin.getRoiMaskFromPath(new_roi_path)
        new_roi_resp = self.plugin.getRoiDataFromPath(roi_path=new_roi_path)
        if new_roi_resp is None:
            print('No pixels in selected roi')
            return
        # update list of roi data
        self.roi_mask.append(mask)
        self.roi_path.append(new_roi_path)  # list of lists of paths
        self.roi_response.append(new_roi_resp)
        # update slider to show most recently drawn roi response
        self.current_roi_index = len(self.roi_response)-1
        self.roiSlider.setValue(self.current_roi_index)

        # Update figures
        self.redrawRoiTraces()

    def sliderUpdated(self):
        self.current_roi_index = self.roiSlider.value()
        self.redrawRoiTraces()

    def zSliderUpdated(self):
        self.current_z_slice = self.zSlider.value()
        if self.roi_image is not None:
            self.refreshLassoWidget(keep_paths=True)


    def redrawRoiTraces(self):
        self.responsePlot.clear()
        if self.current_roi_index < len(self.roi_response):
            current_raw_trace = np.squeeze(self.roi_response[self.current_roi_index])
            fxn_name = self.RoiResponseTypeComboBox.currentText()
            display_trace = getattr(self.plugin, 'getRoiResponse_{}'.format(fxn_name))([current_raw_trace])
            self.responsePlot.plot(display_trace, color=self.colors[self.current_roi_index], linewidth=1, alpha=0.5)
        self.responseCanvas.draw()

        self.refreshLassoWidget(keep_paths=False)

# %% # # # # # # # # LOADING / SAVING / COMPUTING ROIS # # # # # # # # # # # # # # # # # # #

    def loadRois(self, roi_set_path):
        file_path = os.path.join(self.experiment_file_directory, self.experiment_file_name + '.hdf5')
        self.roi_response, self.roi_image, self.roi_path, self.roi_mask = self.plugin.loadRoiSet(file_path, roi_set_path)
        self.zSlider.setValue(0)
        self.zSlider.setMaximum(self.roi_image.shape[2]-1)


    def saveRois(self):
        file_path = os.path.join(self.experiment_file_directory, self.experiment_file_name + '.hdf5')
        roi_set_name = self.le_roiSetName.text()
        if roi_set_name in plugin.base.getAvailableRoiSetNames(file_path, self.series_number):
            buttonReply = QMessageBox.question(self,
                                               'Overwrite roi set',
                                               "Are you sure you want to overwrite roi set: {}?".format(roi_set_name),
                                               QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if buttonReply == QMessageBox.Yes:
                self.plugin.saveRoiSet(file_path,
                                       series_number=self.series_number,
                                       roi_set_name=roi_set_name,
                                       roi_mask=self.roi_mask,
                                       roi_response=self.roi_response,
                                       roi_image=self.roi_image,
                                       roi_path=self.roi_path)
                print('Saved roi set {} to series {}'.format(roi_set_name, self.series_number))
                self.populateGroups()
                self.updateExistingRoiSetList()
            else:
                print('Overwrite aborted - pick a unique roi set name')
        else:
            self.plugin.saveRoiSet(file_path,
                                   series_number=self.series_number,
                                   roi_set_name=roi_set_name,
                                   roi_mask=self.roi_mask,
                                   roi_response=self.roi_response,
                                   roi_image=self.roi_image,
                                   roi_path=self.roi_path)
            print('Saved roi set {} to series {}'.format(roi_set_name, self.series_number))
            self.populateGroups()
            self.updateExistingRoiSetList()

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
        self.roi_image = None
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

    def selectChannel(self):
        self.current_channel = int(self.ChannelComboBox.currentText())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DataGUI()
    sys.exit(app.exec_())
