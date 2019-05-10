#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 13:15:58 2018

@author: mhturner
"""
import numpy as np
import matplotlib.cm as cm

import pyqtgraph as pg
from pyqtgraph.widgets.MatplotlibWidget import MatplotlibWidget
from pyqtgraph.widgets.PlotWidget import PlotWidget
from PyQt5.QtWidgets import (QPushButton, QWidget, QGridLayout, QLineEdit, QComboBox)
from matplotlib import path
from matplotlib.widgets import LassoSelector
from matplotlib.widgets import EllipseSelector

from visanalysis import plot_tools


class MultiROISelector(QWidget):
    
    def __init__(self, ImagingData, roiType = 'freehand', roiRadius = None):
        super().__init__()
        self.ImagingData = ImagingData
        self.roiType = roiType
        self.roiRadius = roiRadius #if set to a number, will produce pre-defined size ROI where user places ROI

        self.maxRois = 8

        self.plotWidgets = []
        
        self.xDim = self.ImagingData.roi_image.shape[1]
        self.yDim = self.ImagingData.roi_image.shape[0]
        self.colors = self.ImagingData.colors
        
        self.initUI()

    def initUI(self):  
        self.grid = QGridLayout()
        self.grid.setSpacing(3)

        self.refreshLassoWidget(self.ImagingData.roi_image)
        
        # Traces for seleted ROIs and delete button for each
        for tt in range(self.maxRois):
            newDeleteButton = QPushButton("Delete ROI_" + str(tt), self)
            newDeleteButton.clicked.connect(self.onPressedDeleteButton)
            self.grid.addWidget(newDeleteButton, tt, 5)
            
            newPlotWidget = PlotWidget()
            self.plotWidgets.append(newPlotWidget)
            self.grid.addWidget(newPlotWidget, tt, 6, 1, 5)

        # Clear all ROIs button
        self.clearROIsButton = QPushButton("Clear ROIs", self)
        self.clearROIsButton.clicked.connect(self.onPressedClearRoisButton) 
        self.grid.addWidget(self.clearROIsButton, 10, 0)
        
        # ROI type drop-down
        self.RoiTypeComboBox = QComboBox(self)
        self.RoiTypeComboBox.addItem("freehand")
        radii = [1, 2, 3, 4, 6, 8]
        for radius in radii:
            self.RoiTypeComboBox.addItem("circle:"+str(radius))
        self.RoiTypeComboBox.activated.connect(self.onSelectedRoiType)
        self.grid.addWidget(self.RoiTypeComboBox, 9, 0)
        
        # Save ROIs button
        self.saveROIsButton = QPushButton("Save ROIs", self)
        self.saveROIsButton.clicked.connect(self.onPressedSaveRoisButton) 
        self.grid.addWidget(self.saveROIsButton, 10, 1)
        
        # ROIset file name line edit box
        self.defaultRoiSetName = "roi_set_name"
        self.le_roiSetName = QLineEdit(self.defaultRoiSetName)
        self.grid.addWidget(self.le_roiSetName, 11, 1)
        
        # Available ROIs combobox
        self.RoiComboBox = QComboBox(self)
        self.RoiComboBox.addItem("(select an ROI set)")
        for roi_set in self.ImagingData.getAvailableROIsets():
            self.RoiComboBox.addItem(roi_set)
        self.grid.addWidget(self.RoiComboBox, 9, 2)
        
        # Load ROIs button
        self.loadROIsButton = QPushButton("Load ROI set", self)
        self.loadROIsButton.clicked.connect(self.onPressedLoadRoisButton) 
        self.grid.addWidget(self.loadROIsButton, 10, 2)

        self.setLayout(self.grid) 
        self.setGeometry(100, 100, 1200, 200)
        self.setWindowTitle('Multi-ROI selector')    
        
        self.redrawRoiTraces()
        
        self.show()

    def getNewROI(self,indices):
        array = np.zeros((self.yDim, self.xDim))
        lin = np.arange(array.size)
        newArray = array.flatten()
        newArray[lin[indices]] = 1
        return newArray.reshape(array.shape)
    
    def onselectFreehand(self,verts):
        global array, pix
        new_roi_path = path.Path(verts)
        ind = new_roi_path.contains_points(self.pix, radius=1)
        newRoiArray = self.getNewROI(ind)
        newRoiArray = newRoiArray == 1 #convert to boolean for masking

        newRoiResp = (np.mean(self.ImagingData.current_series[:,newRoiArray], axis = 1, keepdims=True) -
                   np.min(self.ImagingData.current_series)).T
        # Update list of ROIs and ROI responses
        self.newRoiResp = newRoiResp;
        self.ImagingData.roi_mask.append(newRoiArray)
        self.ImagingData.roi_response.append(newRoiResp)
        self.ImagingData.roi_path.append(new_roi_path)
        
        # Update figures
        self.redrawRoiTraces()

    def onselectEllipse(self,pos1,pos2,definedRadius = None):
        global array, pix
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
        ind = new_roi_path.contains_points(self.pix, radius=0.5)

        newRoiArray = self.getNewROI(ind)
        newRoiArray = newRoiArray == 1 #convert to boolean for masking

        newRoiResp = (np.mean(self.ImagingData.current_series[:,newRoiArray], axis = 1, keepdims=True) -
                   np.min(self.ImagingData.current_series)).T
        # Update list of ROIs and ROI responses
        self.newRoiResp = newRoiResp;
        self.ImagingData.roi_mask.append(newRoiArray)
        self.ImagingData.roi_response.append(newRoiResp)
        self.ImagingData.roi_path.append(new_roi_path)

        # Update figures
        self.redrawRoiTraces()
        
    def onSelectedRoiType(self):
        self.roiType = self.RoiTypeComboBox.currentText().split(':')[0]
        if 'circle' in self.RoiTypeComboBox.currentText():
            self.roiRadius = int(self.RoiTypeComboBox.currentText().split(':')[1])
        else:
            self.roiRadius = None
        self.redrawRoiTraces()
        
    def onPressedDeleteButton(self):
        roiIndex = int(self.sender().text().split('_')[1])
        self.ImagingData.roi_mask.pop(roiIndex)
        self.ImagingData.roi_response.pop(roiIndex)
        self.ImagingData.roi_path.pop(roiIndex)
        self.redrawRoiTraces()
        
    def onPressedClearRoisButton(self):
        self.ImagingData.roi_mask = []
        self.ImagingData.roi_response = []
        self.ImagingData.roi_path = []
        for roiIndex in range(self.maxRois):
            self.plotWidgets[roiIndex].clear()
            
        self.redrawRoiTraces()
        
    def onPressedSaveRoisButton(self):
        self.ImagingData.saveRois(roi_set_name = self.le_roiSetName.text())

    def onPressedLoadRoisButton(self):
        self.ImagingData.loadRois(str(self.RoiComboBox.currentText()))
        
        self.redrawRoiTraces()

    def redrawRoiTraces(self):
        for roiIndex in range(self.maxRois):
            self.plotWidgets[roiIndex].clear()
            if roiIndex < len(self.ImagingData.roi_response):
                penStyle = pg.mkPen(color = tuple([255*x for x in self.colors[roiIndex]]))
                self.plotWidgets[roiIndex].plot(np.squeeze(self.ImagingData.roi_response[roiIndex].T), pen=penStyle )
        
        if len(self.ImagingData.roi_mask) > 0:
            newImage = plot_tools.overlayImage(self.ImagingData.roi_image, self.ImagingData.roi_mask , 0.5, self.colors)
        else:
            newImage = self.ImagingData.roi_image
        self.refreshLassoWidget(newImage)
        self.show()
 
    def refreshLassoWidget(self, image):
        # Image canvas for image and lasso widget
        self.imageCanvas = MatplotlibWidget()
        self.grid.addWidget(self.imageCanvas, 0, 0, 8, 4)

        fig = self.imageCanvas.getFigure();
        ax1 = fig.add_subplot(1,1,1)
        ax1.imshow(image, cmap = cm.gray)
        ax1.set_aspect('equal')
        ax1.set_axis_off()
        
        # Pixel coordinates of lasso selector
        pixX = np.arange(self.ImagingData.roi_image.shape[1])
        pixY = np.arange(self.ImagingData.roi_image.shape[0])
        yv, xv = np.meshgrid(pixX, pixY)
        self.pix = np.vstack((yv.flatten(), xv.flatten())).T
        if self.roiType == 'circle':
            self.lasso = EllipseSelector(ax1, self.onselectEllipse)
        elif self.roiType == 'freehand':
            self.lasso = LassoSelector(ax1, self.onselectFreehand)
        else:
            print('Warning ROI type not recognized. Choose circle or freehand')
        
        
        