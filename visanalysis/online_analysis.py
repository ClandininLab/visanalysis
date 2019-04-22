#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib.widgets import LassoSelector
from matplotlib import path
import matplotlib.pyplot as plt
import imaging_data
import os
from pyqtgraph.widgets.MatplotlibWidget import MatplotlibWidget
import socket
import protocol_analysis as pa


import sys
from PyQt5.QtWidgets import (QPushButton, QWidget, QGridLayout, QApplication, QComboBox, QToolBar)


class OnlineAnalysisGUI(QWidget):

    def __init__(self):
        super().__init__()
        self.protocolID = None
        self.fastZStackFlag = False
        self.rawSeries = None
        self.registeredSeries = None
        if socket.gethostname() == "MHT-laptop": # (laptop, for dev.)
            self.dataDirectory = '/Users/mhturner/Dropbox/ClandininLab/CurrentData/'
            self.flyStimDataDirectory = self.dataDirectory + 'FlystimData/'
        else: #TODO: specify hostname for bruker computer
            self.dataDirectory = 'E:/Max/ForTransfer/'
            self.flyStimDataDirectory = 'E:/Max/FlystimData/'
        
        # get flystim data files in the flyStimDataDirectory
        dataFolderContents = os.listdir(self.flyStimDataDirectory)
        self.flystimDataFileNames = [s for s in dataFolderContents if "hdf5" in s]
        self.flystimFileName = self.flystimDataFileNames[-1].split('.')[0] # default is most recent

        self.updateImageDirectory()

        self.initUI()
  
    def initUI(self):  
        self.grid = QGridLayout()
        self.grid.setColumnStretch(1, 1)
        
        # # Data file drop-down # #
        df_comboBox = QComboBox(self)
        for newFileName in reversed(self.flystimDataFileNames):
            df_comboBox.addItem(newFileName.split('.')[0])
        df_comboBox.activated[str].connect(self.onSelectedDataFile)
        df_comboBox.setMaximumSize(300,100)
        self.grid.addWidget(df_comboBox, 1, 0)

        # Time series ID drop-down:
        self.ts_comboBox = None
        self.updateTimeSeriesList()
        
        # Do analysis button:
        doAnalysisButton = QPushButton("Do analysis", self)
        doAnalysisButton.clicked.connect(self.onPressedButton) 
        doAnalysisButton.setMaximumSize(300,100)
        self.grid.addWidget(doAnalysisButton, 3, 0)
        
        # Registration button:
        registrationButton = QPushButton("Do Registration", self)
        registrationButton.clicked.connect(self.onPressedButton) 
        registrationButton.setMaximumSize(150,100)
        self.grid.addWidget(registrationButton, 4, 0)

        # Roi image canvas
        self.roi_canvas = MatplotlibWidget()
        self.grid.addWidget(self.roi_canvas, 5, 0)
        win = self.roi_canvas.window()
        toolbar = win.findChild(QToolBar)
        toolbar.setVisible(False)
        
        # Results canvas
        self.results_canvas = MatplotlibWidget()
        self.grid.addWidget(self.results_canvas, 6, 0)
        win = self.results_canvas.window()
        toolbar = win.findChild(QToolBar)
        toolbar.setVisible(False)

        self.setLayout(self.grid) 
        self.setGeometry(200, 200, 600, 800)
        self.setWindowTitle('Online Analysis')    
        self.show()
        
    def onSelectedDataFile(self,text):
        self.flystimFileName = text
        self.updateImageDirectory()
        self.updateTimeSeriesList()
        
    def onSelectedTimeSeries(self, text):
        if text == "*select a time series*":
            return
        else:
            self.seriesString = text
            series_number = int(self.seriesString.split('-')[-1])
            dateStr = self.seriesString.split('-')[1]
            file_name = dateStr[:4] + '-' + dateStr[4:6] + '-' + dateStr[6:]
            self.ImagingData = imaging_data.ImagingDataObject(file_name, series_number, 
                                       data_directory = self.dataDirectory, 
                                       flystim_data_directory = self.flyStimDataDirectory)
            self.ImagingData.loadImageSeries()

    def onPressedButton(self):
        sender = self.sender()
        if sender.text() == 'Do Registration':
            
            self.ImagingData.registerStack()

        elif sender.text() == 'Do analysis':
            self.roi_canvas.fig.clear()
            self.results_canvas.fig.clear()
            LassoROI(self.ImagingData, roi_canvas = self.roi_canvas, results_canvas = self.results_canvas)
            
        self.roi_canvas.draw()
        self.results_canvas.draw()
        
    def updateTimeSeriesList(self):
        if self.ts_comboBox is not None:
            self.ts_comboBox.clear()
        self.ts_comboBox = QComboBox(self)
        self.ts_comboBox.addItem("*select a time series*")
        # Add available time series from directory
        imageFolderContents = os.listdir(self.imageDirectory)
        #uses Bruker .env file to detect available series names
        symphonyDataFileNames = [s for s in imageFolderContents if "env" in s]
        for newFileName in symphonyDataFileNames:
            self.ts_comboBox.addItem(newFileName.split('.')[0])
        self.ts_comboBox.setMaximumSize(300,100)
        self.ts_comboBox.activated[str].connect(self.onSelectedTimeSeries)
        self.grid.addWidget(self.ts_comboBox, 2, 0)

    def updateImageDirectory(self):
        self.imageDirectory = self.dataDirectory + self.flystimFileName.replace('-','') + '/'
        
        
class LassoROI:
    def updateArray(self,indices):
        array = np.zeros((self.yDim, self.xDim))
        lin = np.arange(array.size)
        newArray = array.flatten()
        newArray[lin[indices]] = 1
        return newArray.reshape(array.shape)
    
    def onselect(self,verts):
        global array, pix
        p = path.Path(verts)
        ind = p.contains_points(self.pix, radius=1)
        self.roiPoints = []
        self.roiPoints = self.updateArray(ind)
        self.roiPoints = self.roiPoints == 1 #convert to boolean for masking

        roiResp = (np.mean(self.currentSeries[:,self.roiPoints], axis = 1, keepdims=True) -
                   np.min(self.currentSeries)).T
            
        #Prep response matrix and roi response
        self.ImagingData.getResponseTraces(roiResp)
        
        # Call doOnlineAnalysis for protocol analysis class
        analysis_module_name = self.ImagingData.run_parameters['protocol_ID']
        #Init analysis object:
        analysis_object=getattr(getattr(pa,analysis_module_name), analysis_module_name+'Analysis')()
        #Do online analysis:
        self.results_canvas.fig.clear()
        analysis_object.doOnlineAnalysis(ImagingData = self.ImagingData, fig_handle = self.results_fig)
        

    def __init__(self,ImagingData, roi_canvas = None, results_canvas = None):
        self.ImagingData = ImagingData
        self.currentSeries = self.ImagingData.current_series

        self.roiImage = np.squeeze(np.mean(self.currentSeries, axis = 0))
        self.xDim = self.roiImage.shape[1]
        self.yDim = self.roiImage.shape[0]

        if roi_canvas is None:
            self.roi_fig = plt.figure()
        else:
            self.roi_canvas = roi_canvas
            self.roi_fig = self.roi_canvas.getFigure()
            
        if results_canvas is None:
            self.results_fig = plt.figure()
        else:
            self.results_canvas = results_canvas
            self.results_fig = self.results_canvas.getFigure()
            
        ax1 = self.roi_fig.add_subplot(111)
        ax1.imshow(self.roiImage)
        ax1.set_aspect('equal')
        ax1.set_axis_off
        ax1.set_title(self.ImagingData.run_parameters['protocol_ID'])
        
        # Empty array to be filled with lasso selector
        self.roiPoints = np.zeros((self.currentSeries.shape[1], self.currentSeries.shape[2]))
        
        # Pixel coordinates
        pixX = np.arange(self.roiImage.shape[1])
        pixY = np.arange(self.roiImage.shape[0])
        yv, xv = np.meshgrid(pixX, pixY)
        self.pix = np.vstack((yv.flatten(), xv.flatten())).T
        self.lasso = LassoSelector(ax1, self.onselect)
        if roi_canvas is None:
            plt.show()
        else:
            self.roi_canvas.draw()
            self.results_canvas.draw()

if __name__ == '__main__':
    
    app = QApplication(sys.argv)
    ex = OnlineAnalysisGUI()
    sys.exit(app.exec_())

