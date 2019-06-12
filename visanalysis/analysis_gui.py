#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from PyQt5.QtWidgets import (QPushButton, QWidget, QGridLayout, QApplication,
                             QComboBox, QLabel, QFrame, QListWidget,
                             QListWidgetItem, QCheckBox, QLineEdit, QInputDialog)
import PyQt5.QtCore as QtCore
import os
import inspect
import sys
import yaml

from visanalysis import protocol_analysis as pa
from visanalysis import region
import visanalysis

# TODO:flexible filtering window. For arbitrary params/metadata (param values, date, fly info etc etc)
# TODO: select a series to do quick-view analysis
# TODO: highlight series w/o named roi, button to draw roi 

class AnalysisGUI(QWidget):
    
    def __init__(self):
        super().__init__()
        
        # Import configuration settings
        path_to_config_file = os.path.join(inspect.getfile(visanalysis).split('visanalysis')[0], 'visanalysis', 'config', 'config.yaml')
        with open(path_to_config_file, 'r') as ymlfile:
            cfg = yaml.safe_load(ymlfile)

        envs = cfg.get('analysis_settings').keys()
        analysis_id, ok = QInputDialog.getItem(self, "select analysis settings", 
                                             "Available analysis settings", envs, 0, False)
        self.analysis_settings = cfg.get('analysis_settings').get(analysis_id)
        
        if self.analysis_settings.get('rig') == 'AODscope':
            self.imaging_data_type = 'AodScopeData'
        elif self.analysis_settings.get('rig') == 'Bruker':
            self.imaging_data_type = 'BrukerData'
        else:
            self.imaging_data_type = 'ImagingData'
        
        self.ProtocolAnalysisObject = pa.ProtocolAnalysis.BaseAnalysis()

        self.initUI()
                
    def initUI(self):
        self.grid = QGridLayout()
        self.grid.setColumnStretch(1, 1)
        
         # # Protocol drop-down # #
        self.protocol_comboBox = QComboBox(self)
        for new_protocol in self.ProtocolAnalysisObject.available_protocols:
            self.protocol_comboBox.addItem(new_protocol.split('.')[0])
        self.protocol_comboBox.activated[str].connect(self.updateFilterResults)
        self.protocol_comboBox.setMaximumSize(300,100)
        self.grid.addWidget(self.protocol_comboBox, 1, 0)
        
        # # Driver drop-down # #
        self.driver_comboBox = QComboBox(self)
        for new_driver in self.ProtocolAnalysisObject.available_drivers:
            self.driver_comboBox.addItem(new_driver.split('.')[0])
        self.driver_comboBox.activated[str].connect(self.updateFilterResults)
        self.driver_comboBox.setMaximumSize(300,100)
        self.grid.addWidget(self.driver_comboBox, 2, 0)
        
        # # Indicator drop-down # #
        self.indicator_comboBox = QComboBox(self)
        for new_indicator in self.ProtocolAnalysisObject.available_indicators:
            self.indicator_comboBox.addItem(new_indicator.split('.')[0])
        self.indicator_comboBox.activated[str].connect(self.updateFilterResults)
        self.indicator_comboBox.setMaximumSize(300,100)
        self.grid.addWidget(self.indicator_comboBox, 3, 0)

        # # n for current filter selections # #
        self.n_label = QLabel()
        self.n_label.setFrameShadow(QFrame.Shadow(1))
        self.grid.addWidget(self.n_label, 9, 1)
        
        # # filter results combobox for example cell # #
        self.series_label = QLabel('Example series:')
        self.grid.addWidget(self.series_label, 7, 0)
        self.example_combobox = QComboBox(self)
        self.grid.addWidget(self.example_combobox, 8, 0)
        
        # # filter results list # # 
        self.pop_label = QLabel('Population series:')
        self.grid.addWidget(self.pop_label, 9, 0)
        self.filter_results_list = QListWidget(self)
        self.filter_results_list.itemClicked.connect(self.onClickedItem)
        self.grid.addWidget(self.filter_results_list, 10, 0)
        
        self.updateFilterResults()
        
        # # Editable combobox for roi set name # #
        self.roi_set_name_box = QComboBox()
        self.roi_set_name_box.setEditable(True)
        for roi_name in self.analysis_settings.get('roi_names'):
            self.roi_set_name_box.addItem(roi_name)
        self.grid.addWidget(self.roi_set_name_box, 1, 1)

        # # True/False checkbox for igor export # # 
        self.igor_export_checkbox = QCheckBox("Igor export")
        self.grid.addWidget(self.igor_export_checkbox, 5, 1)
        
        # # Button for example cell analysis # # 
        self.example_analysis_button = QPushButton("Do example", self)
        self.example_analysis_button.clicked.connect(self.doExampleAnalysis) 
        self.example_analysis_button.setMaximumSize(150,100)
        self.grid.addWidget(self.example_analysis_button, 6, 1)

        # # Button for population analysis # #
        self.population_analysis_button = QPushButton("Do population", self)
        self.population_analysis_button.clicked.connect(self.doPopulationAnalysis) 
        self.population_analysis_button.setMaximumSize(150,100)
        self.grid.addWidget(self.population_analysis_button, 7, 1)
        
        self.setLayout(self.grid) 
        self.setGeometry(200, 200, 400, 600)
        self.setWindowTitle('Analysis GUI')    
        self.show()
        
    def updateFilterResults(self):
        protocol_id = self.protocol_comboBox.currentText()
        kwargs = {} #TODO flexible filtering here
        self.ProtocolAnalysisObject = getattr(getattr(pa,protocol_id),protocol_id+'Analysis')()
        self.ProtocolAnalysisObject.getTargetFileNames(driver = self.driver_comboBox.currentText(),
                                               protocol_ID = self.protocol_comboBox.currentText(),
                                               indicator = self.indicator_comboBox.currentText(), **kwargs)
        self.updateN()
        
        #update list & combobox of filter results
        self.filter_results_list.clear()
        self.example_combobox.clear()
        for series_ind in range(len(self.ProtocolAnalysisObject.target_file_names)):
            file_name = self.ProtocolAnalysisObject.target_file_names[series_ind]
            series_number = self.ProtocolAnalysisObject.target_series_numbers[series_ind]
            print(file_name)
            print(series_number)
            
            #list
            item = QListWidgetItem(self.filter_results_list)
            item.file_name = file_name
            item.series_number = series_number
            item.setToolTip('Click to define ROIs')
            
            ImagingData = getattr(getattr(visanalysis.imaging_data,self.imaging_data_type),self.imaging_data_type+'Object')(file_name, series_number)
            
            roisets = ImagingData.roi.keys()
            if self.roi_set_name_box.text() not in roisets:
                item.setBackground(QtCore.Qt.red)
            ch = QCheckBox(file_name + ',' + str(series_number))
            ch.setChecked(True) #Default to checked
            self.filter_results_list.setItemWidget(item, ch)
            
            #combobox
            self.example_combobox.addItem(file_name + ',' + str(series_number))
            
    def onClickedItem(self, list_item): #pick ROIs
        ImagingData = ID.ImagingDataObject(list_item.file_name, list_item.series_number)
        ImagingData.loadImageSeries()
        MRS = region.MultiROISelector(ImagingData, roiType = 'freehand', roiRadius = 2)

    def getSelectedSeries(self):
        self.selected_file_names = []
        self.selected_series_numbers = []
        for ind in range(self.filter_results_list.count()):
            check_box = self.filter_results_list.itemWidget(self.filter_results_list.item(ind))
            if check_box.checkState():
                self.selected_file_names.append(check_box.text().split(',')[0])
                self.selected_series_numbers.append(int(check_box.text().split(',')[1]))

    def updateN(self):
        self.n = len(self.ProtocolAnalysisObject.target_file_names)
        self.n_label.setText('n = ' + str(self.n))
        
    def doPopulationAnalysis(self):
        self.getSelectedSeries()
        
        self.ProtocolAnalysisObject.doPopulationAnalysis(file_names = self.selected_file_names,
                                                 series_numbers = self.selected_series_numbers,
                                                 roi_set_name = self.roi_set_name_box.text(),
                                                 export_to_igor_flag = self.igor_export_checkbox.checkState())
    def doExampleAnalysis(self):
        self.ProtocolAnalysisObject.doExampleAnalysis(file_name = self.example_combobox.currentText().split(',')[0], 
                                                    series_number = self.example_combobox.currentText().split(',')[1], 
                                                    roi_set_name = self.roi_set_name_box.text(),
                                                    export_to_igor_flag = self.igor_export_checkbox.checkState())
        
    def doQuickAnalysis(self):
        pass #TODO
        
        
if __name__ == '__main__':
    
    app = QApplication(sys.argv)
    ex = AnalysisGUI()
    sys.exit(app.exec_())
