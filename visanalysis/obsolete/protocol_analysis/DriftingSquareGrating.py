#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from visanalysis import protocol_analysis as pa
from visanalysis import plot_tools

class DriftingSquareGratingAnalysis(pa.ProtocolAnalysis.BaseAnalysis):
    def __init__(self):
        super().__init__() #call the parent class init method first

    def initializeAnalysis(self, roi_name):
        self.roi_name = roi_name
        orientation = [];
        for ep in self.ImagingData.epoch_parameters:
            orientation.append(ep['angle'])
    
        parameter_values = np.array([orientation]).T
        self.unique_parameter_values, self.mean_trace_matrix, self.sem_trace_matrix, self.individual_traces = pa.ProtocolAnalysis.getTraceMatrixByStimulusParameter(self.ImagingData.roi.get(self.roi_name).get('epoch_response'), parameter_values)
    
        self.orientations = np.unique(np.array(orientation))
        
    def makeExamplePlots(self, fig_handle = None, eg_trace_ind = 0, export_to_igor_flag = False):
        #plot stuff
        if fig_handle is None:
            fig_handle = plt.figure(figsize=(10,2))
            
        grid = plt.GridSpec(1, len(self.orientations), wspace=0.2, hspace=0.15)
        fig_axes = fig_handle.get_axes()
        ax_ind = 0
            
        plot_y_max = 1.1*np.max(self.mean_trace_matrix)
        plot_y_min = 1*np.min(self.mean_trace_matrix)    
        
        for ind_o, o in enumerate(self.orientations):
            pull_ind = np.where((self.unique_parameter_values == o).all(axis = 1))[0][0]
            ax_ind += 1
            if len(fig_axes) > 1:
                new_ax = fig_axes[ax_ind]
                new_ax.clear()
            else: 
                new_ax = fig_handle.add_subplot(grid[0,ind_o])
            
            new_ax.plot(self.ImagingData.roi.get(self.roi_name).get('time_vector'), self.mean_trace_matrix[:,pull_ind,:].T)
            new_ax.set_ylim([plot_y_min, plot_y_max])
            new_ax.set_axis_off()
            new_ax.set_title(int(o))
            if ind_o == 0: # scale bar
                plot_tools.addScaleBars(new_ax, 1, 1, F_value = -0.1, T_value = -0.2)
        fig_handle.canvas.draw()
        
        
        