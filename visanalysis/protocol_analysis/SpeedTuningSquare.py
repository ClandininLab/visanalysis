#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from visanalysis import protocol_analysis as pa
from visanalysis import plot_tools

class SpeedTuningSquareAnalysis(pa.ProtocolAnalysis.BaseAnalysis):
    def __init__(self):
        super().__init__() #call the parent class init method first
    
    def initializeAnalysis(self):
        speed = [];
        for ep in self.ImagingData.epoch_parameters:
            speed.append(ep['currentSpeed'])
    
        parameter_values = np.array([speed]).T
        self.unique_parameter_values, self.mean_trace_matrix, self.sem_trace_matrix, self.individual_traces = pa.ProtocolAnalysis.getTraceMatrixByStimulusParameter(self.ImagingData.response_matrix, parameter_values)

        self.speeds = np.unique(np.array(speed))

    def makeExamplePlots(self, fig_handle = None, eg_trace_ind = 0):
        if fig_handle is None:
            fig_handle = plt.figure(figsize=(8,4))
        fig_axes = fig_handle.get_axes()
        ax_ind = 0
        plot_y_max = 1.1*np.max(self.mean_trace_matrix)
        plot_y_min = 1.5*np.min(self.mean_trace_matrix)    
        
        grid = plt.GridSpec(2, len(self.speeds), wspace=0.2, hspace=0.15)
        for ind_s, s in enumerate(self.speeds):
            pull_ind = np.where((self.unique_parameter_values == s).all(axis = 1))[0][0]
            
            ax_ind += 1
            if len(fig_axes) > 1:
                new_ax = fig_axes[ax_ind]
                new_ax.clear()
            else: 
                new_ax = fig_handle.add_subplot(grid[0, ind_s])
                
            new_ax.plot(self.ImagingData.time_vector, self.mean_trace_matrix[:,pull_ind,:].T)
            new_ax.set_ylim([plot_y_min, plot_y_max])
            new_ax.set_axis_off()
            new_ax.set_title(int(s))
            if ind_s == 0: # scale bar
                plot_tools.addScaleBars(new_ax, 1, 1, F_value = -0.4, T_value = -0.2)
        

        response_peaks = np.max(self.mean_trace_matrix, axis = 2).T
        response_peaks_norm = response_peaks.T / response_peaks.max(axis = 0)[:,np.newaxis]
        new_ax = fig_handle.add_subplot(grid[1,4:len(self.speeds)])
        new_ax.plot(self.speeds,response_peaks_norm.T)
        new_ax.set_ylabel('Peak dF/F (norm)')
        new_ax.set_xlabel('Speed (deg./s.)')
        new_ax.set_ylim([0, 1])
        
        fig_handle.canvas.draw()
