#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from visanalysis import protocol_analysis as pa
from visanalysis import plot_tools

class FlickeringPatchAnalysis(pa.ProtocolAnalysis.BaseAnalysis):
    def __init__(self):
        super().__init__() #call the parent class init method first
        
    def initializeAnalysis(self):
        frequency = [];
        for ep in self.ImagingData.epoch_parameters:
            frequency.append(ep['current_temporal_frequency'])
    
        parameter_values = np.array([frequency]).T
        self.unique_parameter_values, self.mean_trace_matrix, self.sem_trace_matrix, self.individual_traces = pa.ProtocolAnalysis.getTraceMatrixByStimulusParameter(self.ImagingData.response_matrix, parameter_values)
    
        self.frequencies = np.unique(np.array(frequency))
    def makeExamplePlots(self, fig_handle = None, eg_trace_ind = 0, export_to_igor_flag = False):
        #plot stuff
        if fig_handle is None:
            fig_handle = plt.figure(figsize=(10,2))
            
        grid = plt.GridSpec(1, len(self.frequencies), wspace=0.2, hspace=0.15)
        fig_axes = fig_handle.get_axes()
        ax_ind = 0
            
        plot_y_max = 1.1
        plot_y_min = -0.2 
        
        for ind_f, f in enumerate(self.frequencies):
            pull_ind = np.where((self.unique_parameter_values == f).all(axis = 1))[0][0]
            ax_ind += 1
            if len(fig_axes) > 1:
                new_ax = fig_axes[ax_ind]
                new_ax.clear()
            else: 
                new_ax = fig_handle.add_subplot(grid[0,ind_f])
                
            plot_tools.addLine(new_ax, self.ImagingData.time_vector, self.mean_trace_matrix[eg_trace_ind,pull_ind,:].T, line_name = 'mean', color = self.plot_colors[eg_trace_ind])
            plot_tools.addErrorBars(new_ax, self.ImagingData.time_vector, np.squeeze(self.individual_traces[pull_ind][eg_trace_ind]),
                                line_name = 'err', stat = 'sem', mode = 'snake', color = self.plot_colors[eg_trace_ind])
            
            new_ax.set_ylim([plot_y_min, plot_y_max])
            new_ax.set_axis_off()
            new_ax.set_title(int(f))
            if ind_f == 0: # scale bar
                plot_tools.addScaleBars(new_ax, 1, 0.5, F_value = -0.2, T_value = -0.2)
                
            if export_to_igor_flag:
                name_suffix = self.ImagingData.getAxisStructureNameSuffix()
                plot_tools.makeIgorStructure(new_ax, file_name = 'FP_trace_' + str(int(f)) + name_suffix, subdirectory = 'LC_RFs')
                
        fig_handle.canvas.draw()
