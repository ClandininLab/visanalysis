#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from visanalysis import protocol_analysis as pa
from visanalysis import plot_tools

class CenterSurroundDriftingSquareGratingAnalysis(pa.ProtocolAnalysis.BaseAnalysis):
    def __init__(self):
        super().__init__() #call the parent class init method first
        
    def initializeAnalysis(self):
        center_rate = []
        surround_rate = []
        for ep in self.ImagingData.epoch_parameters:
            center_rate.append(ep['current_center_rate'])
            surround_rate.append(ep['current_surround_rate'])
    
        parameter_values = np.array([center_rate,surround_rate]).T
        self.unique_parameter_values, self.mean_trace_matrix, self.sem_trace_matrix, self.individual_traces = pa.ProtocolAnalysis.getTraceMatrixByStimulusParameter(self.ImagingData.response_matrix, parameter_values)

        self.center_rates = np.unique(np.array(center_rate))
        self.surround_rates = np.unique(np.array(surround_rate))

    def makeExamplePlots(self, fig_handle = None, export_to_igor_flag = False):
        if fig_handle is None:
            fig_handle = plt.figure(figsize=(8,4))
        fig_axes = fig_handle.get_axes()
        ax_ind = 0
        plot_y_max = 3
        plot_y_min = -0.5

        grid = plt.GridSpec(len(self.center_rates), len(self.surround_rates), wspace=0.2, hspace=0.15)
        for ind_s, s in enumerate(self.surround_rates):
            for ind_c, c in enumerate(self.center_rates):
                surround_inds = np.where(self.unique_parameter_values[:,1]==s)[0]
                center_inds = np.where(self.unique_parameter_values[:,0]==c)[0]
                pull_ind = np.intersect1d(center_inds,surround_inds)[0]
                current_mean = self.mean_trace_matrix[:,pull_ind,:]
                current_traces = self.individual_traces[pull_ind]
            
                ax_ind += 1
                if len(fig_axes) > 1:
                    new_ax = fig_axes[ax_ind]
                    new_ax.clear()
                else: 
                    new_ax = fig_handle.add_subplot(grid[ind_c, ind_s])
                
                new_ax.plot(self.ImagingData.time_vector, current_mean.T, 'k', 'LineWidth', 2)
#                new_ax.plot(self.ImagingData.time_vector, current_traces[0,:,:].T, alpha = 0.3)
                
                new_ax.set_ylim([plot_y_min, plot_y_max])
                new_ax.set_xticklabels([])
                new_ax.set_yticklabels([])
                plt.setp(new_ax.spines.values(), linewidth=0)
                new_ax.xaxis.set_tick_params(width=0)
                new_ax.yaxis.set_tick_params(width=0)
                if ind_c == len(self.center_rates)-1:
                    new_ax.set_xlabel(str(s))
                if ind_s == 0:
                    new_ax.set_ylabel(str(c))
                if ind_s == 0 and ind_c == 0: # scale bar
                    plot_tools.addScaleBars(new_ax, 1, 1, F_value = -0.2, T_value = -0.2)
        
        fig_handle.canvas.draw()
        
    def getSummaryStatistics(self):
        response_peaks = np.max(self.mean_trace_matrix, axis = 2).T
        summary_statistics = {'response_peaks':response_peaks}
        
        return summary_statistics
