#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from visanalysis import protocol_analysis as pa
from visanalysis import plot_tools

class DriftingVsStationaryGratingAnalysis(pa.ProtocolAnalysis.BaseAnalysis):
    def __init__(self):
        super().__init__() #call the parent class init method first
        
    def initializeAnalysis(self):
        stationary_code = []
        temporal_frequency = []
        for ep in self.ImagingData.epoch_parameters:
            stationary_code.append(ep['current_stationary_code'])
            temporal_frequency.append(ep['current_temporal_frequency'])
    
        parameter_values = np.array([stationary_code,temporal_frequency]).T
        self.unique_parameter_values, self.mean_trace_matrix, self.sem_trace_matrix, self.individual_traces = pa.ProtocolAnalysis.getTraceMatrixByStimulusParameter(self.ImagingData.response_matrix, parameter_values)

        self.stationary_codes = np.unique(np.array(stationary_code))
        self.temporal_frequencies = np.unique(np.array(temporal_frequency))
        
        self.drifting_inds = np.where(self.unique_parameter_values[:,0] == 1)[0]
        self.stationary_inds = np.where(self.unique_parameter_values[:,0] == 0)[0]

    def makeExamplePlots(self, fig_handle = None, eg_trace_ind = 0, export_to_igor_flag = False):
        if fig_handle is None:
            fig_handle = plt.figure(figsize=(8,4))
        fig_axes = fig_handle.get_axes()
        ax_ind = 0
        plot_y_max = 3
        plot_y_min = -0.5

        grid = plt.GridSpec(2, len(self.temporal_frequencies), wspace=0.2, hspace=0.15)
        for ind_f, f in enumerate(self.temporal_frequencies):
            f_ind = np.where(self.unique_parameter_values[:,1] == f)[0]

            drifting_mean = self.mean_trace_matrix[:,np.intersect1d(f_ind, self.drifting_inds)[0],:].T
            drifting_sem= self.sem_trace_matrix[eg_trace_ind,np.intersect1d(f_ind, self.drifting_inds)[0],:].T
            drifting_traces = self.individual_traces[np.intersect1d(f_ind, self.drifting_inds)[0]]
            
            stationary_mean = self.mean_trace_matrix[:,np.intersect1d(f_ind, self.stationary_inds)[0],:].T
            stationary_sem= self.sem_trace_matrix[eg_trace_ind,np.intersect1d(f_ind, self.stationary_inds)[0],:].T
            stationary_traces = self.individual_traces[np.intersect1d(f_ind, self.stationary_inds)[0]]
            
            ax_ind += 1
            if len(fig_axes) > 1:
                new_ax = fig_axes[ax_ind]
                new_ax.clear()
            else: 
                new_ax = fig_handle.add_subplot(grid[0, ind_f]) 
                
            new_ax.plot(self.ImagingData.time_vector, drifting_mean, 'g', 'LineWidth',2)
#            new_ax.plot(self.ImagingData.time_vector, np.squeeze(drifting_traces.T), 'k', 'LineWidth',1, alpha = 0.3)
            new_ax.set_ylim([plot_y_min, plot_y_max])
            new_ax.set_xticklabels([])
            new_ax.set_yticklabels([])
            plt.setp(new_ax.spines.values(), linewidth=0)
            new_ax.xaxis.set_tick_params(width=0)
            new_ax.yaxis.set_tick_params(width=0)
            new_ax.set_title(f)
            if ind_f == 0: # scale bar
                plot_tools.addScaleBars(new_ax, 1, 1, F_value = -0.4, T_value = -0.2)
            
            new_ax = fig_handle.add_subplot(grid[1, ind_f])
            new_ax.plot(self.ImagingData.time_vector, stationary_mean, 'k', 'LineWidth',2)
#            new_ax.plot(self.ImagingData.time_vector, np.squeeze(stationary_traces.T), 'k', 'LineWidth',1, alpha = 0.3)

            new_ax.set_ylim([plot_y_min, plot_y_max])
            new_ax.set_xticklabels([])
            new_ax.set_yticklabels([])
            plt.setp(new_ax.spines.values(), linewidth=0)
            new_ax.xaxis.set_tick_params(width=0)
            new_ax.yaxis.set_tick_params(width=0)
            new_ax.set_title(f)
            
            if ind_f == 0: # scale bar
                plot_tools.addScaleBars(new_ax, 1, 1, F_value = -0.4, T_value = -0.2)
            
        fig_handle.canvas.draw()
            
    def getSummaryStatistics(self):
        response_peaks = np.max(self.mean_trace_matrix, axis = 2).T
        summary_statistics = {'response_peaks':response_peaks}
        
        return summary_statistics