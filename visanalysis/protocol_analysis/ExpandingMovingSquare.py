#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from visanalysis import protocol_analysis as pa
from visanalysis import plot_tools

class ExpandingMovingSquareAnalysis(pa.ProtocolAnalysis.BaseAnalysis):
    def __init__(self):
        super().__init__() #call the parent class init method first
        
    def initializeAnalysis(self):
        width = [];
        for ep in self.ImagingData.epoch_parameters:
            new_val = ep.get('current_width')
            if new_val is None:
                new_val = ep.get('currentWidth')
            width.append(new_val)
    
        parameter_values = np.array([width]).T
        self.unique_parameter_values, self.mean_trace_matrix, self.sem_trace_matrix, self.individual_traces = pa.ProtocolAnalysis.getTraceMatrixByStimulusParameter(self.ImagingData.response_matrix, parameter_values)  
        self.widths = np.unique(np.array(width))

    def makeExamplePlots(self, fig_handle = None, export_to_igor_flag = False, eg_trace_ind = 0):
        if fig_handle is None:
            fig_handle = plt.figure(figsize=(8,4))
        fig_axes = fig_handle.get_axes()
        ax_ind = 0
        plot_y_max = 1.1*np.max(self.mean_trace_matrix)
        plot_y_min = 1*np.min(self.mean_trace_matrix)    
        
        if type(eg_trace_ind) is int:
            plot_ind = eg_trace_ind
        elif type(eg_trace_ind) is tuple:
            plot_ind = eg_trace_ind[0]
        
        grid = plt.GridSpec(2, len(self.widths), wspace=0.2, hspace=0.15)
        for ind_w, w in enumerate(self.widths):
            pull_ind = np.where((self.unique_parameter_values == w).all(axis = 1))[0][0]
            
            ax_ind += 1
            if len(fig_axes) > 1:
                new_ax = fig_axes[ax_ind]
                new_ax.clear()
            else: 
                new_ax = fig_handle.add_subplot(grid[0, ind_w])
                
            new_ax.set_ylim([plot_y_min, plot_y_max])
            new_ax.set_axis_off()
            new_ax.set_title(int(w))
            if ind_w == 0: # scale bar
                plot_tools.addScaleBars(new_ax, 1.0, 1.0, F_value = -0.2, T_value = -0.3)
                
            if export_to_igor_flag:
                plot_tools.addLine(new_ax, self.ImagingData.time_vector, self.mean_trace_matrix[:,pull_ind,:].T, 
                           line_name = 'mean', color = self.plot_colors[2])
                plot_tools.addErrorBars(new_ax, self.ImagingData.time_vector, np.squeeze(self.individual_traces[pull_ind]),
                                line_name = 'err', stat = 'sem', mode = 'snake', color = self.plot_colors[2])
                new_ax.igor_name = 'es_trace_' + str(int(w))
            else:
                new_ax.plot(self.ImagingData.time_vector, self.mean_trace_matrix[plot_ind,pull_ind,:].T, 'k')
                new_ax.fill_between(self.ImagingData.time_vector,
                                    self.mean_trace_matrix[plot_ind,pull_ind,:].T - self.sem_trace_matrix[plot_ind,pull_ind,:].T,
                                    self.mean_trace_matrix[plot_ind,pull_ind,:].T + self.sem_trace_matrix[plot_ind,pull_ind,:].T, alpha = 0.2, color = 'k')
                
                
        response_peaks = np.max(self.mean_trace_matrix, axis = 2).T
        response_peaks_norm = response_peaks.T / response_peaks.max(axis = 0)[:,np.newaxis]
        new_ax = fig_handle.add_subplot(grid[1,1:7])
        new_ax.plot(self.widths,response_peaks_norm.T, linewidth = 2)
        new_ax.set_ylabel('Peak dF/F (norm.)')
        new_ax.set_xlabel('Square size (deg.)')

        fig_handle.canvas.draw()
            
    def getSummaryStatistics(self):
        response_peaks = np.max(self.mean_trace_matrix, axis = 2).T
        response_peaks_norm = response_peaks.T / response_peaks.max(axis = 0)[:,np.newaxis]
        
        summary_statistics = {'width':self.widths,
                                   'response_peaks_norm':response_peaks_norm}
        
        return summary_statistics

    def makePopulationPlots(self, summary_statistics):
        fig_handle = plt.figure(figsize=(5,3))
        ax = fig_handle.add_subplot(111)
        
        mean_x = np.mean(summary_statistics['width'], axis = 0)
        mean_y = np.mean(summary_statistics['response_peaks_norm'], axis = 0)
        
        plot_tools.addLine(ax, mean_x, mean_y, 
                           line_name = 'pop_mean', marker = '.', color = self.plot_colors[2])
    
        plot_tools.addErrorBars(ax, summary_statistics['width'], summary_statistics['response_peaks_norm'],
                                line_name = 'pop', stat = 'sem', mode = 'sticks', color = self.plot_colors[2])
        
        # add individual cell traces:
        for ind in range(len(summary_statistics['width'])):
            plot_tools.addLine(ax, summary_statistics['width'][ind], summary_statistics['response_peaks_norm'][ind], 
                           line_name = 'cell' + str(ind), marker = 'None', color = 'k')

        ax.set_ylabel('Peak dF/F (norm.)')
        ax.set_xlabel('Square size (deg.)')
        ax.set_xlim(left = 0)
        ax.set_ylim(top = 1)
        ax.igor_name = 'es_pop_'
        
        fig_handle.canvas.draw()   
        return fig_handle
     
    