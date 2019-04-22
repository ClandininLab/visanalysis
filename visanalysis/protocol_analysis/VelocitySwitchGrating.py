#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from visanalysis import protocol_analysis as pa
from visanalysis import plot_tools

class VelocitySwitchGratingAnalysis(pa.ProtocolAnalysis.BaseAnalysis):
    def __init__(self):
        super().__init__() #call the parent class init method first
    
    def initializeAnalysis(self):
        start_rate = []
        switch_rate = []
        for ep in self.ImagingData.epoch_parameters:
            start_rate.append(ep['current_start_rate'])
            switch_rate.append(ep['current_switch_rate'])
    
        parameter_values = np.array([start_rate,switch_rate]).T
        self.unique_parameter_values, self.mean_trace_matrix, self.sem_trace_matrix, self.individual_traces = pa.ProtocolAnalysis.getTraceMatrixByStimulusParameter(self.ImagingData.response_matrix, parameter_values)

        self.start_rates = np.unique(np.array(start_rate))
        self.switch_rates = np.unique(np.array(switch_rate))

    def makeExamplePlots(self, fig_handle = None, export_to_igor_flag = False):
        if fig_handle is None:
            fig_handle = plt.figure(figsize=(8,4))
        fig_axes = fig_handle.get_axes()
        ax_ind = 0
        plot_y_max = 1.1*np.max(self.mean_trace_matrix)
        plot_y_min = 1.5*np.min(self.mean_trace_matrix)  

        grid = plt.GridSpec(len(self.start_rates), len(self.switch_rates), wspace=0.2, hspace=0.15)
        for ind_r1, r1 in enumerate(self.start_rates):
            for ind_r2, r2 in enumerate(self.switch_rates):
                start_inds = np.where(self.unique_parameter_values[:,0]==r1)[0]
                switch_inds = np.where(self.unique_parameter_values[:,1]==r2)[0]
                pull_ind = np.intersect1d(start_inds,switch_inds)[0]
                current_mean = self.mean_trace_matrix[:,pull_ind,:]
                current_traces = self.individual_traces[pull_ind]
            
                ax_ind += 1
                if len(fig_axes) > 1:
                    new_ax = fig_axes[ax_ind]
                    new_ax.clear()
                else: 
                    new_ax = fig_handle.add_subplot(grid[ind_r1, ind_r2])
                
                new_ax.plot(self.ImagingData.time_vector, current_mean.T, 'k', 'LineWidth',2)
                new_ax.plot(self.ImagingData.time_vector, current_traces[0,:,:].T, alpha = 0.3)
                
                new_ax.set_ylim([plot_y_min, plot_y_max])
                new_ax.set_xticklabels([])
                new_ax.set_yticklabels([])
                plt.setp(new_ax.spines.values(), linewidth=0)
                new_ax.xaxis.set_tick_params(width=0)
                new_ax.yaxis.set_tick_params(width=0)
                if ind_r1 == len(self.start_rates)-1:
                    new_ax.set_xlabel(str(r2))
                if ind_r2 == 0:
                    new_ax.set_ylabel(str(r1))
                if ind_r1 == 0 and ind_r2 == 0: # scale bar
                    plot_tools.addScaleBars(new_ax, 1, 1, F_value = -0.2, T_value = -0.2)
                    
        fig_handle.canvas.draw()
