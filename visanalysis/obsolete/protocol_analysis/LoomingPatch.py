#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from visanalysis import protocol_analysis as pa
from visanalysis import plot_tools

class LoomingPatchAnalysis(pa.ProtocolAnalysis.BaseAnalysis):
    def __init__(self):
        super().__init__() #call the parent class init method first
    
    def initializeAnalysis(self):
        trajectory_type = []
        rv_ratio = []
        for ep in self.ImagingData.epoch_parameters:
            trajectory_type.append(ep['current_trajectory_type'])
            rv_ratio.append(ep['current_rv_ratio'])
    
        self.parameter_values = np.array([trajectory_type, rv_ratio]).T
        self.unique_parameter_values, self.mean_trace_matrix, self.sem_trace_matrix, self.individual_traces = pa.ProtocolAnalysis.getTraceMatrixByStimulusParameter(self.ImagingData.response_matrix, self.parameter_values)
        
        self.expanding_inds = np.where(self.unique_parameter_values[:,0] == 'expanding')[0]
        self.contracting_inds = np.where(self.unique_parameter_values[:,0] == 'contracting')[0]
        self.randomized_inds = np.where(self.unique_parameter_values[:,0] == 'randomized')[0]
        
        self.rv_ratios = np.unique(self.unique_parameter_values[:,1])
        
    def makeExamplePlots(self, fig_handle = None, eg_trace_ind = 0, export_to_igor_flag = False):
        #plot stuff
        if fig_handle is None:
            fig_handle = plt.figure(figsize=(8,4))
        fig_axes = fig_handle.get_axes()
        ax_ind = 0
        plot_y_max = np.max((1.5*np.max(self.mean_trace_matrix),1))
        plot_y_min = -0.5    
        
        grid = plt.GridSpec(2, len(self.rv_ratios), wspace=0.2, hspace=0.15)
        for ind_r, rv in enumerate(self.rv_ratios):
            rv_ind = np.where(self.unique_parameter_values[:,1] == rv)
            
            expanding_mean = self.mean_trace_matrix[eg_trace_ind,np.intersect1d(rv_ind, self.expanding_inds),:].T
            expanding_sem= self.sem_trace_matrix[eg_trace_ind,np.intersect1d(rv_ind, self.expanding_inds),:].T
            
            contracting_mean = self.mean_trace_matrix[eg_trace_ind,np.intersect1d(rv_ind, self.contracting_inds),:].T
            contracting_sem = self.sem_trace_matrix[eg_trace_ind,np.intersect1d(rv_ind, self.contracting_inds),:].T
            
            randomized_mean = self.mean_trace_matrix[eg_trace_ind,np.intersect1d(rv_ind, self.randomized_inds),:].T
            randomized_sem = self.sem_trace_matrix[eg_trace_ind,np.intersect1d(rv_ind, self.randomized_inds),:].T

            ax_ind += 1
            if len(fig_axes) > 1:
                new_ax = fig_axes[ax_ind]
                new_ax.clear()
            else: 
                new_ax = fig_handle.add_subplot(grid[0, ind_r])
            
            new_ax.plot(self.ImagingData.time_vector, expanding_mean)
            new_ax.fill_between(self.ImagingData.time_vector, np.squeeze(expanding_mean - expanding_sem), np.squeeze(expanding_mean + expanding_sem), alpha = 0.2)
            
            new_ax.plot(self.ImagingData.time_vector, contracting_mean)
            new_ax.fill_between(self.ImagingData.time_vector, np.squeeze(contracting_mean - contracting_sem), np.squeeze(contracting_mean + contracting_sem), alpha = 0.2)
            
            new_ax.plot(self.ImagingData.time_vector, randomized_mean)
            new_ax.fill_between(self.ImagingData.time_vector, np.squeeze(randomized_mean - randomized_sem), np.squeeze(randomized_mean + randomized_sem), alpha = 0.2)

            new_ax.set_ylim([plot_y_min, plot_y_max])
            new_ax.set_axis_off()
            new_ax.set_title(rv)
            

            if ind_r == 0:
                plot_tools.addScaleBars(new_ax, 1, 1, F_value = -0.4, T_value = -0.2)
                plt.legend(('exp','contr','rand'))

            
            #plot stim angular size vs time
            new_ax = fig_handle.add_subplot(grid[1, ind_r])
            temp_rv = np.where(self.parameter_values[:,1] == rv)[0]
            temp_e = np.where(self.parameter_values[:,0] == 'expanding')[0]
            temp_c = np.where(self.parameter_values[:,0] == 'contracting')[0]
            temp_r = np.where(self.parameter_values[:,0] == 'randomized')[0]

            e_ind = np.intersect1d(temp_rv,temp_e)[0]
            new_ax.plot(self.ImagingData.run_parameters['pre_time'] + self.ImagingData.epoch_parameters[e_ind]['time_steps'], self.ImagingData.epoch_parameters[e_ind]['angular_size'],color = self.plot_colors[0])
            
            c_ind = np.intersect1d(temp_rv,temp_c)[0]
            new_ax.plot(self.ImagingData.run_parameters['pre_time'] + self.ImagingData.epoch_parameters[c_ind]['time_steps'], self.ImagingData.epoch_parameters[c_ind]['angular_size'],color = self.plot_colors[1])
            new_ax.set_xlim([0,self.ImagingData.run_parameters['pre_time'] + self.ImagingData.run_parameters['stim_time'] + self.ImagingData.run_parameters['tail_time']])
            
            r_ind = np.intersect1d(temp_rv,temp_r)[0]
            new_ax.plot(self.ImagingData.epoch_parameters[r_ind]['time_steps'], self.ImagingData.epoch_parameters[r_ind]['angular_size'],color = self.plot_colors[2])

        fig_handle.canvas.draw()
