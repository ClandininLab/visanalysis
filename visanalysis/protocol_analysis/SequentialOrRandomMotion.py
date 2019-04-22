#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler


from visanalysis import protocol_analysis as pa
from visanalysis import plot_tools

class SequentialOrRandomMotionAnalysis(pa.ProtocolAnalysis.BaseAnalysis):
    def __init__(self):
        super().__init__() #call the parent class init method first

    def initializeAnalysis(self):
        random_flag = [];
        for ep in self.ImagingData.epoch_parameters:
            random_flag.append(ep['randomized_order'])
    
        parameter_values = np.array([random_flag]).T
        self.unique_parameter_values, self.mean_trace_matrix, self.sem_trace_matrix, self.individual_traces = pa.ProtocolAnalysis.getTraceMatrixByStimulusParameter(self.ImagingData.response_matrix, parameter_values)
    
        self.random_ind = np.where(self.unique_parameter_values == True)[0][0]
        self.sequential_ind = np.where(self.unique_parameter_values == False)[0][0]
        
    def makeExamplePlots(self, fig_handle = None, eg_trace_ind = 0, export_to_igor_flag = False):
        #plot stuff
        if fig_handle is None:
            fig_handle = plt.figure(figsize=(8,4))
        fig_axes = fig_handle.get_axes()
        ax_ind = 0
        plot_y_max = 1.1*np.max(self.mean_trace_matrix)
        plot_y_min = -0.4  
        
        
        
        ax_ind += 1
        if len(fig_axes) > 1:
            new_ax = fig_axes[ax_ind]
            new_ax.clear()
        else: 
            new_ax = fig_handle.add_subplot(211)
            
        if export_to_igor_flag:
            plot_tools.addLine(new_ax, self.ImagingData.time_vector, self.mean_trace_matrix[:,self.random_ind,:].T, 
                       line_name = 'random', color = self.plot_colors[2], linestyle = '--')
            plot_tools.addErrorBars(new_ax, self.ImagingData.time_vector, np.squeeze(self.individual_traces[self.random_ind]),
                            line_name = 'random_err', stat = 'sem', mode = 'snake', color = self.plot_colors[2])
            
            plot_tools.addLine(new_ax, self.ImagingData.time_vector, self.mean_trace_matrix[:,self.sequential_ind,:].T, 
                       line_name = 'sequential', color = 'k', linestyle = '-')
            plot_tools.addErrorBars(new_ax, self.ImagingData.time_vector, np.squeeze(self.individual_traces[self.sequential_ind]),
                            line_name = 'sequential_err', stat = 'sem', mode = 'snake', color = 'k')

            color_code = str(self.ImagingData.epoch_parameters[0]['color']).replace('.','')
            name_suffix = self.ImagingData.getAxisStructureNameSuffix()
            plot_tools.makeIgorStructure(new_ax, file_name = 'srm_tr_'+ name_suffix + '_' + color_code, subdirectory = 'LC_RFs')
        else:
            plt.gca().set_prop_cycle((cycler('color', self.plot_colors)))
            new_ax.plot(self.ImagingData.time_vector, self.mean_trace_matrix[:,self.random_ind,:].T, linestyle = ':')
            plt.gca().set_prop_cycle((cycler('color', self.plot_colors)))
            new_ax.plot(self.ImagingData.time_vector, self.mean_trace_matrix[:,self.sequential_ind,:].T)
            new_ax.set_ylim([plot_y_min, plot_y_max])
            new_ax.set_axis_off()
            plot_tools.addScaleBars(new_ax, 1, 1, F_value = -0.2, T_value = -0.2)
            
        fig_handle.canvas.draw()

    def getSummaryStatistics(self):
        sequential_peak = np.max(self.mean_trace_matrix[:,self.sequential_ind,:], axis = 1)
        random_peak = np.max(self.mean_trace_matrix[:,self.random_ind,:], axis = 1)
        
        summary_statistics = {'sequential_peak':sequential_peak,
                                   'random_peak':random_peak}
        
        return summary_statistics
    
    def makePopulationPlots(self, summary_statistics):
        fig_handle = plt.figure(figsize=(4,4))
        ax = fig_handle.add_subplot(111)
        ax.set_ylabel('Response to sequential')
        ax.set_xlabel('Response to randomized')
        
        mean_x = np.mean(summary_statistics['random_peak'], axis = 0)
        mean_y = np.mean(summary_statistics['sequential_peak'], axis = 0)
        unity_max = np.max(summary_statistics['sequential_peak'])
        
        plot_tools.addLine(ax, mean_x, mean_y, 
                           line_name = 'pop_mean', marker = '.', color = self.plot_colors[2])
        
        plot_tools.addLine(ax, [0, unity_max], [0, unity_max], 
                           line_name = 'unity', marker = 'None', color = 'k', linestyle = '--')
    
        plot_tools.addErrorBars(ax, summary_statistics['random_peak'], summary_statistics['sequential_peak'],
                                line_name = 'pop', stat = 'sem', mode = 'sticks', color = self.plot_colors[2])
        
        for ind in range(len(summary_statistics['random_peak'])):
            plot_tools.addLine(ax, summary_statistics['random_peak'][ind], summary_statistics['sequential_peak'][ind], 
                           line_name = 'cell' + str(ind), marker = 'o', color = 'k')
        
        name_suffix = self.ImagingData.getAxisStructureNameSuffix()
        plot_tools.makeIgorStructure(ax, file_name = 'srm_pop_'+ name_suffix, subdirectory = 'LC_RFs')