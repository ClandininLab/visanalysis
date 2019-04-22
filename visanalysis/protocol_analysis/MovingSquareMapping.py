#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For MovingSquareMapping protocol
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy import signal

from visanalysis import protocol_analysis as pa
from visanalysis import plot_tools

class MovingSquareMappingAnalysis(pa.ProtocolAnalysis.BaseAnalysis):
    def __init__(self):
        super().__init__()
        
    def initializeAnalysis(self, screen_center = (130, 120)):
        axis = []; location = []
        for ep in self.ImagingData.epoch_parameters:
            if ep['current_search_axis'] == 'azimuth':
                axis.append(1)
                location.append(ep['current_location'] - screen_center[0])
            elif ep['current_search_axis'] == 'elevation':
                axis.append(2)
                location.append(ep['current_location'] - screen_center[1])
    
        parameter_values = np.array([axis, location]).T
        self.unique_parameter_values, self.mean_trace_matrix, self.sem_trace_matrix, self.individual_traces = pa.ProtocolAnalysis.getTraceMatrixByStimulusParameter(self.ImagingData.response_matrix, parameter_values)
    
        self.azimuth_inds = np.where(self.unique_parameter_values[:,0] == 1)[0]
        self.elevation_inds = np.where(self.unique_parameter_values[:,0] == 2)[0]
    
        self.azimuth_locations = np.unique(np.array(location)[np.where(np.array(axis) == 1)])
        self.elevation_locations = np.unique(np.array(location)[np.where(np.array(axis) == 2)])
        
        self.az = self.unique_parameter_values[self.azimuth_inds,1]
        self.el = self.unique_parameter_values[self.elevation_inds,1]
        
        self.getRFHeatMap()
        
    def makeExamplePlots(self, fig_handle = None, export_to_igor_flag = False, eg_trace_ind = 0):
        if fig_handle is None:
            fig_handle = plt.figure(figsize=(9,6))
        fig_axes = fig_handle.get_axes()
        ax_ind = 0
        plot_y_max = 1.3*np.max(self.mean_trace_matrix)
        plot_y_min = -0.6
        if type(eg_trace_ind) is int:
            plot_ind = eg_trace_ind
        elif type(eg_trace_ind) is tuple:
            plot_ind = eg_trace_ind[0]
        
        self.grid = plt.GridSpec(len(self.elevation_locations)+1, len(self.azimuth_locations)+1, wspace=0.4, hspace=0.3)
        for ind_a, a in enumerate(self.azimuth_inds):
            current_loc = self.unique_parameter_values[a,1]
            current_mean = self.mean_trace_matrix[:,a,:]
            current_sem = self.sem_trace_matrix[:,a,:]
    
            ax_ind += 1
            if len(fig_axes) > 1:
                new_ax = fig_axes[ax_ind]
                new_ax.clear()
            else: 
                new_ax = fig_handle.add_subplot(self.grid[0,ind_a+1])
                
            new_ax.plot(self.ImagingData.time_vector, current_mean[plot_ind,:].T,alpha=1, color = self.plot_colors[plot_ind], linewidth=2)
            new_ax.fill_between(self.ImagingData.time_vector,
                                current_mean[plot_ind,:].T - current_sem[plot_ind,:].T,
                                current_mean[plot_ind,:].T + current_sem[plot_ind,:].T, alpha = 0.2)
            if not export_to_igor_flag:
                new_ax.set_title(str(int(current_loc)), fontsize=8) 
            new_ax.set_ylim([plot_y_min, plot_y_max])
            new_ax.set_axis_off()
            if ind_a == 0:
                plot_tools.addScaleBars(new_ax, 1, 1, F_value = -0.4, T_value = -0.4)
            new_ax.igor_name = 'MSM_az_trace' + str(int(current_loc))
            fig_handle.canvas.draw()
            
        for ind_e, e in enumerate(self.elevation_inds):
            current_loc = self.unique_parameter_values[e,1]
            current_mean = self.mean_trace_matrix[:,e,:]
            current_sem = self.sem_trace_matrix[:,e,:]
            ax_ind += 1
            if len(fig_axes) > 1:
                new_ax = fig_axes[ax_ind]
                new_ax.clear()
            else: 
                new_ax = fig_handle.add_subplot(self.grid[ind_e+1,0])
                
            new_ax.plot(self.ImagingData.time_vector, current_mean[plot_ind,:].T,alpha=1, color = self.plot_colors[plot_ind], linewidth=2)
            new_ax.fill_between(self.ImagingData.time_vector,
                                current_mean[plot_ind,:].T - current_sem[plot_ind,:].T,
                                current_mean[plot_ind,:].T + current_sem[plot_ind,:].T, alpha = 0.2)
            if not export_to_igor_flag:
                new_ax.set_title(str(int(current_loc)), fontsize=8) 
            new_ax.set_ylim([plot_y_min, plot_y_max])
            new_ax.set_axis_off()
            if ind_e == 0:
                plot_tools.addScaleBars(new_ax, 1, 1, F_value = -0.4, T_value = -0.4)
            new_ax.igor_name = 'MSM_el_trace' + str(int(current_loc))
            fig_handle.canvas.draw()
         

        # single heat mapt to go with example traces
        heat_map_ax = fig_handle.add_subplot(fig_handle.add_subplot(self.grid[1:len(self.elevation_locations)+1, 1:len(self.azimuth_locations)+1]))
        self.plotRfHeatMap(ax_handle = heat_map_ax, export_to_igor_flag = export_to_igor_flag, eg_trace_ind = eg_trace_ind)
        
        #multi=panel heatmap
        fh = plt.figure()
        for rr in range(len(self.ImagingData.roi_response)):
            new_ax = fh.add_subplot(3,3,rr+1)
            self.plotRfHeatMap(ax_handle = new_ax, export_to_igor_flag = False, eg_trace_ind = rr)
            new_ax.yaxis.tick_right()
            new_ax.set_axis_off()
            new_ax.set_title(r'$\blacksquare$',color = self.plot_colors[rr], fontsize = 'large')
       
    def getSummaryStatistics(self):
        max_el_ind = np.argmax(self.resp_el)
        max_el_response = np.squeeze(self.mean_trace_matrix[:,self.elevation_inds[max_el_ind],:])

        resampled_tuple = signal.resample(max_el_response, 100, self.ImagingData.time_vector)
        
        summary_statistics = {'resp_mat':self.resp_mat,
                                   'fit_params':self.fit_params,
                                   'max_response':resampled_tuple[0]}
        
        return summary_statistics
    
    def makePopulationPlots(self, summary_statistics):
        fig_handle = plt.figure(figsize=(3,3))
        ax = fig_handle.add_subplot(111)

        for ind, fit_params in enumerate(summary_statistics['fit_params']):
            fit_params = fit_params.copy()
            fit_params[1] = 0 # center on (0,0)
            fit_params[2] = 0
            
            xx, yy = np.meshgrid(np.arange(-20,20,1), np.arange(-20,20,1))
            fit_xy_tuple = (xx,yy)
            fit_resp = self.Gauss_2D(fit_xy_tuple, *fit_params).reshape(xx.shape)
            ax.contour(xx, yy, fit_resp, [0.5 * fit_params[0]], colors = 'k')
            plt.gca().set_aspect('equal', adjustable='box')
            
        mean_sigma = np.mean(summary_statistics['fit_params'][:,3:5],axis = 0)
        print(mean_sigma)

    def getRFHeatMap(self):
        self.resp_az = np.max(self.mean_trace_matrix[:,self.azimuth_inds,:], axis = 2)
        self.resp_el = np.max(self.mean_trace_matrix[:,self.elevation_inds,:], axis = 2)
        
        self.resp_mat = []
        self.fit_params = []
        for rr in range(len(self.ImagingData.roi_response)):
            new_resp_mat = np.outer(self.resp_az[rr,:],self.resp_el[rr,:]).T
            self.resp_mat.append(new_resp_mat)
            self.fit_params.append(self.fitGauss2D(new_resp_mat))
            
    def plotRfHeatMap(self, ax_handle, export_to_igor_flag = False, eg_trace_ind = None):
        extent=[self.azimuth_locations[0], self.azimuth_locations[-1],
                    self.elevation_locations[-1], self.elevation_locations[0]]
        
        heat_map = self.resp_mat[eg_trace_ind]
        ax_handle.imshow(heat_map, extent=extent, cmap=plt.cm.Reds,interpolation='none')
        ax_handle.yaxis.tick_right()
        ax_handle.tick_params(axis='x', which='major', labelsize=12)
        ax_handle.tick_params(axis='y', which='major', labelsize=12)
        self.plotGaussBoundary(ax_handle, ind = eg_trace_ind)
        
    def plotResponsePeakTraces(self, export_to_igor_flag = False, eg_trace_ind = None):
         if eg_trace_ind is None:
            fh = plt.figure(figsize=(4.5,6))
            az_ax = fh.add_subplot(211)
            el_ax = fh.add_subplot(212)
            for rr in range(len(self.ImagingData.roi_response)):
                #Peak resp vs. az / el
                plot_tools.addLine(az_ax, self.az, self.resp_az[rr,:], line_name = 'az'+str(rr), color = self.plot_colors[rr])
                plot_tools.addLine(el_ax, self.el, self.resp_el[rr,:], line_name = 'el'+str(rr), color = self.plot_colors[rr])
                            
            az_ax.set_ylabel('Peak dF/F')
            az_ax.set_xlabel('Azimuth (deg.)')
            
            el_ax.set_ylabel('Peak dF/F')
            el_ax.set_xlabel('Elevation (deg.)')
            
            if export_to_igor_flag:
                name_suffix = self.ImagingData.getAxisStructureNameSuffix()
                plot_tools.makeIgorStructure(az_ax, file_name = 'az_resp'+ name_suffix, subdirectory = 'LC_RFs')
                plot_tools.makeIgorStructure(el_ax, file_name = 'el_resp'+ name_suffix, subdirectory = 'LC_RFs')


    def plotGaussBoundary(self, ax, ind = 0):
        #make upsampled smooth fit:
        self.xx, self.yy = np.meshgrid(np.arange(np.min(self.az), np.max(self.az),1), np.arange(np.min(self.el), np.max(self.el),1))
        fit_xy_tuple = (self.xx,self.yy)
        fit_resp = self.Gauss_2D(fit_xy_tuple, *self.fit_params[ind]).reshape(self.xx.shape)
        ax.contour(self.xx, self.yy, fit_resp, [0.5 * self.fit_params[ind][0]], colors = 'k')

    def fitGauss2D(self, data):      
        x, y = np.meshgrid(self.az, self.el)
        fit_xy_tuple = (x,y)
        
        #fit to Gauss_2D to get params
        initial_guess = (np.max(np.hstack(data)),np.median(self.az),np.median(self.el),3,3)
        fit_params, pcov = opt.curve_fit(self.Gauss_2D, fit_xy_tuple, np.hstack(data), 
                                             p0=initial_guess,
                                             bounds=([0, np.min(self.az), np.min(self.el), 0, 0], [1e3, np.max(self.az), np.max(self.el), 25, 25]))
        return fit_params

    def Gauss_2D(self, xy_tuple, amplitude, xo, yo, sigma_x, sigma_y):
        # https://en.wikipedia.org/wiki/Gaussian_function
        # https://stackoverflow.com/questions/21566379/fitting-a-2d-gaussian-function-using-scipy-optimize-curve-fit-valueerror-and-m
        # set theta = 0 for this, because this mapping shouldn't really allow for angled fits
        theta = 0
        x, y = xy_tuple
        xo = float(xo)
        yo = float(yo)    
        a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
        b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
        c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
        resp = amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
        return resp.ravel()