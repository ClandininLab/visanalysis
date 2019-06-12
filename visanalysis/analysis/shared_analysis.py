#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import matplotlib.pyplot as plt
import numpy as np

from visanalysis import plot_tools


def getTraceMatrixByStimulusParameter(response_matrix,parameter_values):
    # parameter values is nTrials x nParams  numpy array
    # returns:
    #   uniqueParameterValues (nConditions x nParams)
    #   meanTraceMatrix and semTraceMatrix (nRois x nConditions, time)
    unique_parameter_values = np.unique(parameter_values, axis = 0)
    
    no_rois = response_matrix.shape[0]
    epoch_len = response_matrix.shape[2]
    no_conditions = len(unique_parameter_values)
    
    mean_trace_matrix = np.empty(shape=(no_rois, no_conditions, epoch_len), dtype=float)
    mean_trace_matrix[:] = np.nan
    
    sem_trace_matrix = np.empty(shape=(no_rois, no_conditions, epoch_len), dtype=float)
    sem_trace_matrix[:] = np.nan
    
    individual_traces = []
    for vInd, V in enumerate(unique_parameter_values):
        pull_inds = np.where((parameter_values == V).all(axis = 1))[0]
        current_responses = response_matrix[:,pull_inds,:]
        mean_trace_matrix[:,vInd,:] = np.mean(current_responses, axis = 1)
        sem_trace_matrix[:,vInd,:] = np.std(current_responses, axis = 1) / np.sqrt(len(pull_inds))
        individual_traces.append(current_responses)
        
    return unique_parameter_values, mean_trace_matrix, sem_trace_matrix, individual_traces


def plotResponseByCondition(ImagingData, roi_name, eg_ind = 0, condition = None, fig_handle = None):
    conditioned_param = [];
    for ep in ImagingData.epoch_parameters:
        conditioned_param.append(ep[condition])

    parameter_values = np.array([conditioned_param]).T
    unique_parameter_values, mean_trace_matrix, sem_trace_matrix, individual_traces = getTraceMatrixByStimulusParameter(ImagingData.roi[roi_name]['epoch_response'], parameter_values)

    unique_params = np.unique(np.array(conditioned_param))
    
    #plot stuff
    if fig_handle is None:
        fig_handle = plt.figure(figsize=(10,2))
        
    grid = plt.GridSpec(1, len(unique_params), wspace=0.2, hspace=0.15)
    fig_axes = fig_handle.get_axes()
    ax_ind = 0
        
    plot_y_max = 1.1*np.max(mean_trace_matrix)
    plot_y_min = 1*np.min(mean_trace_matrix)    
    
    for ind_v, val in enumerate(unique_params):
        pull_ind = np.where((unique_parameter_values == val).all(axis = 1))[0][0]
        ax_ind += 1
        if len(fig_axes) > 1:
            new_ax = fig_axes[ax_ind]
            new_ax.clear()
        else: 
            new_ax = fig_handle.add_subplot(grid[0,ind_v])
        
        new_ax.plot(ImagingData.roi[roi_name]['time_vector'], mean_trace_matrix[eg_ind,pull_ind,:].T)
        new_ax.set_ylim([plot_y_min, plot_y_max])
        new_ax.set_axis_off()
        new_ax.set_title(int(val))
        if ind_v == 0: # scale bar
            plot_tools.addScaleBars(new_ax, 1, 1, F_value = -0.1, T_value = -0.2)
    fig_handle.canvas.draw()
    return fig_handle

def plotRoiResponses(ImagingData, roi_name, fig_handle = None):
    if fig_handle is None:
        fig_handle = plt.figure(figsize=(10,2))
        
        
    plot_y_min = -0.5
    plot_y_max = 1
     
    no_rois = ImagingData.roi.get(roi_name).get('epoch_response').shape[0]
    cols = 4
    rows = np.ceil(no_rois/cols)
    fig_axes = fig_handle.get_axes()
    ax_ind = 0
    for roi in range(no_rois):
        ax_ind += 1
        if len(fig_axes) > 1:
            new_ax = fig_axes[ax_ind]
            new_ax.clear()
        else: 
            new_ax = fig_handle.add_subplot(cols,rows,roi+1)
        
        no_trials = ImagingData.roi[roi_name]['epoch_response'][roi,:,:].shape[0]
        time_vector = ImagingData.roi[roi_name]['time_vector']
        current_mean = np.mean(ImagingData.roi[roi_name]['epoch_response'][roi,:,:], axis = 0)
        current_std = np.std(ImagingData.roi[roi_name]['epoch_response'][roi,:,:], axis = 0)
        current_sem = current_std / np.sqrt(no_trials)
        
        new_ax.plot(time_vector, current_mean, 'k')
        new_ax.fill_between(time_vector, current_mean - current_sem, current_mean + current_sem, alpha = 0.5)
        new_ax.set_ylim([plot_y_min, plot_y_max])
        new_ax.set_axis_off()
        new_ax.set_title(int(roi))
        
        if roi == 0: # scale bar
            plot_tools.addScaleBars(new_ax, 1, 1, F_value = -0.1, T_value = -0.2)
            
    fig_handle.canvas.draw()
    return fig_handle

def getEpochResponseHyperstack(ImagingData):
    """
    getEpochResponseHyperstack(ImagingData)
        Takes in long stack and splits it up into substacks for each stimulus epoch
    
    Args:
        stack (ndarray): image stack.
            shape = (num frames, dimY dimX)
        ImagingData object
    Returns:
        timeVector (ndarray): in seconds. Time points of each frame acquisition within each epoch
        responseMatrix (ndarray): sub-hyperstacks for each epoch
            shape = (num epochs, num frames per epoch, xDim, yDim)
    """
    stack = ImagingData.current_series
    
    stimulus_start_times = ImagingData.stimulus_timing['stimulus_start_times']
    stimulus_end_times = ImagingData.stimulus_timing['stimulus_end_times']
    pre_time = ImagingData.run_parameters['pre_time'] * 1e3 #sec -> msec
    tail_time = ImagingData.run_parameters['tail_time'] * 1e3 #sec -> msec
    epoch_start_times = stimulus_start_times - pre_time
    epoch_end_times = stimulus_end_times + tail_time

    sample_period = ImagingData.response_timing['sample_period'] #msec
    stack_times = ImagingData.response_timing['stack_times'] #msec
    no_trials = len(epoch_start_times)

    # Use measured stimulus lengths for stim time instead of epoch param
    # cut off a bit of the end of each epoch to allow for slop in how many frames were acquired
    epoch_time = 0.99 * np.mean(epoch_end_times - epoch_start_times) #msec
    
    # find how many acquisition frames correspond to pre, stim, tail time
    epoch_frames = int(epoch_time / sample_period) #in acquisition frames
    pre_frames = int(pre_time / sample_period) #in acquisition frames
    time_vector = np.arange(0,epoch_frames) * sample_period / 1e3 # msec -> sec

    dimX = stack.shape[1]; dimY = stack.shape[2]; dimT = stack.shape[0];
    epoch_response_hyperstack = np.empty(shape=(no_trials, epoch_frames, dimX, dimY), dtype=float)
    epoch_response_hyperstack[:] = np.nan
    cutInds = np.empty(0, dtype = int)
    for idx, val in enumerate(epoch_start_times): #for stimulus epochs
        stackInds = np.where(np.logical_and(stack_times < epoch_end_times[idx], stack_times >= epoch_start_times[idx]))[0]
        if len(stackInds) == 0: # no imaging frames during this stimulus presentation
            cutInds = np.append(cutInds,idx)
            continue
        if np.any(stackInds > dimT): # stimulus epochs extend beyond imaging frames
            cutInds = np.append(cutInds,idx)
            continue
        if idx is not 0:
            if len(stackInds) < epoch_frames: #not enough imaging frames to fill out entire epoch
                cutInds = np.append(cutInds,idx)
                continue
        newTrialResponseStack = stack[stackInds,:,:]
        
        epoch_response_hyperstack[idx,:,:,:] = newTrialResponseStack[0:epoch_frames,:,:]
    
    epoch_response_hyperstack = np.delete(epoch_response_hyperstack,cutInds, axis = 0)
        
    return time_vector, epoch_response_hyperstack