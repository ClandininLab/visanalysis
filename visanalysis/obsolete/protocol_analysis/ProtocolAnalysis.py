#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base analysis parent class

Children subclasses should define:
   initializeAnalysis()
   makeExamplePlots()
   getSummaryStatistics()
   makePopulationPlots()


"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
import glob
import os
import h5py
import inspect
import yaml

import visanalysis
from visanalysis.imaging_data import AodScopeData, BrukerData


class BaseAnalysis():
    def __init__(self):
# =============================================================================
#         SET DEFAULTS FOR PLOTTING AND DIRECTORIES
# =============================================================================
        self.plot_colors = sns.color_palette("deep",n_colors = 10)     
        plt.rc('lines', linewidth=1)
        plt.rc('axes', prop_cycle=(cycler('color', self.plot_colors)))
    
        self.export_to_igor_flag = False
        self.igor_export_directory = 'LC_RFs'
        
        # Import configuration settings
        path_to_config_file = os.path.join(inspect.getfile(visanalysis).split('visanalysis')[0], 'visanalysis', 'config', 'config.yaml')
        with open(path_to_config_file, 'r') as ymlfile:
            cfg = yaml.safe_load(ymlfile)
        
        self.flystim_data_directory = cfg['flystim_data_directory']
        self.getAvailableFileNames()
        
# =============================================================================
#         FUNCTIONS TO GET AVAILABLE DATA AND RETRIEVE FILTERED DATAFILE LISTS
# =============================================================================
    def getAvailableFileNames(self):
        ""
        
        ""
        fileNames = glob.glob(os.path.join(self.flystim_data_directory,"*.hdf5"))
        self.all_series = []
        for ind, fn in enumerate(fileNames): #get all series
            dataFile = h5py.File(fn, 'r')
            
            epoch_run_keys = list(dataFile.get('epoch_runs').keys())

            for er in epoch_run_keys:
                newSeries = {'series_number':int(er),
                             'file_name':os.path.split(fn)[-1].split('.')[0]}
                current_run = dataFile.get('epoch_runs')[er]
                # Pull out run parameters
                for a in current_run.attrs:
                    newSeries[a] = current_run.attrs[a]
                
                newSeries['protocol_ID'] = current_run.attrs['protocol_ID']
                newSeries['driver'] = current_run.attrs['fly:driver_1'][0:4]
                newSeries['indicator'] = current_run.attrs['fly:indicator_1']
                newSeries['fly_id'] = current_run.attrs['fly:fly_id']
                
                newSeries['rig'] = dataFile.attrs['rig']
                newSeries['date'] = dataFile.attrs['date']
                newSeries['experimenter'] = dataFile.attrs['experimenter']


                # Pull out convenience parameters
                first_epoch = current_run[list(current_run.keys())[0]]     
                for a in first_epoch['convenience_parameters'].attrs:
                    newSeries[a] = first_epoch['convenience_parameters'].attrs[a]

                self.all_series.append(newSeries)
                
            self.available_drivers = np.unique([x['driver'] for x in self.all_series])
            self.available_indicators = np.unique([x['indicator'] for x in self.all_series])
            self.available_protocols = np.unique([x['protocol_ID'] for x in self.all_series])   
            
    def getTargetFileNames(self, **kwargs):
        ""
        
        ""
        self.target_file_names = []
        self.target_series_numbers = []
        for sInd, s in enumerate(self.all_series):
            include = True
            for key in kwargs:
                if (s.get(key) is not None) & (s.get(key) != kwargs[key]):
                    include = False
                    continue
            if include:
                self.target_file_names.append(s['file_name'])
                self.target_series_numbers.append(s['series_number'])

# =============================================================================
#     FUNCTIONS TO PLOT AND EXPORT RESULTS
# =============================================================================
    def doExampleAnalysis(self, file_name, series_number, roi_set_name, export_to_igor_flag = False, eg_trace_ind=0):
        ""
        
        ""
        self.ImagingData = AodScopeData.ImagingDataObject(file_name, series_number) #Load imaging data object for this imaging series
        self.initializeAnalysis(roi_set_name) #Initialize the subclass analysis
        
        fig_handle = plt.figure(figsize=(8,4))
        self.makeExamplePlots(fig_handle = fig_handle, export_to_igor_flag = export_to_igor_flag, eg_trace_ind = eg_trace_ind)
        
        if export_to_igor_flag:
            name_suffix = self.ImagingData.getAxisStructureNameSuffix()
            for ah in fig_handle.get_axes(): 
                if hasattr(ah, 'igor_name'):
                    visanalysis.plot_tools.makeIgorStructure(ah, file_name = ah.igor_name + name_suffix, subdirectory = self.igor_export_directory)
        
        self.ImagingData.generateRoiMap(scale_bar_length = 20)
        
        plt.show()
 
    def doPopulationAnalysis(self, file_names, series_numbers, roi_set_name, export_to_igor_flag = False):
        ""
        
        ""
        self.population_summary_statistics = {}
        self.file_names = []
        f_count = 0
        for f_ind, file_name in enumerate(file_names):
            self.ImagingData = imaging_data.ImagingDataObject(file_name, series_numbers[f_ind])
            # check that a saved ROI exists, if not, skip this file in pop analysis:
            if roi_set_name in self.ImagingData.getAvailableROIsets():
                 self.ImagingData.loadRois(roi_set_name)
                 self.initializeAnalysis()
            else:
                continue

            new_summary_statistics = self.getSummaryStatistics()

            for key in new_summary_statistics.keys():
                if f_count == 0:
                    self.population_summary_statistics[key] = []
                
                self.population_summary_statistics[key].append(new_summary_statistics[key])

            self.file_names.append(file_name)
            f_count+=1
            
        for key in new_summary_statistics.keys():
            self.population_summary_statistics[key] = np.vstack(self.population_summary_statistics[key])
            
        fig_handle = self.makePopulationPlots(self.population_summary_statistics)
        if export_to_igor_flag:
            name_suffix = self.ImagingData.getAxisStructureNameSuffix()
            for ah in fig_handle.get_axes(): 
                visanalysis.plot_tools.makeIgorStructure(ah, file_name = ah.igor_name + name_suffix, subdirectory = self.igor_export_directory)
        
        plt.show()

    def doOnlineAnalysis(self, ImagingData, fig_handle = None):
        self.ImagingData = ImagingData
        self.initializeAnalysis() #Initialize the subclass analysis
        self.makeExamplePlots(fig_handle = fig_handle, eg_trace_ind = 0, export_to_igor_flag = False)
        
    # # Overwrite these methods in the analysis subclass # #
    def makeExamplePlots(self, fig_handle = None, export_to_igor_flag = False, eg_trace_ind = 0):
        print('No subclass method defined for makeExamplePlots')
        
    def getSummaryStatistics(self):
        print('No subclass method defined for getSummaryStatistics')
        
    def makePopulationPlots(self, summary_statistics):
        print('No subclass method defined for makePopulationPlots')
        

# =============================================================================
# DATA PROCESSING FUNCTIONS FOR SORTING AND ORGANIZING RESPONSES
# =============================================================================
#TODO: clean up var names etc

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