#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 14:15:33 2018

@author: mhturner

"""
import skimage.io as io
from registration import CrossCorr
import numpy as np
import os
import h5py
from operator import itemgetter
import pylab
import seaborn as sns
import scipy.signal as signal
import inspect
import yaml

import visanalysis

class ImagingDataObject():
    def __init__(self, file_name, series_number):
        # Import configuration settings
        path_to_config_file = os.path.join(inspect.getfile(visanalysis).split('visanalysis')[0], 'visanalysis', 'config', 'config.yaml')
        with open(path_to_config_file, 'r') as ymlfile:
            cfg = yaml.safe_load(ymlfile)
        
        self.file_name = file_name
        self.series_number = series_number
        if os.path.isdir(os.path.join(cfg['data_directory'], file_name.replace('-',''))): #sub-dirs for expt days
            self.image_data_directory = os.path.join(cfg['data_directory'], file_name.replace('-',''))
        else:
            self.image_data_directory = cfg['data_directory']
        self.flystim_data_directory = cfg['flystim_data_directory']
        
         # Get stimulus metadata from Flystim hdf5 file
        self.epoch_parameters, self.run_parameters, self.notes = self.getEpochGroupMetadata()

        self.colors = sns.color_palette("deep",n_colors = 20)
        
    def checkEpochNumberCount(self):          
        flystim_epochs = len(self.epoch_parameters)
        presented_epochs = len(self.stimulus_timing['stimulus_start_times'])
        if not flystim_epochs == presented_epochs:
            print('WARNING: metadata epochs do not equal presented epochs')

# %%
    def loadImageSeries(self):
        # Load image series
        #   Check to see if this series has already been registered
        self.raw_file_name = os.path.join(self.image_data_directory, self.image_series_name) + '.tif'
        self.reg_file_name = os.path.join(self.image_data_directory, self.image_series_name) + '_reg.tif'
        
        if os.path.isfile(self.raw_file_name):
            self.raw_series = io.imread(self.raw_file_name)
            self.current_series = self.raw_series
        else:
            self.raw_series = None
            
        if os.path.isfile(self.reg_file_name):
            self.registered_series = io.imread(self.reg_file_name)
            self.current_series = self.registered_series
        else:
            self.registered_series = None
            print('Warning: no registered series found, consider calling registerStack()')
        
        self.roi_image = np.squeeze(np.mean(self.current_series, axis = 0))
        self.roi_response = []
        self.roi_mask = []
        self.roi_path = []
        
# %% 
    def registerStack(self):
        """
        """
        reference_time_frame = 1 #sec, first frames to use as reference for registration
        reference_frame = np.where(self.response_timing['stack_times'] > reference_time_frame)[0][0]
        
        reference_image = np.squeeze(np.mean(self.raw_series[0:reference_frame,:,:], axis = 0))
        register = CrossCorr()
        model = register.fit(self.raw_series, reference=reference_image)
        
        self.registered_series = model.transform(self.raw_series)
        if len(self.registered_series.shape) == 3: #xyt
            self.registered_series = self.registered_series.toseries().toarray().transpose(2,0,1) # shape t, y, x
        elif len(self.registered_series.shape) == 4: #xyzt
            self.registered_series = self.registered_series.toseries().toarray().transpose(3,0,1,2) # shape t, z, y, x

        self.current_series = self.registered_series

                    
# %% 
    ##############################################################################
    #Methods for handling roi saving/loading and generating roi responses:
    ##############################################################################

    def saveRois(self, roi_set_name = 'roi'):
        with h5py.File(os.path.join(self.flystim_data_directory, self.file_name) + '.hdf5','r+') as experiment_file:
            # epoch run group
            series_group = experiment_file['/epoch_runs'].get(str(self.series_number))
            roi_parent_group = series_group.require_group("rois") #opens group if it exists or creates it if it doesn't
            
            current_roi_group = roi_parent_group.require_group(roi_set_name)
            if current_roi_group.get("roi_mask"): #roi dataset exists
                del current_roi_group["roi_mask"]
            if current_roi_group.get("roi_response"):
                del current_roi_group["roi_response"]
            if current_roi_group.get("roi_image"):
                del current_roi_group["roi_image"]
                
            for dataset_key in current_roi_group.keys():
                if 'path_vertices' in dataset_key:
                    del current_roi_group[dataset_key]
                  
            current_roi_group.create_dataset("roi_mask", data = self.roi_mask)
            current_roi_group.create_dataset("roi_response", data = self.roi_response)
            current_roi_group.create_dataset("roi_image", data = self.roi_image)
            for p_ind, p in enumerate(self.roi_path):
                current_roi_group.create_dataset("path_vertices_" + str(p_ind), data = p.vertices)
 
    def loadRois(self, roi_set_name):
        with h5py.File(os.path.join(self.flystim_data_directory, self.file_name) + '.hdf5','r') as experiment_file:
            roi_set_group = experiment_file['/epoch_runs'].get(str(self.series_number)).get('rois').get(roi_set_name)

            self.roi_mask = list(roi_set_group.get("roi_mask")[:]) #load from hdf5 metadata file
            self.roi_response = list(roi_set_group.get("roi_response")[:])
            self.roi_image = roi_set_group.get("roi_image")[:]
            
            self.roi_path = []
            new_path = roi_set_group.get("path_vertices_0")[:]
            ind = 0
            while new_path is not None:
                self.roi_path.append(new_path)
                ind += 1
                new_path = roi_set_group.get("path_vertices_" + str(ind))
                
        self.getResponseTraces()

    def filterResponseTraces(self, window_size = 5):
        for ind, rr in enumerate(self.roi_response):
            self.roi_response[ind] = np.expand_dims(signal.medfilt(np.squeeze(rr), window_size),0)

    def getAvailableROIsets(self):
        with h5py.File(os.path.join(self.flystim_data_directory, self.file_name) + '.hdf5','r+') as experiment_file:
            roi_parent_group = experiment_file['/epoch_runs'].get(str(self.series_number)).require_group('rois')
            roi_set_names = []
            for roi_set in roi_parent_group:
                roi_set_names.append(roi_set)
        
        return roi_set_names
    
    
    
# %%
    def getEpochResponseMatrix(self, respose_trace = None):
        """
        getEpochReponseMatrix(self, roi_response = None)
            Takes in long stack response traces and splits them up into each stimulus epoch

        Returns:
            time_vector (ndarray): in seconds. Time points of each frame acquisition within each epoch
            response_matrix (ndarray): response for each roi in each epoch.
                shape = (num rois, num epochs, num frames per epoch)
        """
        if respose_trace is None:
            respose_trace = np.vstack(self.roi_response)
        
        stimulus_start_times = self.stimulus_timing['stimulus_start_times'] #sec
        stimulus_end_times = self.stimulus_timing['stimulus_end_times'] #sec
        pre_time = self.run_parameters['pre_time'] #sec
        tail_time = self.run_parameters['tail_time'] #sec
        epoch_start_times = stimulus_start_times - pre_time
        epoch_end_times = stimulus_end_times +  tail_time
    
        sample_period = self.response_timing['sample_period'] #sec
        stack_times = self.response_timing['stack_times'] #sec
    
        # Use measured stimulus lengths for stim time instead of epoch param
        # cut off a bit of the end of each epoch to allow for slop in how many frames were acquired
        epoch_time = 0.99 * np.mean(epoch_end_times - epoch_start_times) #sec
        
        # find how many acquisition frames correspond to pre, stim, tail time
        epoch_frames = int(epoch_time / sample_period) #in acquisition frames
        pre_frames = int(pre_time / sample_period) #in acquisition frames
        time_vector = np.arange(0,epoch_frames) * sample_period # sec
        
        no_trials = len(epoch_start_times)
        no_rois = respose_trace.shape[0]
        response_matrix = np.empty(shape=(no_rois, no_trials, epoch_frames), dtype=float)
        response_matrix[:] = np.nan
        cut_inds = np.empty(0, dtype = int)
        for idx, val in enumerate(epoch_start_times):
            stack_inds = np.where(np.logical_and(stack_times < epoch_end_times[idx], stack_times >= epoch_start_times[idx]))[0]
            if len(stack_inds) == 0: #no imaging acquisitions happened during this epoch presentation
                cut_inds = np.append(cut_inds,idx)
                continue
            if np.any(stack_inds > respose_trace.shape[1]):
                cut_inds = np.append(cut_inds,idx)
                continue
            if idx is not 0:
                if len(stack_inds) < epoch_frames: #missed images for the end of the stimulus
                    cut_inds = np.append(cut_inds,idx)
                    continue
            #pull out Roi values for these scans. shape of newRespChunk is (nROIs,nScans)
            new_resp_chunk = respose_trace[:,stack_inds]
    
            # calculate baseline using pre frames
            baseline = np.mean(new_resp_chunk[:,0:pre_frames], axis = 1, keepdims = True)
            # to dF/F
            new_resp_chunk = (new_resp_chunk - baseline) / baseline;
            response_matrix[:,idx,:] = new_resp_chunk[:,0:epoch_frames]
            
        response_matrix = np.delete(response_matrix,cut_inds, axis = 1)
        return time_vector, response_matrix


 
# %%        
    ##############################################################################
    #Misc utils:
    ##############################################################################
        
    def getAxisStructureNameSuffix(self):
        cell_type = self.run_parameters['fly:driver_1'][0:4]
        indicator = self.run_parameters['fly:indicator_1']
        if cell_type == 'L2 (21Dhh)':
            cell_type_code = 'L2'
        elif cell_type == 'LC9 (VT032961; VT040569)':
            cell_type_code = 'LC9'
        else:
            cell_type_code = cell_type[0:4]
            
        if indicator == 'SF-iGluSnFR.A184V':
            indicator_code = 'SFiGSFR'
        elif indicator == '10_90_GCaMP6f':
            indicator_code = '1090_GC6f'
        else:
            indicator_code = indicator.replace('.','').replace('-','').replace(' ','')
    
        name_suffix = cell_type_code + '_' + indicator_code + '_'
        return name_suffix

    def getSeriesIDSuffix(self):
        return self.file_name.replace('-','') + '_' + str(self.series_number)


# %%         
    def getEpochGroupMetadata(self):
        """
        For reading stimulus metadata from flystim data file
        """

        with h5py.File(os.path.join(self.flystim_data_directory, self.file_name) + '.hdf5','r') as experiment_file:
            # Notes from entire file
            notes = addAttributesToDictionary(experiment_file['/notes'], {})
        
            # epoch run group
            series_group = experiment_file['/epoch_runs'][str(self.series_number)]
            # get epoch group attributes (run parameters)
            run_parameters = addAttributesToDictionary(series_group, {})
            
            epoch_parameters = []
            for epoch in series_group:
                # Collapse all levels of epoch parameters into a single dict
                if epoch == 'rois':
                    continue
                elif epoch == 'pois':
                    continue
                else:
                    new_epoch_params = addAttributesToDictionary(series_group[epoch],{}) #epoch_time is the only attribute at this level
                    new_epoch_params = addAttributesToDictionary(series_group[epoch + '/epoch_parameters'], new_epoch_params)
                    new_epoch_params = addAttributesToDictionary(series_group[epoch + '/convenience_parameters'], new_epoch_params)
                    epoch_parameters.append(new_epoch_params)
                
            # sort epochs by start time
            epoch_parameters = sorted(epoch_parameters, key=itemgetter('epoch_time')) 
            
        return epoch_parameters, run_parameters, notes
    
# %%
    def getEpochAndFrameTiming(self, time_vector, frame_monitor, sample_rate,
                               plot_trace_flag = True,
                               threshold = 0.6, 
                               minimum_epoch_separation = 2e3, # datapoints
                               frame_slop = 10, #datapoints +/- ideal frame duration
                               command_frame_rate = 120):
        
        # Low-pass filter frame_monitor trace
        b, a = signal.butter(4, 10*command_frame_rate, btype = 'low', fs = sample_rate)
        frame_monitor = signal.filtfilt(b, a, frame_monitor)

        # shift & normalize so frame monitor trace lives on [0 1]
        frame_monitor = frame_monitor - np.min(frame_monitor)
        frame_monitor = frame_monitor / np.max(frame_monitor)
        
        # find lightcrafter frame flip times
        V_orig = frame_monitor[0:-2]
        V_shift = frame_monitor[1:-1]
        ups = np.where(np.logical_and(V_orig < threshold, V_shift >= threshold))[0] + 1
        downs = np.where(np.logical_and(V_orig >= threshold, V_shift < threshold))[0] + 1
        frame_times = np.sort(np.append(ups,downs))
        
        # Use frame flip times to find stimulus start times
        stimulus_start_frames = np.append(0, np.where(np.diff(frame_times) > minimum_epoch_separation)[0] + 1)
        stimulus_end_frames = np.append(np.where(np.diff(frame_times) > minimum_epoch_separation)[0],len(frame_times)-1)
        stimulus_start_times = frame_times[stimulus_start_frames] / sample_rate  # datapoints -> sec
        stimulus_end_times = frame_times[stimulus_end_frames] / sample_rate  # datapoints -> sec
        
        # Find dropped frames and calculate frame rate
        interval_duration = np.diff(frame_times)
        frame_len = interval_duration[np.where(interval_duration < minimum_epoch_separation)]
        ideal_frame_len = 1 / command_frame_rate  * sample_rate #datapoints
        dropped_frame_inds = np.where(np.abs(frame_len - ideal_frame_len)>frame_slop)[0]
        if len(dropped_frame_inds)>0:
            print('Warning! Dropped ' + str(len(dropped_frame_inds)) + ' frame(s)')
        good_frame_inds = np.where(np.abs(frame_len - ideal_frame_len)<frame_slop)[0]
        measured_frame_len = np.mean(frame_len[good_frame_inds]) #datapoints
        frame_rate = 1 / (measured_frame_len / sample_rate) #Hz
        
        if plot_trace_flag:
            pylab.plot(time_vector,frame_monitor)
            pylab.plot(time_vector[frame_times],threshold * np.ones(frame_times.shape),'ko')
            pylab.plot(stimulus_start_times, threshold * np.ones(stimulus_start_times.shape),'go')
            pylab.plot(stimulus_end_times, threshold * np.ones(stimulus_end_times.shape),'ro')
            pylab.plot(frame_times[dropped_frame_inds] / sample_rate, 1 * np.ones(dropped_frame_inds.shape),'ro')
            pylab.show
        
        return {'frame_times':frame_times, 'stimulus_end_times':stimulus_end_times,
                'stimulus_start_times':stimulus_start_times, 'dropped_frame_inds':dropped_frame_inds,
                'frame_rate':frame_rate}


def addAttributesToDictionary(group_object, dictionary, verbose = False):
    for a in group_object.attrs:
        if a in dictionary:
            if verbose:
                print('Warning! - overwriting parameters')
        dictionary[a] = group_object.attrs[a]
    return dictionary