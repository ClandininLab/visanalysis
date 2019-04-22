#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 14:15:33 2018

@author: mhturner

"""
import skimage.io as io
import xml.etree.ElementTree as ET
from registration import CrossCorr
import numpy as np
import os
import h5py
from operator import itemgetter
import pylab
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import path
import matplotlib.patches as patches
import seaborn as sns
import scipy.signal as signal

from visanalysis import protocol_analysis as pa
from visanalysis import plot_tools


class ImagingDataObject():
    def __init__(self, file_name, series_number,
                 data_directory = '/Users/mhturner/Dropbox/ClandininLab/CurrentData',
                 flystim_data_directory = '/Users/mhturner/Dropbox/ClandininLab/CurrentData/FlystimData'):
        
        self.file_name = file_name
        self.series_number = series_number
        if os.path.isdir(os.path.join(data_directory, file_name.replace('-',''))): #sub-dirs for expt days
            self.image_data_directory = os.path.join(data_directory, file_name.replace('-',''))
        else:
            self.image_data_directory = data_directory
        self.flystim_data_directory = flystim_data_directory

        # Image series is of the format: TSeries-YYYYMMDD-00n
        self.image_series_name = 'TSeries-' + file_name.replace('-','') + '-' + ('00' + str(series_number))[-3:]
        
        # get metadata and photodiode timing from bruker xml files
        self.response_timing = self.getAcquisitionTiming()
        self.stimulus_timing = getEpochAndFrameTiming(self.image_data_directory, self.image_series_name, plot_trace_flag=False)
        self.metadata = self.getPVMetadata()

        # Get stimulus metadata data from Flystim hdf5 file
        try:
            self.epoch_parameters, self.run_parameters, self.notes = self.getEpochGroupMetadata()
            
            flystim_epochs = len(self.epoch_parameters)
            presented_epochs = len(self.stimulus_timing['stimulus_start_times'])
            if not flystim_epochs == presented_epochs:
                print('WARNING: metadata epochs do not equal presented epochs')
        except:
            print('Warning: no stimulus timing information loaded')
        
        
        self.colors = sns.color_palette("deep",n_colors = 10)
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
        reference_time_frame = 1000 #msec, first frames to use as reference for registration
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
            if current_roi_group.get("path_vertices"):
                del current_roi_group["path_vertices"]
            if current_roi_group.get("roi_image"):
                del current_roi_group["roi_image"]
                
            current_roi_group.create_dataset("roi_mask", data = self.roi_mask)
            current_roi_group.create_dataset("roi_response", data = self.roi_response)
            current_roi_group.create_dataset("path_vertices", data = [x.vertices for x in self.roi_path])
            current_roi_group.create_dataset("roi_image", data = self.roi_image)
 
    def loadRois(self, roi_set_name):
        with h5py.File(os.path.join(self.flystim_data_directory, self.file_name) + '.hdf5','r') as experiment_file:
            roi_set_group = experiment_file['/epoch_runs'].get(str(self.series_number)).get('rois').get(roi_set_name)

            self.roi_mask = list(roi_set_group.get("roi_mask")[:]) #load from hdf5 metadata file
            self.roi_response = list(roi_set_group.get("roi_response")[:])
            self.roi_path = [path.Path(x) for x in list(roi_set_group.get("path_vertices")[:])]
            self.roi_image = roi_set_group.get("roi_image")[:]

        self.getResponseTraces()
        
    def filterResponseTraces(self, window_size = 5):
        for ind, rr in enumerate(self.roi_response):
            self.roi_response[ind] = np.expand_dims(signal.medfilt(np.squeeze(rr), window_size),0)
  
    def getResponseTraces(self, roi_response = None):
        if roi_response is not None:
            self.roi_response = roi_response # input roi response, e.g. online analysis

        self.time_vector, self.response_matrix = pa.ProtocolAnalysis.getEpochResponseMatrix(self)   

    def getAvailableROIsets(self):
        with h5py.File(os.path.join(self.flystim_data_directory, self.file_name) + '.hdf5','r+') as experiment_file:
            roi_parent_group = experiment_file['/epoch_runs'].get(str(self.series_number)).require_group('rois')
            roi_set_names = []
            for roi_set in roi_parent_group:
                roi_set_names.append(roi_set)
        
        return roi_set_names

# %% 
    ##############################################################################
    #Methods for image generation:
    ##############################################################################
    def generateRoiMap(self, scale_bar_length = 0):
        newImage = plot_tools.overlayImage(self.roi_image, self.roi_mask, 0.5, self.colors)
        
        fh = plt.figure(figsize=(4,4))
        ax = fh.add_subplot(111)
        ax.imshow(newImage)
        ax.set_aspect('equal')
        ax.set_axis_off()
        if scale_bar_length > 0:
            microns_per_pixel = float(self.metadata['micronsPerPixel_XAxis'])
            plot_tools.addImageScaleBar(ax, newImage, scale_bar_length, microns_per_pixel, 'lr')

    # TODO: do this by epoch response rather than entire, raw trace
    def getVoxelCorrelationHeatMap(self, roi_response = None):
        self.getResponseTraces()
        if roi_response is None:
            mean_roi_response = self.roi_response[0]
        else:
            mean_roi_response = roi_response
        
        x_dim = self.current_series.shape[1]
        y_dim = self.current_series.shape[2]
        
        self.heat_map =  np.empty(shape=(x_dim, y_dim), dtype=float)
        self.heat_map[:] = np.nan

        xx, yy = (vec.flatten() for vec in np.meshgrid(np.arange(0,x_dim), np.arange(0,y_dim)))
        for v_ind in range(len(xx)):
            x_loc = xx[v_ind]
            y_loc = yy[v_ind]
            current_voxel = self.current_series[:,x_loc,y_loc]
            new_corr_value = np.corrcoef(current_voxel,mean_roi_response)[0,1]
            
            self.heat_map[x_loc,y_loc] = new_corr_value

        fh = plt.figure()
        ax1 = fh.add_subplot(111)
        hmap = ax1.imshow(self.heat_map,vmin = np.min(-1), vmax = np.max(1), cmap=plt.cm.RdBu,interpolation='none')
        fh.colorbar(hmap, ax=ax1)
        ax1.set_axis_off()

        patch = patches.PathPatch(self.roi_path[0], facecolor='none', lw=1)
        ax1.add_patch(patch)
 
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
        

    def plotResponseTraces(self, roi_ind = 0):
        fig_handle = plt.figure(figsize=(4.5,3.25))
        ax_handle = fig_handle.add_subplot(111)
        ax_handle.plot(self.time_vector, self.response_matrix[roi_ind,:,:].T)
    
# %% 
    #####################################################################
    #Functions for extracting timing information from Bruker data
    #####################################################################
    
    def getAcquisitionTiming(self): #from bruker metadata (xml) file 
        """
        
        Bruker imaging acquisition is based on the bruker metadata file (xml)

        """

        metaData = ET.parse(os.path.join(self.image_data_directory, self.image_series_name) + '.xml')
            
        # Get acquisition times from imaging metadata
        root = metaData.getroot()
        stack_times = []
        frame_times = []

        # Single-plane, xy time series
        for child in root.find('Sequence').getchildren():
            frTime = child.get('relativeTime')
            if frTime is not None:
                stack_times.append(float(frTime))        
        stack_times = np.array(stack_times)
        stack_times = stack_times[1:] #trim extra 0 at start
        frame_times = stack_times
            
        stack_times = stack_times * 1e3 # -> msec
        frame_times = frame_times * 1e3 # -> msec
        sample_period = np.mean(np.diff(stack_times)) # in msec
        return {'stack_times':stack_times, 'frame_times':frame_times, 'sample_period':sample_period }
  
# %%         
    ####################################################################################
    #Functions for reading stimulus metadata from flystim data file(s) and bruker XML
    ####################################################################################
        
    def getEpochGroupMetadata(self):
        """
        
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
                else:
                    new_epoch_params = addAttributesToDictionary(series_group[epoch],{}) #epoch_time is the only attribute at this level
                    new_epoch_params = addAttributesToDictionary(series_group[epoch + '/epoch_parameters'], new_epoch_params)
                    new_epoch_params = addAttributesToDictionary(series_group[epoch + '/convenience_parameters'], new_epoch_params)
                    epoch_parameters.append(new_epoch_params)
                
            # sort epochs by start time
            epoch_parameters = sorted(epoch_parameters, key=itemgetter('epoch_time')) 
            
        return epoch_parameters, run_parameters, notes
    
    def getPVMetadata(self):
        metaData = ET.parse(os.path.join(self.image_data_directory, self.image_series_name) + '.xml')
        root = metaData.getroot()

        metadata = {}
        for child in list(root.find('PVStateShard')):
            if child.get('value') is None:
                for subchild in list(child):
                    new_key = child.get('key') + '_' + subchild.get('index')
                    new_value = subchild.get('value')
                    metadata[new_key] = new_value
        
            else:
                new_key = child.get('key')
                new_value = child.get('value')
                metadata[new_key] = new_value
        
        
        metadata['version'] = root.get('version')
        metadata['date'] = root.get('date')
        metadata['notes'] = root.get('notes')

        return metadata

def getEpochAndFrameTiming(image_data_directory, image_series_name, 
                           plot_trace_flag = True,
                           threshold = 0.6, 
                           minimum_epoch_separation = 2e3, # datapoints
                           frame_slop = 10, #datapoints +/- ideal frame duration
                           command_frame_rate = 115, #Hz
                           v_rec_suffix = '_Cycle00001_VoltageRecording_001'):
    
    """
    Stimulus (epoch) timing is based on the frame monitor trace, which is saved out as a 
        .csv file with each image series. Assumes a frame monitor signal that flips from 
        0 to 1 every other frame of a presentation and is 0 between epochs.

    """

    #photodiode metadata:
    metadata = ET.parse(os.path.join(image_data_directory, image_series_name) + v_rec_suffix + '.xml')
    root = metadata.getroot()
    rate_node = root.find('Experiment').find('Rate')
    sample_rate = int(rate_node.text)
    
    active_channels = []
    signal_list = root.find('Experiment').find('SignalList').getchildren()
    for signal_node in signal_list:
        is_channel_active = signal_node.find('Enabled').text
        channel_name = signal_node.find('Name').text
        if is_channel_active == 'true':
            active_channels.append(channel_name)

    # Load frame tracker signal and pull frame/epoch timing info
    data_frame = pd.read_csv(os.path.join(image_data_directory, image_series_name) + v_rec_suffix + '.csv');
    
    tt = data_frame.loc[:]['Time(ms)']
    #for now takes first enabled channel. 
    #TODO: Change to handle multiple photodiode signals
    frame_monitor = data_frame.loc[:][' ' + active_channels[0]]
    # shift & normalize so frame monitor trace lives on [0 1]
    frame_monitor = frame_monitor - np.min(frame_monitor)
    frame_monitor = frame_monitor / np.max(frame_monitor)
    
    # find lightcrafter frame flip times
    V_orig = frame_monitor.iloc[0:-2]
    V_shift = frame_monitor.iloc[1:-1]
    ups = np.where(np.logical_and(V_orig.values < threshold, V_shift.values >= threshold))[0] + 1
    downs = np.where(np.logical_and(V_orig.values >= threshold, V_shift.values < threshold))[0] + 1
    frame_times = np.sort(np.append(ups,downs))
    
    # Use frame flip times to find stimulus start times
    stimulus_start_frames = np.append(0, np.where(np.diff(frame_times) > minimum_epoch_separation)[0] + 1)
    stimulus_end_frames = np.append(np.where(np.diff(frame_times) > minimum_epoch_separation)[0],len(frame_times)-1)
    stimulus_start_times = frame_times[stimulus_start_frames] / sample_rate * 1e3 # datapoints -> msec
    stimulus_end_times = frame_times[stimulus_end_frames] / sample_rate * 1e3 # datapoints -> msec
    
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
        pylab.plot(tt,frame_monitor)
        pylab.plot(tt.iloc[frame_times],threshold * np.ones(frame_times.shape),'ko')
        pylab.plot(stimulus_start_times, threshold * np.ones(stimulus_start_times.shape),'go')
        pylab.plot(stimulus_end_times, threshold * np.ones(stimulus_end_times.shape),'ro')
        pylab.plot(frame_times[dropped_frame_inds] / sample_rate * 1e3, 1 * np.ones(dropped_frame_inds.shape),'ro')
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