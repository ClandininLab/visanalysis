# -*- coding: utf-8 -*-
"""
Created on Fri May 31 09:43:59 2019

@author: mhturner
"""
import os
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import h5py
import skimage.io as io
from registration import CrossCorr
import matplotlib.patches as patches
import warnings
from scipy import stats

from visanalysis import imaging_data
from visanalysis import plot_tools

class ImagingDataObject(imaging_data.ImagingData.ImagingDataObject):
    def __init__(self, file_name, series_number, load_rois = True):
        super().__init__(file_name, series_number) #call the parent class init
        # Image series is of the format: TSeries-YYYYMMDD-00n
        self.image_series_name = 'TSeries-' + file_name.replace('-','') + '-' + ('00' + str(series_number))[-3:]

        # Get timing info for acquisition and stimulus
        self.__getAcquisitionTiming()
        self.__getStimulusTiming()
        self.__checkEpochNumberCount()
        
        self.metadata = self.__getPVMetadata()
        
        if load_rois:
            # Get epoch responses for rois
            self.getEpochResponses()

    def getEpochResponses(self):
        """
        Assigns:
            -self.roi (dict): each key-value pair is a roi set name and dictionary
                Each component dict has keys 'roi_response', 'epoch_response' and 'time_vector'
                    roi_response: ndarray, shape = (n_rois, n_timepoints_per_series). Entire image series trace
                    epoch_response: ndarray, shape = (n_rois, n_trials, n_timepoints_per_trial)
                    time_vector: ndarray, shape = (n_timepoints_per_trial,) (sec.)
        """
        with h5py.File(os.path.join(self.flystim_data_directory, self.file_name) + '.hdf5','r') as experiment_file:
            roi_group = experiment_file['/epoch_runs'].get(str(self.series_number)).get('rois')
            if roi_group is None:
                warnings.warn("!!No rois found for this image series!!")
                return
            
            self.roi = {}
            for gr in roi_group:
                new_roi = {}
                if type(roi_group.get(gr)) is h5py._hl.group.Group:
                    new_roi['roi_mask'] = list(roi_group.get(gr).get("roi_mask")[:])
                    new_roi['roi_image'] = list(roi_group.get(gr).get("roi_image")[:])
                                        
                    new_roi['roi_path'] = []
                    new_path = roi_group.get(gr).get("path_vertices_0")
                    ind = 0
                    while new_path is not None:
                        new_roi['roi_path'].append(new_path)
                        ind += 1
                        new_path = roi_group.get(gr).get("path_vertices_" + str(ind))
                        
                    new_roi['roi_path'] = [x[:] for x in new_roi['roi_path']]

                    new_roi['roi_response'] = np.squeeze(roi_group.get(gr).get("roi_response")[:], axis = 1)
                    
                    time_vector, response_matrix = self.getEpochResponseMatrix(respose_trace = new_roi['roi_response'])
                    new_roi['epoch_response'] = response_matrix
                    new_roi['time_vector'] = time_vector
                    
                    self.roi[gr] = new_roi

# %%        
    ##############################################################################
    #Image plotting functions
    ##############################################################################
    def generateRoiMap(self, roi_name, scale_bar_length = 0):
        newImage = plot_tools.overlayImage(self.roi.get(roi_name).get('roi_image'), self.roi.get(roi_name).get('roi_mask'), 0.5, self.colors)
        
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
    #Functions for image series data
    ##############################################################################
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
    #Functions for volumetric data
    ##############################################################################       \
    def getTrialAlignedVoxelResponses(self, brain):
        #brain is shape (x,y,z,t)
        
        #zero values are from registration. Replace with nan
        brain[np.where(brain ==0)] = np.nan
        #set to minimum
        brain[np.where(np.isnan(brain))] = np.nanmin(brain)
        
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
        time_vector = np.arange(0,epoch_frames) * sample_period # sec
        
        no_trials = len(epoch_start_times)
        brain_trial_matrix = np.ndarray(shape=(brain.shape[0], brain.shape[1], brain.shape[2], no_trials, epoch_frames), dtype='float32') #x, y, z, trials, time_vector
        brain_trial_matrix[:] = np.nan
        cut_inds = np.empty(0, dtype = int)
        for idx, val in enumerate(epoch_start_times):
            stack_inds = np.where(np.logical_and(stack_times < epoch_end_times[idx], stack_times >= epoch_start_times[idx]))[0]
            if len(stack_inds) == 0: #no imaging acquisitions happened during this epoch presentation
                cut_inds = np.append(cut_inds,idx)
                continue
            if np.any(stack_inds > brain.shape[3]):
                cut_inds = np.append(cut_inds,idx)
                continue
            if idx is not 0:
                if len(stack_inds) < epoch_frames: #missed images for the end of the stimulus
                    cut_inds = np.append(cut_inds,idx)
                    continue
                
            #Get voxel responses for this epoch
            new_resp_chunk = brain[:,:,:,stack_inds]

            brain_trial_matrix[:,:,:,idx] = new_resp_chunk[:,:,:,0:epoch_frames]
            
        brain_trial_matrix = np.delete(brain_trial_matrix, cut_inds, axis = 4)

        return time_vector, brain_trial_matrix

    def getConcatenatedMeanVoxelResponses(self, brain_trial_matrix, parameter_keys):
        parameter_values = np.ndarray((len(self.epoch_parameters), len(parameter_keys)))
        for ind_e, ep in enumerate(self.epoch_parameters):
            for ind_k, k in enumerate(parameter_keys):
                new_val = ep.get(k)
                if new_val == 'elevation':
                    new_val = 0
                elif new_val == 'azimuth':
                    new_val = 90
                parameter_values[ind_e, ind_k] = new_val
                
        
        unique_parameter_values = np.unique(parameter_values, axis = 0)

        pre_frames = int(self.run_parameters['pre_time'] / self.response_timing.get('sample_period'))
        stim_frames = int(self.run_parameters['stim_time'] / self.response_timing.get('sample_period'))
        tail_frames = int(self.run_parameters['tail_time'] / self.response_timing.get('sample_period'))
        
        x_dim = brain_trial_matrix.shape[0]
        y_dim = brain_trial_matrix.shape[1]
        z_dim = brain_trial_matrix.shape[2]
        
        mean_resp = []
        std_resp = []
        p_values = []
        for up in unique_parameter_values:
            pull_inds = np.where((up == parameter_values).all(axis = 1))[0]
            
            #get baseline timepoints for each voxel
            baseline_pre = brain_trial_matrix[:,:,:,pull_inds,0:pre_frames]
            baseline_tail = brain_trial_matrix[:,:,:,pull_inds,-int(tail_frames/2):]
            baseline_points = np.concatenate((baseline_pre, baseline_tail), axis = 4)
            
            #dF/F
            baseline = np.mean(baseline_points, axis = (3,4))
            baseline = np.expand_dims(np.expand_dims(baseline, axis = 3), axis = 4)
            response_dff = (brain_trial_matrix[:,:,:,pull_inds,:] - baseline) / baseline
            baseline_dff = (baseline_points - baseline) / baseline
#            
#            #set any nans (baseline 0) to 0. 0s come from registration
#            baseline_dff[np.isnan(baseline_dff)] = 0
#            response_dff[np.isnan(response_dff)] = 0
#            
            _, p = stats.ttest_ind(np.reshape(baseline_dff, (x_dim, y_dim, z_dim, -1)), 
                                   np.reshape(response_dff[:,:,:,:,pre_frames:(pre_frames+stim_frames)], (x_dim, y_dim, z_dim, -1)), axis = 3)
        
            p_values.append(p)
            
            mean_resp.append(np.mean(response_dff, axis = 3))
            std_resp.append(np.std(response_dff, axis = 3))
        
        p_values = np.stack(p_values, axis = 3) #x y z stimulus
        mean_resp = np.concatenate(mean_resp, axis = 3)
        std_resp = np.concatenate(std_resp, axis = 3)
        
        return mean_resp, std_resp, p_values, unique_parameter_values
    
    def nonresponsiveVoxelsToNan(self, response_matrix, p_values, p_cutoff = 0.01, significant_stimuli = 3):
        sig_responses = (p_values < p_cutoff).sum(axis = 3)
        responsive_voxels = sig_responses > significant_stimuli
        response_matrix[~responsive_voxels] = np.nan

        return response_matrix

        
# %%        
    ##############################################################################
    #Private functions for timing and metadata
    ##############################################################################

    def __getAcquisitionTiming(self): #from bruker metadata (xml) file 
        """
        Bruker imaging acquisition metadata based on the bruker metadata file (xml)
        """
        metaData = ET.parse(os.path.join(self.image_data_directory, self.image_series_name) + '.xml')
            
        # Get acquisition times from imaging metadata
        root = metaData.getroot()
        
        if root.find('Sequence').get('type') == 'TSeries ZSeries Element':
            # volumetric xyz time series
            num_t = len(root.findall('Sequence'))
            num_z = len(root.find('Sequence').findall('Frame'))
            frame_times = np.ndarray(shape = (num_t, num_z), dtype = float)
            frame_times[:] = np.nan
            for t_ind, t_step in enumerate(root.findall('Sequence')):
                for z_ind, z_step in enumerate(t_step.findall('Frame')):
                    frame_times[t_ind,z_ind] = z_step.get('relativeTime')
                    
            stack_times = frame_times[:,0]
            sample_period = np.mean(np.diff(stack_times))
            
            self.response_timing = {'frame_times':frame_times, 'stack_times':stack_times, 'sample_period':sample_period}
            
        elif root.find('Sequence').get('type') == 'TSeries Timed Element':     
            # Single-plane, xy time series
            stack_times = []
            for frame in root.find('Sequence').findall('Frame'):
                frTime = frame.get('relativeTime')
                stack_times.append(float(frTime))
                
            stack_times = np.array(stack_times)
        
            sample_period = np.mean(np.diff(stack_times)) #. sec
            self.response_timing = {'stack_times':stack_times, 'sample_period':sample_period}
            
            
    def __getPVMetadata(self):
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

    def __getStimulusTiming(self, v_rec_suffix = '_Cycle00001_VoltageRecording_001'):
        
        """
        Stimulus (epoch) timing is based on the frame monitor trace, which is saved out as a 
            .csv file with each image series. Assumes a frame monitor signal that flips from 
            0 to 1 every other frame of a presentation and is 0 between epochs.
        
        """
        
        #photodiode metadata:
        metadata = ET.parse(os.path.join(self.image_data_directory, self.image_series_name) + v_rec_suffix + '.xml')
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
        data_frame = pd.read_csv(os.path.join(self.image_data_directory, self.image_series_name) + v_rec_suffix + '.csv');
        
        tt = data_frame.get('Time(ms)').values / 1e3 #sec
        #for now takes first enabled channel. 
        #TODO: Change to handle multiple photodiode signals
        frame_monitor = data_frame.get(' ' + active_channels[0]).values
        
        self.stimulus_timing = self.getEpochAndFrameTiming(tt, frame_monitor, sample_rate, plot_trace_flag = True, command_frame_rate = 115)