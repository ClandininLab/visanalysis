# -*- coding: utf-8 -*-
"""
Created on Fri May 31 09:43:59 2019

@author: mhturner
"""
import os
from visanalysis import imaging_data
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import h5py

class BrukerDataObject(imaging_data.ImagingData.ImagingDataObject):
    def __init__(self, file_name, series_number):
        super().__init__(file_name, series_number) #call the parent class init
        # Image series is of the format: TSeries-YYYYMMDD-00n
        self.image_series_name = 'TSeries-' + file_name.replace('-','') + '-' + ('00' + str(series_number))[-3:]

        # Get timing info for acquisition and stimulus
        self.getAcquisitionTiming()
        self.getStimulusTiming()
        
        # Get epoch responses for rois
        self.getEpochResponses()
        self.checkEpochNumberCount()
        
        self.metadata = self.getPVMetadata()
        
    def getEpochResponses(self):
        """
        Assigns:
            -roi (dict), with (at minimum) keys 'roi_response', 'epoch_response' and 'time_vector'
        """
        with h5py.File(os.path.join(self.flystim_data_directory, self.file_name) + '.hdf5','r') as experiment_file:
            roi_group = experiment_file['/epoch_runs'].get(str(self.series_number)).get('rois')
            
            self.roi = {}
            for gr in roi_group:
                new_roi = {}
                if type(roi_group.get(gr)) is h5py._hl.group.Group:
                    new_roi['roi_mask'] = list(roi_group.get(gr).get("roi_mask")[:])
                    new_roi['roi_image'] = list(roi_group.get(gr).get("roi_image")[:])
                    
                    new_roi['roi_path'] = roi_group.get(gr).get("path_vertices")[:]
#                    ind = 0
#                    while new_path is not None:
#                        new_roi['roi_path'].append(new_path)
#                        ind += 1
#                        new_path = roi_group.get(gr).get("path_vertices_" + str(ind))
                            
                    new_roi['roi_response'] = np.squeeze(roi_group.get(gr).get("roi_response")[:], axis = 1)
                    
                    time_vector, response_matrix = self.getEpochResponseMatrix(respose_trace = new_roi['roi_response'])
                    new_roi['epoch_response'] = response_matrix
                    new_roi['time_vector'] = time_vector
                    
                    self.roi[gr] = new_roi

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
            
        stack_times = stack_times # sec
        frame_times = frame_times # sec
        sample_period = np.mean(np.diff(stack_times)) # sec
        self.response_timing = {'stack_times':stack_times, 'frame_times':frame_times, 'sample_period':sample_period }
            
      

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

    def getStimulusTiming(self, v_rec_suffix = '_Cycle00001_VoltageRecording_001'):
        
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
        
        self.stimulus_timing = self.getEpochAndFrameTiming(tt, frame_monitor, sample_rate)