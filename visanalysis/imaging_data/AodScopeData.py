# -*- coding: utf-8 -*-
"""
Created on Fri May 31 09:45:02 2019

@author: mhturner
"""
import h5py
import os
import numpy as np
from matplotlib import path


from visanalysis import plot_tools
from visanalysis import imaging_data

class AodScopeDataObject(imaging_data.ImagingData.ImagingDataObject):
    def __init__(self, file_name, series_number):
        super().__init__(file_name, series_number) #call the parent class init method
        
        self.getPoiData()
        
        # Get timing info for acquisition and stimulus
        self.getAcquisitionTiming()
        self.getStimulusTiming()
        
        # Get epoch responses for rois
        self.getEpochResponses()
        self.checkEpochNumberCount()

    def getPoiData(self):
        with h5py.File(os.path.join(self.flystim_data_directory, self.file_name) + '.hdf5','r') as experiment_file:
            poi_group = experiment_file['/epoch_runs'].get(str(self.series_number)).get('pois')
            
            self.poi_data = {}
            self.poi_data['poi_data_matrix'] = poi_group.get('poi_data_matrix')[:]
            self.poi_data['poi_time'] = poi_group.get('time_points')[:]
            
            self.poi_data['photodiode_time'] = poi_group.get('photodiode_time')[:]
            self.poi_data['photodiode_input'] = poi_group.get('photodiode_input')[:]
            
            self.poi_data['snap_image'] = poi_group.get('snap_image')[:]
            
            self.poi_data['poi_map'] = poi_group.get('poi_map')[:]
            self.poi_data['poi_locations']  = poi_group.get('poi_locations')[:]
            

            poi_mask = []
            pixX = np.arange(self.poi_data['snap_image'].shape[1])
            pixY = np.arange(self.poi_data['snap_image'].shape[0])
            yv, xv = np.meshgrid(pixX, pixY)
            pix = np.vstack((yv.flatten(), xv.flatten())).T
            
            for poi_loc in self.poi_data['poi_locations']:
                center = poi_loc
                new_roi_path = path.Path.circle(center = center, radius = 5)
                ind = new_roi_path.contains_points(pix, radius=0.5)
                
                array = np.zeros(self.poi_data['snap_image'].shape)
                lin = np.arange(array.size)
                newArray = array.flatten()
                newArray[lin[ind]] = 1
                poi_mask.append(newArray.reshape(array.shape))

#            poi_mask[self.poi_data['poi_locations'][:,1], self.poi_data['poi_locations'][:,0]] = 1
#   
            poi_overlay = plot_tools.overlayImage(self.poi_data['snap_image'], poi_mask, 1.0, self.colors)
            
            self.poi_data['poi_overlay'] = poi_overlay
            
    
        
    def getAcquisitionTiming(self):
        #get random access timing info
        self.response_timing = {}
        self.response_timing['stack_times'] = self.poi_data['poi_time'] / 1e3 #msec -> sec
        self.response_timing['sample_period'] = np.unique(np.diff(self.poi_data['poi_time']))[0] / 1e3 #msec -> sec
        
    def getStimulusTiming(self):
        #get stimulus timing info from photodiode
        sample_rate = 1e4
        self.stimulus_timing = self.getEpochAndFrameTiming(self.poi_data['photodiode_time'], self.poi_data['photodiode_input'], sample_rate, plot_trace_flag = False)
        
    def getEpochResponses(self):
        """
        Assigns:
            -roi (dict), with (at minimum) keys 'roi_response', 'epoch_response' and 'time_vector'
        """
        with h5py.File(os.path.join(self.flystim_data_directory, self.file_name) + '.hdf5','r') as experiment_file:
            poi_group = experiment_file['/epoch_runs'].get(str(self.series_number)).get('pois')
            
            self.roi = {}
            #Get background point trace, if it exists
            if poi_group.get('background'):
                dark_number = poi_group.get('background').get('poi_numbers')[:]
                dark_trace = self.poi_data['poi_data_matrix'][dark_number,:]
            else:
                dark_trace = 0
                            
            for gr in poi_group:
                new_roi = {}
                if type(poi_group.get(gr)) is h5py._hl.group.Group:
                    poi_numbers = poi_group.get(gr).get('poi_numbers')[:]
                    new_roi['roi_response'] = self.poi_data['poi_data_matrix'][poi_numbers,:] - dark_trace #dark subtract to remove stimulus artifact
                    
                    time_vector, response_matrix = self.getEpochResponseMatrix(respose_trace = new_roi['roi_response'])
                    
                    new_roi['epoch_response'] = response_matrix
                    new_roi['time_vector'] = time_vector
                    
                    self.roi[gr] = new_roi
            