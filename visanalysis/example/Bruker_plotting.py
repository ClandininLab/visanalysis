# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 11:12:26 2019

@author: mhturner
"""
import matplotlib.pyplot as plt
import numpy as np
   
from visanalysis.imaging_data import BrukerData

file_name = '2019-06-13'
series_number = 1

ImagingData = BrukerData.ImagingDataObject(file_name, series_number)


for k in ImagingData.roi.keys():
    print(k)


# %% Plot roi responses
roi_name = 'heather_1'
    
fh = plt.figure()
for roi in range(ImagingData.roi[roi_name]['epoch_response'].shape[0]):
    ax = fh.add_subplot(3,2,roi+1)
    time_vector = ImagingData.roi[roi_name]['time_vector']
    no_trials = ImagingData.roi[roi_name]['epoch_response'][roi,:,:].shape[0]
    current_mean = np.mean(ImagingData.roi[roi_name]['epoch_response'][roi,:,:], axis = 0)
    current_std = np.std(ImagingData.roi[roi_name]['epoch_response'][roi,:,:], axis = 0)
    current_sem = current_std / np.sqrt(no_trials)
    
    ax.plot(time_vector, current_mean, 'k')
    ax.fill_between(time_vector, current_mean - current_sem, current_mean + current_sem, alpha = 0.5)
    

# %% Use analysis functions
from visanalysis.analysis import shared_analysis

fig_handle = shared_analysis.plotResponseByCondition(ImagingData, roi_name, eg_ind = 0, condition = 'intensity', fig_handle = None)

ImagingData.generateRoiMap(roi_name, scale_bar_length=20)
fig_handle = shared_analysis.plotRoiResponses(ImagingData, roi_name, fig_handle = None)