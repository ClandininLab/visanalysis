from visanalysis.imaging_data import AodScopeData, BrukerData

import matplotlib.pyplot as plt
import numpy as np

# AODscope poi data:
file_name = '2019-06-28'
series_number = 3
ImagingData = AodScopeData.ImagingDataObject(file_name, series_number)
roi_name = 'axon'

# Bruker data:
# file_name = '2018-11-06'
# series_number = 9
# ImagingData = BrukerData.ImagingDataObject(file_name, series_number)
# roi_name = 'multi_dendrite'


# %%
fh = plt.figure(figsize=(24, 12))
for roi in range(ImagingData.roi[roi_name]['epoch_response'].shape[0]):
    if roi > 11:
        continue
    ax = fh.add_subplot(3,4,roi+1)
    time_vector = ImagingData.roi[roi_name]['time_vector']
    no_trials = ImagingData.roi[roi_name]['epoch_response'][roi,:,:].shape[0]
    current_mean = np.mean(ImagingData.roi[roi_name]['epoch_response'][roi,:,:], axis = 0)
    current_std = np.std(ImagingData.roi[roi_name]['epoch_response'][roi,:,:], axis = 0)
    current_sem = current_std / np.sqrt(no_trials)

    ax.plot(time_vector, current_mean, 'k')
    ax.fill_between(time_vector, current_mean - current_sem, current_mean + current_sem, alpha = 0.5)

# %%

ImagingData.generateRoiMap(roi_name, scale_bar_length=20)
# %% Use analysis functions
from visanalysis.analysis import shared_analysis
fh = plt.figure(figsize=(24, 4))
fig_handle = shared_analysis.plotResponseByCondition(ImagingData, roi_name, eg_ind = 0, condition = 'current_location', fig_handle = fh)
