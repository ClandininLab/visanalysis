from visanalysis.imaging_data import AodScopeData

import matplotlib.pyplot as plt
import numpy as np

file_name = '2019-06-06'
series_number = 10

ImagingData = AodScopeData.ImagingDataObject(file_name, series_number)

for k in ImagingData.roi.keys():
    print(k)

# %% Plot roi responses
roi_name = 'Med'
import seaborn as sns


fh = plt.figure()
no_pois = ImagingData.roi[roi_name]['epoch_response'].shape[0]
for roi in [6]:
    ax = fh.add_subplot(1,1,1)
    time_vector = ImagingData.roi[roi_name]['time_vector']
    no_trials = ImagingData.roi[roi_name]['epoch_response'][roi,:,:].shape[0]
    current_mean = np.mean(ImagingData.roi[roi_name]['epoch_response'][roi,:,:], axis = 0)
    current_std = np.std(ImagingData.roi[roi_name]['epoch_response'][roi,:,:], axis = 0)
    current_sem = current_std / np.sqrt(no_trials)

    ax.plot(time_vector, current_mean, 'r')
    colors = sns.color_palette("deep",n_colors = no_trials)
    ax.plot(time_vector, ImagingData.roi[roi_name]['epoch_response'][roi,:,:].T, alpha = 0.1)
#    ax.fill_between(time_vector, current_mean - current_std, current_mean + current_std, alpha = 0.5)


fh2 = plt.figure()
ax = fh2.add_subplot(111)
ax.imshow(ImagingData.poi_data['poi_overlay'])

# %%

fh = plt.figure()
ax = fh.add_subplot(111)
ax.plot(ImagingData.roi.get('M2').get('roi_response')[6,:])
ax.plot(ImagingData.roi.get('bg').get('roi_response')[0,:],'r')

ax.plot(ImagingData.roi.get('M2').get('roi_response')[6,:] - ImagingData.roi.get('bg').get('roi_response')[0,:], 'b')


# %% Use analysis functions
from visanalysis.analysis import shared_analysis

fig_handle = shared_analysis.plotResponseByCondition(ImagingData, roi_name, eg_ind = 5, condition = 'current_angle', fig_handle = None)


fig_handle = shared_analysis.plotRoiResponses(ImagingData, roi_name, fig_handle = None)

ImagingData.generateRoiMap(roi_name, scale_bar_length=20)
