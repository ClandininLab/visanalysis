"""
Example script to interact with ImagingDataObject, and extract data.

https://github.com/ClandininLab/visanalysis
mhturner@stanford.edu
"""
# %%
from visanalysis.analysis import imaging_data, shared_analysis
from visanalysis.util import plot_tools
import matplotlib.pyplot as plt
import os

experiment_file_directory = '/Users/mhturner/GitHub/visanalysis/example_data/responses/Bruker'
experiment_file_name = '2021-07-07'
series_number = 1

file_path = os.path.join(experiment_file_directory, experiment_file_name + '.hdf5')

# ImagingDataObject wants a path to an hdf5 file and a series number from that file
ID = imaging_data.ImagingDataObject(file_path,
                                    series_number,
                                    quiet=False)

# %%

ID.getVolumeFrameOffsets()

# %% PARAMETERS & METADATA

# all run_parameters as a dict
run_parameters = ID.getRunParameters()

# specified run parameter
protocol_ID = ID.getRunParameters('protocol_ID')
print(protocol_ID)

# epoch_parameters: list of dicts of all epoch parameters, one for each epoch (trial)
epoch_parameters = ID.getEpochParameters()
# Pass a param key to return a list of specified param values, one for each trial
current_rv_ratio = ID.getEpochParameters('current_intensity')

# fly_metadata: dict
fly_metadata = ID.getFlyMetadata()
prep = ID.getFlyMetadata('prep')
print(prep)

# acquisition_metadata: dict
acquisition_metadata = ID.getAcquisitionMetadata()
sample_period = ID.getAcquisitionMetadata('sample_period')
print(sample_period)
# %% ROIS AND RESPONSES

# Get list of rois present in the hdf5 file for this series
roi_set_names = ID.getRoiSetNames()
roi_set_names
# getRoiResponses() wants a ROI set name, returns roi_data (dict)
roi_data = ID.getRoiResponses('set_2')
roi_data.keys()

# See the ROI overlaid on top of the image
ID.generateRoiMap(roi_name='set_2', z=1)

# Plot whole ROI response across entire series
fh0, ax0 = plt.subplots(1, 1, figsize=(12, 4))
ax0.plot(roi_data.get('roi_response')[0].T)
ax0.set_xlabel('Frame')
ax0.set_ylabel('Avg ROI intensity')

# Plot ROI response for trials 10 thru 15
# 'epoch_response' is shape (rois, trials, time)
fh1, ax1 = plt.subplots(1, 1, figsize=(6, 4))
ax1.plot(roi_data.get('time_vector'), roi_data.get('epoch_response')[0, 10:15, :].T)
ax1.set_ylabel('Response (dF/F)')
ax1.set_xlabel('Time (s)')

# %% Plot trial-average responses by specified parameter name

unique_parameter_values, mean_response, sem_response, trial_response_by_stimulus = ID.getTrialAverages(roi_data.get('epoch_response'), parameter_key='current_rv_ratio')
roi_data.keys()

fh, ax = plt.subplots(1, len(unique_parameter_values), figsize=(10, 2))
[x.set_ylim([-0.15, 0.25]) for x in ax.ravel()]
[plot_tools.cleanAxes(x) for x in ax.ravel()]
for u_ind, up in enumerate(unique_parameter_values):
    ax[u_ind].plot(roi_data['time_vector'], mean_response[:, u_ind, :].T, color='k')
    ax[u_ind].set_title('rv = {}'.format(up))
plot_tools.addScaleBars(ax[0], dT=2.0, dF=0.10, T_value=-0.5, F_value=-0.14)

# %% Some other convenience methods...



# Quickly look at roi responses (averaged across all trials)
shared_analysis.plotRoiResponses(ID, roi_name='glom')

# %%
shared_analysis.plotResponseByCondition(ID, roi_name='set_2', condition='current_intensity', eg_ind=0)
