from visanalysis.analysis import imaging_data, shared_analysis
import matplotlib.pyplot as plt
import numpy as np

experiment_file_directory = '/home/mhturner/CurrentData'
experiment_file_name = '2021-04-21'
series_number = 1

kwargs = {'plot_trace_flag': True}

ID = imaging_data.ImagingDataObject(experiment_file_directory, experiment_file_name, series_number, kwargs=kwargs)


ID
# %%

ID.getRoiResponses()
roi_set_name = 'LC16'
ID.roi.get(roi_set_name)['roi_response'][0].shape

plt.plot(ID.roi.get(roi_set_name)['roi_response'][0].T)

ID.roi.get('LC16').get('epoch_response').shape

ID.roi.get('LC16').get('epoch_response')[:, 10, :]

ID.epoch_parameters[0]

# %%
ID.getRoiResponses(background_subtraction=True)

# %%
shared_analysis.plotResponseByCondition(ID, roi_name=roi_set_name, condition='component_stim_type', eg_ind=0)

# %%
shared_analysis.plotRoiResponses(ID, roi_name=roi_set_name)

# %%
ID.generateRoiMap(roi_name=roi_set_name, z=6)
