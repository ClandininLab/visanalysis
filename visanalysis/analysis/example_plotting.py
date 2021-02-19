from visanalysis.analysis import imaging_data, shared_analysis
import matplotlib.pyplot as plt
import numpy as np

experiment_file_directory = '/home/mhturner/CurrentData'
experiment_file_name = '2021-02-17'
series_number = 8

kwargs = {'plot_trace_flag': True}

ID = imaging_data.ImagingDataObject(experiment_file_directory, experiment_file_name, series_number, kwargs=kwargs)

# %%
roi_set_name = 'terms'
ID.getRoiResponses()
ID.roi.get(roi_set_name)['roi_response'][0].shape

# %%
ID.getRoiResponses(background_subtraction=True)

# %%
roi_name = 'terms'
shared_analysis.plotResponseByCondition(ID, roi_name=roi_name, condition='current_intensity', eg_ind=2)

# %%
shared_analysis.plotRoiResponses(ID, roi_name=roi_name)

# %%
ID.generateRoiMap(roi_name=roi_name)
