from visanalysis.analysis import imaging_data, shared_analysis
import matplotlib.pyplot as plt

experiment_file_directory = '/Users/mhturner/Dropbox/ClandininLab/CurrentData/AODscope/'
experiment_file_name = '2019-12-03'
series_number = 2

ID = imaging_data.ImagingDataObject(experiment_file_directory, experiment_file_name, series_number)
# %%
ID.getRoiResponses()
ID.roi.get(roi_set_name)['roi_response'][0].shape

# %%
ID.getRoiResponses(background_subtraction=True)

# %%
roi_name = 'roi_set_name'
shared_analysis.plotResponseByCondition(ID, roi_name=roi_name, condition='component_stim_type')

# %%
shared_analysis.plotRoiResponses(ID, roi_name=roi_name)

# %%
ID.generateRoiMap(roi_name=roi_name)
