from visanalysis.analysis import imaging_data, shared_analysis
import matplotlib.pyplot as plt

experiment_file_directory = r'C:\Users\mhturner\Dropbox\ClandininLab\CurrentData'
experiment_file_name = '2019-11-13'
series_number = 5

ID = imaging_data.ImagingDataObject(experiment_file_directory, experiment_file_name, series_number)

# %%
roi_name = 'test_1'
shared_analysis.plotResponseByCondition(ID, roi_name=roi_name, condition='component_stim_type')

# %%
shared_analysis.plotRoiResponses(ID, roi_name=roi_name)

# %%
ID.generateRoiMap(roi_name=roi_name)
