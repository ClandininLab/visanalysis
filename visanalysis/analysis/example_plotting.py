from visanalysis.analysis import imaging_data, shared_analysis

experiment_file_directory = r'C:\Users\mhturner\Dropbox\ClandininLab\CurrentData\FlystimData'
experiment_file_name = '2019-08-26'
series_number = 3

roi_name = 'single_dendrite'

ID = imaging_data.ImagingDataObject(experiment_file_directory, experiment_file_name, series_number)

# %%
shared_analysis.plotResponseByCondition(ID, roi_name=roi_name, condition='current_width')

# %%
shared_analysis.plotRoiResponses(ID, roi_name=roi_name)

# %%
ID.generateRoiMap(roi_name=roi_name)
