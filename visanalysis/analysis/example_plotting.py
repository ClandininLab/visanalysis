from visanalysis.analysis import imaging_data, shared_analysis

experiment_file_directory = r'C:\Users\mhturner\Dropbox\ClandininLab\CurrentData\FlystimData'
experiment_file_name = '2019-09-17'
series_number = 1
roi_name = 'test1'

ID = imaging_data.ImagingDataObject(experiment_file_directory, experiment_file_name, series_number)

ID.run_parameters


# %%
shared_analysis.plotResponseByCondition(ID, roi_name=roi_name, condition='current_angle')

# %%
shared_analysis.plotRoiResponses(ID, roi_name=roi_name)

# %%
ID.generateRoiMap(roi_name=roi_name)
