from visanalysis.analysis import imaging_data
import matplotlib.pyplot as plt
import numpy as np

experiment_file_directory = r'C:\Users\mhturner\Dropbox\ClandininLab\CurrentData\FlystimData'
experiment_file_name = '2019-08-26'
series_number = 10

roi_name = 'multi_dendrite'

ID = imaging_data.ImagingDataObject(experiment_file_directory, experiment_file_name, series_number)


ID.getRoi(roi_name)

ID.roi.keys()
len(ID.roi.get('roi_response'))

ID.getEpochResponseMatrix()

ID.response.get('response_matrix').shape

plt.plot(np.mean(ID.response.get('response_matrix')[0,:,:], axis = 0))
