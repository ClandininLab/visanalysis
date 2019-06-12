
from visanalysis.imaging_data import BrukerData

file_name = '2018-11-06'
series_number = 9

ImagingData = BrukerData.ImagingDataObject(file_name, series_number)
ImagingData.loadImageSeries()

# %% Choose Rois
from visanalysis import region
MRS = region.MultiROISelector(ImagingData, roiType = 'freehand', roiRadius = 2)