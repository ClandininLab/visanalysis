
from visanalysis.imaging_data import BrukerData

file_name = '2019-06-13'
series_number = 3

ImagingData = BrukerData.ImagingDataObject(file_name, series_number, load_rois=False)
ImagingData.loadImageSeries()

# %% Choose Rois
from visanalysis import region
MRS = region.MultiROISelector(ImagingData, roiType = 'freehand', roiRadius = 2)


