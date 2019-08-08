
from visanalysis.imaging_data import BrukerData

file_name = '2019-06-13'
series_number = 1

ImagingData = BrukerData.ImagingDataObject(file_name, series_number, load_rois=False)
ImagingData.loadImageSeries()

# %% Choose Rois
from visanalysis import region
import sys
from PyQt5.QtWidgets import (QPushButton, QWidget, QGridLayout, QLineEdit, QComboBox, QApplication)

def run_app():
    app = QApplication(sys.argv)

    mainWin = region.MultiROISelector(ImagingData, roiType = 'freehand', roiRadius = 2)
    mainWin.show()
    app.exec_()
run_app()
