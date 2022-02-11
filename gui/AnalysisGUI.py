#!/usr/bin/env python3
"""
ROI drawer for bruker image series.

https://github.com/ClandininLab/visanalysis
mhturner@stanford.edu
"""
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib import path
import matplotlib.colors as mcolors
from matplotlib.widgets import LassoSelector, EllipseSelector
import matplotlib.cm as cm
from PyQt5.QtWidgets import (QPushButton, QWidget, QLabel, QGridLayout,
                             QApplication, QComboBox, QLineEdit, QFileDialog, QSlider)
import PyQt5.QtCore as QtCore
import PyQt5.QtGui as QtGui
import numpy as np
import os
import nibabel as nib
from visanalysis.plugin import bruker

from visanalysis.util import plot_tools

import pickle


class AnalysisGUI(QWidget):

    def __init__(self):
        super().__init__()

        self.max_rois = 50
        self.roi_type = 'freehand'
        self.roi_radius = None
        self.existing_roi_set_paths = {}

        self.image_filename = ''
        self.image_filepath = ''
        self.roi_filename = ''
        self.roi_filepath = ''

        self.markpoints_metadata = None
        self.image_metadata = None

        self.channel = 1 # index

        self.image_series = None
        self.roi_image = None

        self.current_roi_index = 0
        self.current_z_slice = 0

        self.roi_response = []
        self.roi_mask = []
        self.roi_path = []
        self.roi_path_list = []

        self.blank_image = np.zeros((1, 1))
        self.colors = [mcolors.to_rgb(x) for x in list(mcolors.TABLEAU_COLORS)[:20]]

        self.initUI()

    def initUI(self):
        self.grid = QGridLayout(self)

        self.file_control_grid = QGridLayout()
        self.file_control_grid.setSpacing(3)
        self.grid.addLayout(self.file_control_grid, 0, 0)

        self.roi_control_grid = QGridLayout()
        self.roi_control_grid.setSpacing(3)
        self.grid.addLayout(self.roi_control_grid, 1, 0)

        self.plot_grid = QGridLayout()
        self.plot_grid.setSpacing(3)
        self.grid.addLayout(self.plot_grid, 2, 0)

        # # # # File control browser: # # # # # # # # (0,0)
        load_button = QPushButton("Load image file", self)
        load_button.clicked.connect(self.selectImage)
        # Label with current expt file
        self.image_filename_label = QLabel(self.image_filename)
        self.file_control_grid.addWidget(load_button, 0, 0)
        self.file_control_grid.addWidget(self.image_filename_label, 1, 0)

        roi_file_button = QPushButton("Select roi file", self)
        roi_file_button.clicked.connect(self.selectRoiFile)
        self.file_control_grid.addWidget(roi_file_button, 0, 1)
        self.roi_file_label = QLabel('')
        self.roi_file_label.setFont(QtGui.QFont('SansSerif', 8))
        self.file_control_grid.addWidget(self.roi_file_label, 1, 1)

        # Channel drop down
        ch_label = QLabel('Channel:')
        self.channel_combobox = QComboBox(self)
        self.channel_combobox.addItem("1")
        self.channel_combobox.addItem("0")
        self.channel_combobox.activated.connect(self.selectChannel)
        self.file_control_grid.addWidget(ch_label, 2, 0)
        self.file_control_grid.addWidget(self.channel_combobox, 2, 1)

        # Reload image series button
        reaload_button = QPushButton("Reload image", self)
        reaload_button.clicked.connect(self.reloadImage)
        self.file_control_grid.addWidget(reaload_button, 2, 2)

        # # # # Roi control # # # # # # # # (0, 1)
        # ROI type drop-down
        self.RoiTypeComboBox = QComboBox(self)
        self.RoiTypeComboBox.addItem("freehand")
        radii = [1, 2, 3, 4, 6, 8]
        for radius in radii:
            self.RoiTypeComboBox.addItem("circle:"+str(radius))
        self.RoiTypeComboBox.activated.connect(self.selectRoiType)
        self.roi_control_grid.addWidget(self.RoiTypeComboBox, 0, 0)

        # Clear all ROIs button
        self.clearROIsButton = QPushButton("Clear ROIs", self)
        self.clearROIsButton.clicked.connect(self.clearRois)
        self.roi_control_grid.addWidget(self.clearROIsButton, 0, 2)

        # Response display type dropdown
        self.roi_response_type_combobox = QComboBox(self)

        self.roi_response_type_combobox.addItem("RawTrace")
        self.roi_response_type_combobox.addItem("ZapResponse")
        self.roi_control_grid.addWidget(self.roi_response_type_combobox, 2, 2)

        # ROIset file name line edit box
        self.defaultRoiSetName = "roi_set_name"
        self.le_roiSetName = QLineEdit(self.defaultRoiSetName)
        self.roi_control_grid.addWidget(self.le_roiSetName, 1, 1)

        # Save ROIs button
        self.saveROIsButton = QPushButton("Save ROIs", self)
        self.saveROIsButton.clicked.connect(self.saveRois)
        self.roi_control_grid.addWidget(self.saveROIsButton, 1, 0)

        # Load ROI set combobox
        self.loadROIsComboBox = QComboBox(self)
        self.loadROIsComboBox.addItem("(load existing ROI set)")
        self.loadROIsComboBox.activated.connect(self.selectedExistingRoiSet)
        self.roi_control_grid.addWidget(self.loadROIsComboBox, 1, 2)
        self.updateExistingRoiSetList()

        # Delete current roi button
        self.deleteROIButton = QPushButton("Delete ROI", self)
        self.deleteROIButton.clicked.connect(self.deleteRoi)
        self.roi_control_grid.addWidget(self.deleteROIButton, 2, 0)

        # Current roi slider
        self.roiSlider = QSlider(QtCore.Qt.Horizontal, self)
        self.roiSlider.setMinimum(0)
        self.roiSlider.setMaximum(self.max_rois)
        self.roiSlider.valueChanged.connect(self.sliderUpdated)
        self.roi_control_grid.addWidget(self.roiSlider, 2, 1, 1, 1)

        plt.rc_context({'axes.edgecolor': 'white',
                        'xtick.color': 'white',
                        'ytick.color': 'white',
                        'figure.facecolor': 'black',
                        'axes.facecolor': 'black'})
        self.responseFig = plt.figure()
        self.responsePlot = self.responseFig.add_subplot(111)
        self.responseFig.subplots_adjust(left=0.1, bottom=0.4, top=0.95, right=0.98)
        self.responseCanvas = FigureCanvas(self.responseFig)
        self.responseCanvas.draw_idle()
        self.plot_grid.addWidget(self.responseCanvas, 0, 0)

        # # # # Image canvas # # # # # # # # (1, 2)
        self.roi_fig = plt.figure()
        self.roi_ax = self.roi_fig.add_subplot(111)
        self.roi_canvas = FigureCanvas(self.roi_fig)
        self.toolbar = NavigationToolbar(self.roi_canvas, self)
        self.roi_ax.set_aspect('equal')
        self.roi_ax.set_axis_off()
        self.plot_grid.addWidget(self.toolbar, 1, 0)
        self.plot_grid.addWidget(self.roi_canvas, 2, 0)
        self.plot_grid.setRowStretch(0, 1)
        self.plot_grid.setRowStretch(1, 3)
        self.plot_grid.setRowStretch(2, 3)

        # Current z slice slider
        self.zSlider = QSlider(QtCore.Qt.Horizontal, self)
        self.zSlider.setMinimum(0)
        self.zSlider.setMaximum(50)
        self.zSlider.setValue(0)
        self.zSlider.valueChanged.connect(self.zSliderUpdated)
        self.plot_grid.addWidget(self.zSlider, 3, 0)

        self.roi_fig.tight_layout()

        self.setWindowTitle('Visanalysis')
        self.setGeometry(200, 200, 600, 800)
        self.show()

    def selectImage(self):
        self.image_filepath, _ = QFileDialog.getOpenFileName(self, "Open image file (.nii)")
        self.image_filename = os.path.split(self.image_filepath)[1].split('.')[0]

        if self.image_filename != '':
            self.image_filename_label.setText(self.image_filename)
            self.loadImage(self.image_filepath, self.channel)

            # check if mark points metadata exists
            markpoints_fp = self.image_filepath.split('.')[0] + '_Cycle00001_MarkPoints.xml'
            if os.path.exists(markpoints_fp):
                self.markpoints_metadata = bruker.get_mark_points_metadata(markpoints_fp)
                print('Loaded markpoints data from {}'.format(markpoints_fp))
            else:
                self.markpoints_metadata = None

            # check if image metadata exists
            metadata_fp = self.image_filepath.split('.')[0] + '.xml'
            if os.path.exists(metadata_fp):
                self.image_metadata = bruker.getMetaData(metadata_fp)
                print('Loaded image metadata from {}'.format(metadata_fp))
            else:
                self.image_metadata = None

            self.refreshLassoWidget()

    def selectRoiFile(self):
        self.roi_filepath, _ = QFileDialog.getOpenFileName(self, "Open roi file (.hdf5)")
        self.roi_filename = os.path.split(self.roi_filepath)[1].split('.')[0]

        if self.roi_filename != '':
            self.roi_file_label.setText(self.roi_filename)
            self.updateExistingRoiSetList()


    def reloadImage(self):
        self.loadImage(self.image_filepath, self.channel)
        self.refreshLassoWidget()


    def loadImage(self, image_filepath, channel):
        nib_brain = np.asanyarray(nib.load(image_filepath).dataobj)
        brain_dims = nib_brain.shape # xyztc
        # TODO: check this for different input shapes
        image_series = nib_brain[:, :, :, :, channel] # xyzt
        self.image_series = image_series
        self.roi_image = np.mean(image_series, axis=3) # xyz

        self.zSlider.setValue(0)
        self.zSlider.setMaximum(self.roi_image.shape[2]-1)

        print('Loaded ch. {} image from {}: {} (xyzt)'.format(channel, image_filepath, image_series.shape))

    def selectChannel(self):
        self.channel = int(self.channel_combobox.currentText())



# %% # # # # # # # # ROI SELECTOR WIDGET # # # # # # # # # # # # # # # # # # #

    def selectedExistingRoiSet(self):
        pass

    def updateExistingRoiSetList(self):
        pass

    def refreshLassoWidget(self, keep_paths=False):
        self.roi_ax.clear()
        init_lasso = False
        if self.roi_image is not None:
            if len(self.roi_mask) > 0:
                newImage = plot_tools.overlayImage(self.roi_image[:, :, self.current_z_slice], self.roi_mask, 0.5, self.colors, z=self.current_z_slice)
            else:
                newImage = self.roi_image[:, :, self.current_z_slice]
            self.roi_ax.imshow(newImage, cmap=cm.gray)
            init_lasso = True

            if self.markpoints_metadata is not None:
                spiral_diam = float(self.markpoints_metadata.get('Point_1_SpiralWidth')) * self.roi_image.shape[0]
                spiral_x = float(self.markpoints_metadata.get('Point_1_X'))*self.roi_image.shape[0]
                spiral_y = float(self.markpoints_metadata.get('Point_1_Y'))*self.roi_image.shape[1]
                spiral_patch = plt.Circle((spiral_x, spiral_y), spiral_diam, color='r', alpha=0.5)
                self.roi_ax.add_patch(spiral_patch)
                self.roi_ax.plot(float(self.markpoints_metadata.get('Point_1_X'))*self.roi_image.shape[0],
                                 float(self.markpoints_metadata.get('Point_1_Y'))*self.roi_image.shape[1],
                                 'rx')
        else:
            self.roi_ax.imshow(self.blank_image)

        self.roi_canvas.draw()

        if not keep_paths:
            self.roi_path_list = []

        if init_lasso:
            if self.roi_type == 'circle':
                self.lasso_1 = EllipseSelector(self.roi_ax, onselect=self.newEllipse, button=1)
            elif self.roi_type == 'freehand':
                self.lasso_1 = LassoSelector(self.roi_ax, onselect=self.newFreehand, button=1)
                self.lasso_2 = LassoSelector(self.roi_ax, onselect=self.appendFreehand, button=3)
            else:
                print('Warning ROI type not recognized. Choose circle or freehand')

    def newFreehand(self, verts):
        new_roi_path = path.Path(verts)
        new_roi_path.z_level = self.zSlider.value()
        new_roi_path.channel = self.channel
        self.updateRoiSelection([new_roi_path])

    def appendFreehand(self, verts):
        print('Appending rois, hit Enter/Return to finish')
        new_roi_path = path.Path(verts)
        new_roi_path.z_level = self.zSlider.value()
        new_roi_path.channel = self.channel
        self.roi_path_list.append(new_roi_path)

    def keyPressEvent(self, event):
        if type(event) == QtGui.QKeyEvent:
            if np.any([event.key() == QtCore.Qt.Key_Return, event.key() == QtCore.Qt.Key_Enter]):
                if len(self.roi_path_list) > 0:
                    event.accept()
                    self.updateRoiSelection(self.roi_path_list)
                else:
                    event.ignore()
            else:
                event.ignore()
        else:
            event.ignore()

    def newEllipse(self, pos1, pos2):
        x1 = np.round(pos1.xdata)
        x2 = np.round(pos2.xdata)
        y1 = np.round(pos1.ydata)
        y2 = np.round(pos2.ydata)

        radiusX = np.sqrt((x1 - x2)**2)/2
        radiusY = np.sqrt((y1 - y2)**2)/2
        if self.roi_radius is not None:
            radiusX = self.roi_radius

        center = (np.round((x1 + x2)/2), np.round((y1 + y2)/2))
        new_roi_path = path.Path.circle(center=center, radius=radiusX)
        new_roi_path.z_level = self.zSlider.value()
        new_roi_path.channel = self.channel
        self.updateRoiSelection([new_roi_path])

    def updateRoiSelection(self, new_roi_path):
        mask = self.getRoiMaskFromPath(new_roi_path)
        new_roi_resp = self.getRoiDataFromPath(roi_path=new_roi_path)
        if new_roi_resp is None:
            print('No pixels in selected roi')
            return
        # update list of roi data
        self.roi_mask.append(mask)
        self.roi_path.append(new_roi_path)  # list of lists of paths
        self.roi_response.append(new_roi_resp)
        # update slider to show most recently drawn roi response
        self.current_roi_index = len(self.roi_response)-1
        self.roiSlider.setValue(self.current_roi_index)

        # Update figures
        self.redrawRoiTraces()


    def getRoiDataFromPath(self, roi_path):
        """
        Compute roi response from roi path objects.

        param:
            roi_path: list of path objects

        *Must first define self.current_series
        """
        mask = self.getRoiMaskFromPath(roi_path)

        roi_response = np.mean(self.image_series[mask, :], axis=0, keepdims=True) - np.min(self.image_series)

        return roi_response

    def getRoiMaskFromPath(self, roi_path):
        """
        Compute roi mask from roi path objects.

        param:
            roi_path: list of path objects

        *Must first define self.current_series
        """
        x_dim, y_dim, z_dim, t_dim = self.image_series.shape

        pixX = np.arange(y_dim)
        pixY = np.arange(x_dim)
        xv, yv = np.meshgrid(pixX, pixY)
        roi_pix = np.vstack((xv.flatten(), yv.flatten())).T

        mask = np.zeros(shape=(x_dim, y_dim, z_dim))

        for path in roi_path:
            z_level = path.z_level
            xy_indices = np.reshape(path.contains_points(roi_pix, radius=0.5), (x_dim, y_dim))
            mask[xy_indices, z_level] = 1

        mask = mask == 1  # convert to boolean for masking

        return mask

    def sliderUpdated(self):
        self.current_roi_index = self.roiSlider.value()
        self.redrawRoiTraces()

    def zSliderUpdated(self):
        self.current_z_slice = self.zSlider.value()
        if self.roi_image is not None:
            self.refreshLassoWidget(keep_paths=True)


    def redrawRoiTraces(self):
        self.responsePlot.clear()
        if self.current_roi_index < len(self.roi_response):
            current_raw_trace = np.squeeze(self.roi_response[self.current_roi_index])
            fxn_name = self.roi_response_type_combobox.currentText()
            display_trace = getattr(self, 'getRoiResponse_{}'.format(fxn_name))([current_raw_trace])
            frame_times = np.arange(0, display_trace.shape[0]) * float(self.image_metadata.get('framePeriod'))
            self.responsePlot.plot(frame_times, display_trace, linewidth=1, alpha=0.5)
            if len(display_trace.shape)>1:
                self.responsePlot.plot(frame_times, np.mean(display_trace, axis=1), color='k', linewidth=2, alpha=1.0)
            self.responsePlot.set_xlabel('Time (s)')
        self.responseCanvas.draw()

        self.refreshLassoWidget(keep_paths=False)

# %% Roi response analysis / display functions
    def getRoiResponse_RawTrace(self, roi_response):
        return roi_response[0]


    def getRoiResponse_ZapResponse(self, roi_response):
        if (self.markpoints_metadata is not None) & (self.image_metadata is not None):
            InitialDelay = float(self.markpoints_metadata.get('InitialDelay')) # msec
            Duration = float(self.markpoints_metadata.get('Duration')) # msec
            InterPointDelay = float(self.markpoints_metadata.get('InterPointDelay')) # msec
            Repetitions = int(self.markpoints_metadata.get('Repetitions'))
            # zap onsets in msec
            zap_onsets = InitialDelay + np.array([r*(Duration + InterPointDelay) for r in range(Repetitions)])

            # onset_frames
            frame_period = float(self.image_metadata.get('framePeriod')) * 1e3 # sec -> msec
            frame_times = np.array(self.image_metadata.get('frame_times')) * 1e3 # sec -> msec

            start_frames = [np.where(frame_times > onset)[0][0] for onset in zap_onsets]
            zap_duration = int(np.ceil(Duration / frame_period)) # imaging frames

            frames_to_pull = InterPointDelay / frame_period # imaging frames
            pre_frames = np.floor(InitialDelay / frame_period)

            traces = []
            for start_frame in start_frames:
                new_trace = roi_response[0][int(start_frame - pre_frames):int(start_frame + frames_to_pull)]
                new_trace[int(pre_frames):int(pre_frames + zap_duration)] = np.nan # blank out photoactivation artifact
                traces.append(new_trace)

            traces = np.vstack(traces).T


        return  traces


# %% # # # # # # # # LOADING / SAVING / COMPUTING ROIS # # # # # # # # # # # # # # # # # # #

    def loadRois(self, roi_set_path):
        pass #TODO


    def saveRois(self):
        roi_results = {}
        roi_results['roi_mask'] = self.roi_mask
        roi_results['roi_image'] = self.roi_image
        roi_results['roi_path']  = self.roi_path
        roi_results['roi_response'] = self.roi_response
        fxn_name = self.roi_response_type_combobox.currentText()
        roi_results[fxn_name] = []
        for r in range(len(self.roi_response)):
            roi_results[fxn_name].append(self.getRoiResponse_ZapResponse(self.roi_response[r]))

        roi_name = self.le_roiSetName.text()
        fn = self.image_filepath.split('.')[0] + '_roi_{}'.format(roi_name)
        with open('{}.pkl'.format(fn), 'wb') as handle:
            pickle.dump(roi_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print('Saved ROI data at {}'.format(fn))

    def deleteRoi(self):
        if self.current_roi_index < len(self.roi_response):
            self.roi_mask.pop(self.current_roi_index)
            self.roi_response.pop(self.current_roi_index)
            self.roi_path.pop(self.current_roi_index)
            self.roiSlider.setValue(self.current_roi_index-1)
            self.redrawRoiTraces()

    def clearRois(self):
        self.roi_mask = []
        self.roi_response = []
        self.roi_path = []
        self.roi_image = None
        self.responsePlot.clear()
        self.redrawRoiTraces()
        self.roi_ax.clear()

    def selectRoiType(self):
        self.roi_type = self.RoiTypeComboBox.currentText().split(':')[0]
        if 'circle' in self.RoiTypeComboBox.currentText():
            self.roi_radius = int(self.RoiTypeComboBox.currentText().split(':')[1])
        else:
            self.roi_radius = None
        self.redrawRoiTraces()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AnalysisGUI()
    sys.exit(app.exec_())
