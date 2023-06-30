"""
Parent acquisition plugin class.

To define a new acquisition plugin, define the indicated methods
in the plugin subclass to overwrite these placeholders

https://github.com/ClandininLab/visanalysis
mhturner@stanford.edu
"""

import h5py
import numpy as np
import functools
from visanalysis.analysis import imaging_data, shared_analysis
from visanalysis.util import h5io
from matplotlib import path
import os



class BasePlugin():
    def __init__(self):
        super().__init__()
        self.ImagingDataObject = None

    ###########################################################################
    # Core methods - must overwrite these in child plugin definition
    ###########################################################################

    def getRoiImage(self, data_directory, image_file_name, series_number, channel, z_slice):
        """
        Get 2D roi image for display, e.g. for GUI.

        args
            data_directory: string, dir. where acquisition data lives (usually user-indicated)
            image_file_name: string, filename of image file to use
            series_number: int, series in hdf5 data file
            channel: int, which pmt/channel to load
            z_slice: int, which z slice to load from z stack data

        returns
            roiImage: 2D image used to draw rois
        """

    def getRoiDataFromPath(self, roi_path):
        """
        Compute roi response from roi path objects.

        args
            roi_path: matplotlib path object defining roi
        returns
            roi_response: 1D numpy array, value of roi intensity as a function of acquisition time point
        """

    def getRoiMaskFromPath(self, roi_path):
        """
        Compute roi mask from roi path objects.

        args
            roi_path: matplotlib path object defining roi

        returns
            roi_mask: 2D array, boolean indices of where the roi mask was drawn
        """

    def attachData(self, experiment_file_name, file_path, data_directory):
        """
        Attach imaging metadata to visanalysis hdf5 file.

        args
            experiment_file_name: string, name of hdf5 data file
            file_path: string, full path to hdf5 data file
            data_direcory: string, dir. where acquisition data lives (usually user-indicated)


        accesses hdf5 data file and attaches data/metadata to each series
            -stimulus_timing group:
                datasets: frame_monitor (a.u.) and time_vector (sec.)
                attrs: sample_rate ()
            -acquisition group:
                datasets: time_points (sec.)
                attrs: acquisition metadata
                anything else to stash in there
        """

    ###########################################################################
    # Shared methods - may overwrite these in child class
    ###########################################################################

    def getSeriesNumbers(self, file_path):
        """
        Retrieve all epoch series numbers from hdf5 file

        args
            file_path: string, full path to hdf5 data file
        """
        all_series = []
        with h5py.File(file_path, 'r') as experiment_file:
            for fly_id in list(experiment_file['/Subjects'].keys()):
                new_series = list(experiment_file['/Subjects/{}/epoch_runs'.format(fly_id)].keys())
                all_series.append(new_series)
        all_series = [val for s in all_series for val in s]
        series = [int(x.split('_')[-1]) for x in all_series]
        return series

    def getRoiSetPaths(self, file_path):
        all_roiset_paths = {}
        with h5py.File(file_path, 'r') as experiment_file:
            for fly_id in list(experiment_file['/Subjects'].keys()):
                for sn in list(experiment_file['/Subjects/{}/epoch_runs'.format(fly_id)].keys()):
                    for roi_name in experiment_file['/Subjects/{}/epoch_runs/{}/rois'.format(fly_id, sn)].keys():
                        new_path = '/Subjects/{}/epoch_runs/{}/rois/{}'.format(fly_id, sn, roi_name)
                        new_key = '{}:{}:{}'.format(fly_id, sn, roi_name)
                        all_roiset_paths[new_key] = new_path
        return all_roiset_paths

    # ROI METHODS:
    def saveRoiSet(self, file_path, series_number,
                   roi_set_name,
                   roi_mask,
                   roi_response,
                   roi_image,
                   roi_path):

        with h5py.File(file_path, 'r+') as experiment_file:
            find_partial = functools.partial(h5io.find_series, sn=series_number)
            epoch_run_group = experiment_file.visititems(find_partial)
            parent_roi_group = epoch_run_group.require_group('rois')
            current_roi_group = parent_roi_group.require_group(roi_set_name)

            h5io.overwriteDataSet(current_roi_group, 'roi_mask', roi_mask)
            h5io.overwriteDataSet(current_roi_group, 'roi_response', roi_response)
            h5io.overwriteDataSet(current_roi_group, 'roi_image', roi_image)

            for dataset_key in current_roi_group.keys():
                if 'path_vertices' in dataset_key:
                    del current_roi_group[dataset_key]

            for roi_ind, roi_paths in enumerate(roi_path):  # for roi indices
                current_roi_index_group = current_roi_group.require_group('roipath_{}'.format(roi_ind))
                for p_ind, p in enumerate(roi_paths):  # for path objects within a roi index (for appended, noncontiguous rois)
                    current_roi_path_group = current_roi_index_group.require_group('subpath_{}'.format(p_ind))
                    h5io.overwriteDataSet(current_roi_path_group, 'path_vertices', p.vertices)
                    current_roi_path_group.attrs['z_level'] = p.z_level
                    current_roi_path_group.attrs['channel'] = p.channel

    def loadRoiSet(self, file_path, roi_set_path):
        def find_roi_path(name, obj):
            if 'roipath' in name:
                return obj

        def find_roi_subpath(name, obj):
            if 'subpath' in name:
                return obj

        with h5py.File(file_path, 'r') as experiment_file:
            roi_set_group = experiment_file[roi_set_path]
            roi_response = list(roi_set_group.get("roi_response")[:])
            roi_mask = list(roi_set_group.get("roi_mask")[:])
            roi_image = roi_set_group.get("roi_image")[:]
            if len(roi_image.shape) <= 2:
                roi_image = roi_image[:, :, np.newaxis]

            roi_path = []
            for roipath_key, roipath_group in roi_set_group.items():
                if isinstance(roipath_group, h5py._hl.group.Group):
                    subpaths = []
                    for roi_subpath_group in roipath_group.values():
                        if isinstance(roi_subpath_group, h5py._hl.group.Group):
                            new_subpath = path.Path(roi_subpath_group.get("path_vertices")[:])
                            new_subpath.z_level = int(roi_subpath_group.attrs.get('z_level', 0))
                            new_subpath.channel = int(roi_subpath_group.attrs.get('channel', 0))
                            subpaths.append(new_subpath)
                    roi_path.append(subpaths)  # list of list of paths

        return roi_response, roi_image, roi_path, roi_mask

    def updateImagingDataObject(self, experiment_file_directory, experiment_file_name, series_number):
        file_path = os.path.join(experiment_file_directory, experiment_file_name + '.hdf5')
        self.ImagingDataObject = imaging_data.ImagingDataObject(file_path, series_number, quiet=True)

    # roi display computation functions
    def getRoiResponse_TrialAverage(self, roi_response):
        time_vector, response_matrix = self.ImagingDataObject.getEpochResponseMatrix(np.vstack(roi_response), dff=False)
        trial_avg = np.mean(response_matrix, axis=(0, 1))
        return trial_avg

    def getRoiResponse_TrialAverageBright(self, roi_response):
        time_vector, response_matrix = self.ImagingDataObject.getEpochResponseMatrix(np.vstack(roi_response), dff=False)
        query = {'current_intensity': 1}
        response_matrix_filt = shared_analysis.filterTrials(response_matrix, self.ImagingDataObject, query, return_inds=False)
        trial_avg = np.mean(response_matrix_filt, axis=(0, 1))
        return trial_avg

    def getRoiResponse_TrialAverageDFF(self, roi_response):
        time_vector, response_matrix = self.ImagingDataObject.getEpochResponseMatrix(np.vstack(roi_response), dff=True)
        trial_avg = np.mean(response_matrix, axis=(0, 1))
        return trial_avg

    def getRoiResponse_RawTrace(self, roi_response):
        return roi_response[0]

    def getRoiResponse_TrialResponses(self, roi_response):
        time_vector, response_matrix = self.ImagingDataObject.getEpochResponseMatrix(np.vstack(roi_response), dff=False)
        TrialResponses = np.mean(response_matrix, axis=0).T
        return TrialResponses

    def dataIsAttached(self, file_path, series_number):
        with h5py.File(file_path, 'r+') as experiment_file:
            find_partial = functools.partial(h5io.find_series, sn=series_number)
            epoch_run_group = experiment_file.visititems(find_partial)
            acquisition_group = epoch_run_group.get('acquisition')
            if len(acquisition_group.keys()) > 0:
                return True
            else:
                return False
