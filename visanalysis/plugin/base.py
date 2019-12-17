import h5py
import numpy as np
from registration import CrossCorr
import functools
from visanalysis import plugin
from visanalysis.analysis import imaging_data
from matplotlib import path


"""
Parent acquisition plugin class

To define a new acquisition plugin, define the indicated methods
in the plugin subclass to overwrite these placeholders

"""


class BasePlugin():
    def __init__(self):
        super().__init__()
        self.volume_analysis = False
        self.ImagingDataObject = None

    ###########################################################################
    # Core methods - must these in child plugin definition
    ###########################################################################

    def getRoiImage(self, **kwargs):
        """
        kwargs
            'data_directory': string, dir. where acquisition data lives (usually user-indicated)
            'series_number': int, series in hdf5 data file
            'experiment_file_name': string, name of hdf5 data file
            'file_path': string, full path to hdf5 data file
            'pmt': int, which pmt/channel to load

        returns
            roiImage: 2D image used to draw rois
        """

    def getRoiDataFromPath(self, roi_path, data_directory, series_number, experiment_file_name, experiment_file_path):
        """
        args
            roi_path: matplotlib path object defining roi
            data_direcory: string, dir. where acquisition data lives (usually user-indicated)
            series_number: int, series in hdf5 data file
            experiment_file_name: string, name of hdf5 data file
            experiment_file_path: string, full path to hdf5 data file

        returns
            roi_response: 1D numpy array, value of roi intensity as a function of acquisition time point
        """

    def getRoiMaskFromPath(self, roi_path, data_directory, series_number, experiment_file_name, experiment_file_path):
        """
        args
            roi_path: matplotlib path object defining roi
            data_direcory: string, dir. where acquisition data lives (usually user-indicated)
            series_number: int, series in hdf5 data file
            experiment_file_name: string, name of hdf5 data file
            experiment_file_path: string, full path to hdf5 data file

        returns
            roi_mask: 2D array, boolean indices of where the roi mask was drawn
        """

    def attachData(self, experiment_file_name, file_path, data_directory):
        """
        args
            experiment_file_name: string, name of hdf5 data file
            data_direcory: string, dir. where acquisition data lives (usually user-indicated)
            file_path: string, full path to hdf5 data file

        accesses hdf5 data file and attaches data/metadata to each series
            -stimulus_timing group:
                datasets: frame_monitor (a.u.) and time_vector (sec.)
                attrs: sample_rate ()
            -acquisition group:
                datasets: time_points (sec.)
                attrs: acquisition metadata
                anything else to stash in there
        """

    def registerAndSaveStacks(self, experiment_file_name, file_path, data_directory):
        """
        args
            experiment_file_name: string, name of hdf5 data file
            data_direcory: string, dir. where acquisition data lives (usually user-indicated)
            file_path: string, full path to hdf5 data file

        looks in indicated data directory and registers/motion corrects images found in there
        optional to define
        """
        print('No registration function defined for this plugin')

    ###########################################################################
    # Shared methods - may overwrite these in child class
    ###########################################################################

    def getSeriesNumbers(self, file_path):
        all_series = []
        with h5py.File(file_path, 'r') as experiment_file:
            for fly_id in list(experiment_file['/Flies'].keys()):
                new_series = list(experiment_file['/Flies/{}/epoch_runs'.format(fly_id)].keys())
                all_series.append(new_series)
        all_series = [val for s in all_series for val in s]
        series = [int(x.split('_')[-1]) for x in all_series]
        return series

    def registerStack(self, image_series, time_points):
        """
        """

        reference_time_frame = 1  # sec, first frames to use as reference for registration
        reference_frame = np.where(time_points > reference_time_frame)[0][0]

        reference_image = np.squeeze(np.mean(image_series[0:reference_frame,:,:], axis = 0))
        register = CrossCorr()
        model = register.fit(image_series, reference=reference_image)

        registered_series = model.transform(image_series)
        if len(registered_series.shape) == 3:  # xyt
            registered_series = registered_series.toseries().toarray().transpose(2,0,1)  # shape t, y, x
        elif len(registered_series.shape) == 4:  # xyzt
            registered_series = registered_series.toseries().toarray().transpose(3,0,1,2)  # shape t, z, y, x

        return registered_series

    # ROI METHODS:

    def saveRoiSet(self, file_path, series_number,
                   roi_set_name,
                   roi_mask,
                   roi_response,
                   roi_image,
                   roi_path):

        with h5py.File(file_path, 'r+') as experiment_file:
            find_partial = functools.partial(find_series, sn=series_number)
            epoch_run_group = experiment_file.visititems(find_partial)
            parent_roi_group = epoch_run_group.require_group('rois')
            current_roi_group = parent_roi_group.require_group(roi_set_name)

            plugin.base.overwriteDataSet(current_roi_group, 'roi_mask', roi_mask)
            plugin.base.overwriteDataSet(current_roi_group, 'roi_response', roi_response)
            plugin.base.overwriteDataSet(current_roi_group, 'roi_image', roi_image)

            for dataset_key in current_roi_group.keys():
                if 'path_vertices' in dataset_key:
                    del current_roi_group[dataset_key]

            for roi_ind, roi_paths in enumerate(roi_path):  # for roi indices
                current_roi_index_group = current_roi_group.require_group('roipath_{}'.format(roi_ind))
                for p_ind, p in enumerate(roi_paths):  # for path objects within a roi index (for appended, noncontiguous rois)
                    current_roi_path_group = current_roi_index_group.require_group('subpath_{}'.format(p_ind))
                    plugin.base.overwriteDataSet(current_roi_path_group, 'path_vertices', p.vertices)

    def loadRoiSet(self, file_path, roi_set_path):
        def find_roi_path(name, obj):
            if 'roipath' in name:
                return obj

        def find_roi_subpath(name, obj):
            if 'subpath' in name:
                return obj

        roi_set_name = roi_set_path.split('/')[-1]
        with h5py.File(file_path, 'r') as experiment_file:
            if 'series' in roi_set_name:  # roi set from a different series
                series_no = roi_set_name.split(':')[0].split('series')[1]
                roi_name = roi_set_name.split(':')[1]
                #TODO: FIXME
                roi_set_group = experiment_file['/epoch_runs'].get(series_no).get('rois').get(roi_name)
                roi_response = [getRoiDataFromMask(x) for x in roi_mask]

            else:  # from this series
                roi_set_group = experiment_file[roi_set_path]
                roi_response = list(roi_set_group.get("roi_response")[:])
                roi_mask = list(roi_set_group.get("roi_mask")[:])
                roi_image = roi_set_group.get("roi_image")[:]

                roi_path = []
                for roipath_key, roipath_group in roi_set_group.items():
                    if isinstance(roipath_group, h5py._hl.group.Group):
                        subpaths = []
                        for roi_subpath_group in roipath_group.values():
                            if isinstance(roi_subpath_group, h5py._hl.group.Group):
                                subpaths.append(roi_subpath_group.get("path_vertices")[:])
                        subpaths = [path.Path(x) for x in subpaths]  # convert from verts to path object
                        roi_path.append(subpaths)  # list of list of paths

        return roi_response, roi_image, roi_path, roi_mask

    def updateImagingDataObject(self, experiment_file_directory, experiment_file_name, series_number):
        self.ImagingDataObject = imaging_data.ImagingDataObject(experiment_file_directory, experiment_file_name, series_number)

    # roi display computation functions

    def getRoiResponse_TrialAverage(self, roi_response):
        time_vector, response_matrix = self.ImagingDataObject.getEpochResponseMatrix(roi_response, dff=False)
        trial_avg = np.mean(response_matrix, axis=(0, 1))
        return trial_avg

    def getRoiResponse_TrialAverageDFF(self, roi_response):
        time_vector, response_matrix = self.ImagingDataObject.getEpochResponseMatrix(roi_response, dff=True)
        trial_avg = np.mean(response_matrix, axis=(0, 1))
        return trial_avg

    def getRoiResponse_RawTrace(self, roi_response):
        return roi_response[0]

    def getRoiResponse_TrialResponses(self, roi_response):
        time_vector, response_matrix = self.ImagingDataObject.getEpochResponseMatrix(roi_response, dff=False)
        TrialResponses = np.mean(response_matrix, axis=0).T
        return TrialResponses

##############################################################################
# Functions for data file manipulation / access
##############################################################################


def deleteGroup(file_path, group_path):
    group_name = group_path.split('/')[-1]
    with h5py.File(file_path, 'r+') as experiment_file:
        group_to_delete = experiment_file[group_path]
        parent = group_to_delete.parent
        del parent[group_name]


def getPathFromTreeItem(tree_item):
    path = tree_item.text(0)
    parent = tree_item.parent()
    while parent is not None:
        path = parent.text(0) + '/' + path
        parent = parent.parent()
    path = '/' + path
    return path


def changeAttribute(file_path, group_path, attr_key, attr_val):
    # see https://github.com/CCampJr/LazyHDF5
    #TODO: try to keep the type the same?
    with h5py.File(file_path, 'r+') as experiment_file:
        group = experiment_file[group_path]
        group.attrs[attr_key] = attr_val


def getAttributesFromGroup(file_path, group_path):
    # see https://github.com/CCampJr/LazyHDF5
    with h5py.File(file_path, 'r+') as experiment_file:
        group = experiment_file[group_path]
        attr_dict = {}
        for at in group.attrs:
            attr_dict[at] = group.attrs[at]
        return attr_dict


def getHierarchy(file_path, additional_exclusions=None):
    with h5py.File(file_path, 'r') as experiment_file:
        hierarchy = recursively_load_dict_contents_from_group(experiment_file, '/', additional_exclusions=additional_exclusions)
    return hierarchy


def recursively_load_dict_contents_from_group(h5file, path, additional_exclusions=None):
    # https://codereview.stackexchange.com/questions/120802/recursively-save-python-dictionaries-to-hdf5-files-using-h5py
    exclusions = ['acquisition', 'Client', 'epochs', 'stimulus_timing', 'roipath', 'subpath']
    if additional_exclusions is not None:
        exclusions.append(additional_exclusions)
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            pass
        elif isinstance(item, h5py._hl.group.Group):
            if np.any([x in key for x in exclusions]):
                pass
            else:
                ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/', additional_exclusions=additional_exclusions)
    return ans


def overwriteDataSet(group, name, data):
    if group.get(name):
        del group[name]
    group.create_dataset(name, data=data)


def getDataType(file_path):
    with h5py.File(file_path, 'r+') as experiment_file:
        return experiment_file.attrs['rig']


def find_series(name, obj, sn):
    target_group_name = 'series_{}'.format(str(sn).zfill(3))
    if target_group_name in name:
        return obj


def getAvailableRoiSetNames(file_path, series_number):
    with h5py.File(file_path, 'r+') as experiment_file:
        find_partial = functools.partial(find_series, sn=series_number)
        epoch_run_group = experiment_file.visititems(find_partial)
        rois_group = epoch_run_group.get('rois')
        return list(rois_group.keys())
