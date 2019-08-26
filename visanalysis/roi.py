import numpy as np
import h5py
import functools
from visanalysis import plugin
from matplotlib import path


def saveRoiSet(file_path, series_number,
               roi_set_name,
               roi_mask,
               roi_response,
               roi_image,
               roi_path):

    def find_series(name, obj, sn):
        target_group_name = 'series_{}'.format(str(sn).zfill(3))
        if target_group_name in name:
            return obj

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
        for p_ind, p in enumerate(roi_path):
            current_roi_group.create_dataset("path_vertices_" + str(p_ind), data = p.vertices)


def loadRoiSet(file_path, roi_set_path):
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
            new_path = roi_set_group.get("path_vertices_0")
            ind = 0
            while new_path is not None:
                roi_path.append(new_path)
                ind += 1
                new_path = roi_set_group.get("path_vertices_" + str(ind))
            roi_path = [x[:] for x in roi_path]  # path vertices
            roi_path = [path.Path(x) for x in roi_path]  # convert from verts to path object

    return roi_response, roi_image, roi_path, roi_mask


def removeRoiSet(file_path, series_number, roi_set_name):
    def find_series(name, obj, sn):
        target_group_name = 'series_{}'.format(str(sn).zfill(3))
        if target_group_name in name:
            return obj

    with h5py.File(file_path, 'r+') as experiment_file:
        find_partial = functools.partial(find_series, sn=series_number)
        epoch_run_group = experiment_file.visititems(find_partial)
        rois_group = epoch_run_group.get('rois')
        del rois_group[roi_set_name]
        print('Roi group {} from series {} deleted'.format(roi_set_name, series_number))


def getRoiMask(image, indices):
    array = np.zeros((image.shape[0], image.shape[1]))
    lin = np.arange(array.size)
    newRoiArray = array.flatten()
    newRoiArray[lin[indices]] = 1
    newRoiArray = newRoiArray.reshape(array.shape)
    mask = newRoiArray == 1 #convert to boolean for masking
    return mask


def getRoiDataFromMask(current_series, mask):
    roi_response = (np.mean(current_series[:, mask], axis=1, keepdims=True) - np.min(current_series)).T
    return roi_response
