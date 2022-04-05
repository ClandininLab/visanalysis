"""
Functions for hdf5 data file manipulation / access

https://github.com/ClandininLab/visanalysis
mhturner@stanford.edu
"""
import h5py
import numpy as np
import functools


def updateSeriesAttribute(file_path, series_number,
                          attr_key, attr_val):
    """User facing, compared to  changeAttribute"""
    with h5py.File(file_path, 'r+') as experiment_file:
        find_partial = functools.partial(find_series, sn=series_number)
        epoch_run_group = experiment_file.visititems(find_partial)
        epoch_run_group.attrs[attr_key] = attr_val

        
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
    # TODO: try to keep the type the same?
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


def attachImageFileName(file_path, series_number, image_file_name):
    with h5py.File(file_path, 'r+') as experiment_file:
        find_partial = functools.partial(find_series, sn=series_number)
        epoch_run_group = experiment_file.visititems(find_partial)
        acquisition_group = epoch_run_group.require_group('acquisition')
        acquisition_group.attrs['image_file_name'] = image_file_name


def readImageFileName(file_path, series_number):
    with h5py.File(file_path, 'r') as experiment_file:
        find_partial = functools.partial(find_series, sn=series_number)
        epoch_run_group = experiment_file.visititems(find_partial)
        acquisition_group = epoch_run_group.require_group('acquisition')
        image_file_name = acquisition_group.attrs.get('image_file_name')

    return image_file_name
