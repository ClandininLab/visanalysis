"""
Functions for hdf5 data file manipulation / access

https://github.com/ClandininLab/visanalysis
mhturner@stanford.edu
"""
import h5py
import numpy as np
import functools


# Stimpack renamed the group that holds a subject's series from 'epoch_runs' to 'series'.
# Both layouts are in the wild, so every accessor resolves the name instead of hard-coding it.
# ('find_series' below is a substring match on the series group name itself, so it already
#  works under either parent, which is why the write path needed no change.)
SERIES_PARENT_GROUP_NAMES = ('epoch_runs', 'series')

# Likewise, older files call the subject collection 'Flies' rather than 'Subjects'.
SUBJECT_PARENT_GROUP_NAMES = ('Subjects', 'Flies')


def getSubjectParentName(experiment_file):
    """Name of the top-level group holding subjects ('Subjects', or legacy 'Flies')."""
    for name in SUBJECT_PARENT_GROUP_NAMES:
        if name in experiment_file:
            return name
    return None


def getSeriesParentName(subject_group):
    """Name of the group holding one subject's series ('epoch_runs', or newer 'series')."""
    for name in SERIES_PARENT_GROUP_NAMES:
        if name in subject_group:
            return name
    return None


def getSeriesParentGroup(experiment_file, subject_id):
    """Group holding one subject's series, under either naming scheme. None if absent."""
    subject_parent = getSubjectParentName(experiment_file)
    if subject_parent is None:
        return None
    subject_group = experiment_file[subject_parent].get(str(subject_id))
    if subject_group is None:
        return None
    series_parent = getSeriesParentName(subject_group)
    return None if series_parent is None else subject_group[series_parent]


def getSubjectIds(experiment_file):
    """List of subject ids, under either naming scheme."""
    subject_parent = getSubjectParentName(experiment_file)
    if subject_parent is None:
        return []
    return list(experiment_file[subject_parent].keys())


def updateSeriesAttribute(file_path, series_number,
                          attr_key, attr_val):
    """User facing, compared to  changeAttribute"""
    with h5py.File(file_path, 'r+') as experiment_file:
        find_partial = functools.partial(find_series, sn=series_number)
        epoch_run_group = experiment_file.visititems(find_partial)
        epoch_run_group.attrs[attr_key] = attr_val


def deleteSeriesAttribute(file_path, series_number, attr_key):
    with h5py.File(file_path, 'r+') as experiment_file:
        find_partial = functools.partial(find_series, sn=series_number)
        epoch_run_group = experiment_file.visititems(find_partial)
        del epoch_run_group.attrs[attr_key]


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


def readDataSet(file_path, series_number,
                group_name,
                dataset_name):
    with h5py.File(file_path, 'r+') as experiment_file:
        find_partial = functools.partial(find_series, sn=series_number)
        epoch_run_group = experiment_file.visititems(find_partial)
        data_matrix = epoch_run_group[group_name].get(dataset_name)[:]
        sample_rate = epoch_run_group[group_name].attrs['sample_rate']

        return data_matrix, sample_rate


def getDataType(file_path):
    with h5py.File(file_path, 'r+') as experiment_file:
        if 'rig' in experiment_file.attrs:
            return experiment_file.attrs['rig']
        elif 'rig_config' in experiment_file.attrs:
            return experiment_file.attrs['rig_config']
        else:
            return None


def find_series(name, obj, sn):
    target_group_name = 'series_{}'.format(str(sn).zfill(3))
    if target_group_name in name:
        return obj


def getGroupsUnderSeries(file_path, series_number):
    with h5py.File(file_path, 'r+') as experiment_file:
        find_partial = functools.partial(find_series, sn=series_number)
        epoch_run_group = experiment_file.visititems(find_partial)
        return epoch_run_group.keys()


def seriesExists(file_path, series_number):
    with h5py.File(file_path, 'r') as experiment_file:
        find_partial = functools.partial(find_series, sn=series_number)
        epoch_run_group = experiment_file.visititems(find_partial)
        if epoch_run_group is None:
            return False
        else:
            return True


def createEpochRunGroup(file_path, fly_id, series_number):
    if seriesExists(file_path, series_number):
        print('Series {} already exists in {} - ABORTING'.format(series_number, file_path))
    else:
        with h5py.File(file_path, 'r+') as experiment_file:
            fly_group = getSeriesParentGroup(experiment_file, fly_id)
            if fly_group is None:
                raise KeyError('No series group found for subject {} in {}'.format(fly_id, file_path))
            fly_group.create_group('series_{}'.format(str(series_number).zfill(3)))
            print('Added series {} to fly {} in {}'.format(series_number, fly_id, file_path))


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
