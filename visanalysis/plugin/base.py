import h5py
import scipy.signal as signal
import numpy as np
import pylab


class BasePlugin():
    def __init__(self):
        super().__init__()

    def registerAndSaveStacks(self, experiment_file_name, file_path, data_directory):
        print('No registration function defined for this plugin')

    def getSeriesNumbers(self, file_path):
        all_series = []
        with h5py.File(file_path, 'r') as experiment_file:
            for fly_id in list(experiment_file['/Flies'].keys()):
                new_series = list(experiment_file['/Flies/{}/epoch_runs'.format(fly_id)].keys())
                all_series.append(new_series)
        all_series = [val for s in all_series for val in s]
        series = [int(x.split('_')[-1]) for x in all_series]
        return series

    def deleteGroup(self, file_path, group_path):
        group_name = group_path.split('/')[-1]
        with h5py.File(file_path, 'r+') as experiment_file:
            group_to_delete = experiment_file[group_path]
            parent = group_to_delete.parent
            del parent[group_name]

    def changeAttribute(self, file_path, group_path, attr_key, attr_val):
        # see https://github.com/CCampJr/LazyHDF5
        #TODO: try to keep the type the same?
        with h5py.File(file_path, 'r+') as experiment_file:
            group = experiment_file[group_path]
            group.attrs[attr_key] = attr_val

    def getAttributesFromGroup(self, file_path, group_path):
        # see https://github.com/CCampJr/LazyHDF5
        with h5py.File(file_path, 'r+') as experiment_file:
            group = experiment_file[group_path]
            attr_dict = {}
            for at in group.attrs:
                attr_dict[at] = group.attrs[at]
            return attr_dict

    def getPathFromTreeItem(self, tree_item):
        path = tree_item.text(0)
        parent = tree_item.parent()
        while parent is not None:
            path = parent.text(0) + '/' + path
            parent = parent.parent()
        path = '/' + path
        return path

    def getHierarchy(self, file_path):
        with h5py.File(file_path, 'r') as experiment_file:
            hierarchy = recursively_load_dict_contents_from_group(experiment_file, '/')
        return hierarchy


def recursively_load_dict_contents_from_group(h5file, path):
    # https://codereview.stackexchange.com/questions/120802/recursively-save-python-dictionaries-to-hdf5-files-using-h5py
    exclusions = ['acquisition', 'Client', 'epochs', 'stimulus_timing', 'roipath', 'subpath']
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            pass
        elif isinstance(item, h5py._hl.group.Group):
            if np.any([x in key for x in exclusions]):
                pass
            else:
                ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
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
