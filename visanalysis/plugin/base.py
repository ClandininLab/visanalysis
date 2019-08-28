import h5py
import scipy.signal as signal
import numpy as np
import pylab
import functools


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

    def getEpochAndFrameTiming(self, time_vector, frame_monitor, sample_rate,
                               plot_trace_flag = True,
                               threshold = 0.6,
                               minimum_epoch_separation = 2e3, # datapoints
                               frame_slop = 10, #datapoints +/- ideal frame duration
                               command_frame_rate = 120):
        """
        getEpochAndFrameTiming(self, time_vector, frame_monitor, sample_rate)
            returns stimulus timing information based on photodiode voltage trace from alternating frame tracker signal

        """
        # Low-pass filter frame_monitor trace
        b, a = signal.butter(4, 10*command_frame_rate, btype = 'low', fs = sample_rate)
        frame_monitor = signal.filtfilt(b, a, frame_monitor)

        # shift & normalize so frame monitor trace lives on [0 1]
        frame_monitor = frame_monitor - np.min(frame_monitor)
        frame_monitor = frame_monitor / np.max(frame_monitor)

        # find lightcrafter frame flip times
        V_orig = frame_monitor[0:-2]
        V_shift = frame_monitor[1:-1]
        ups = np.where(np.logical_and(V_orig < threshold, V_shift >= threshold))[0] + 1
        downs = np.where(np.logical_and(V_orig >= threshold, V_shift < threshold))[0] + 1
        frame_times = np.sort(np.append(ups, downs))

        # Use frame flip times to find stimulus start times
        stimulus_start_frames = np.append(0, np.where(np.diff(frame_times) > minimum_epoch_separation)[0] + 1)
        stimulus_end_frames = np.append(np.where(np.diff(frame_times) > minimum_epoch_separation)[0],len(frame_times)-1)
        stimulus_start_times = frame_times[stimulus_start_frames] / sample_rate  # datapoints -> sec
        stimulus_end_times = frame_times[stimulus_end_frames] / sample_rate  # datapoints -> sec

        # Find dropped frames and calculate frame rate
        interval_duration = np.diff(frame_times)
        frame_len = interval_duration[np.where(interval_duration < minimum_epoch_separation)]
        ideal_frame_len = 1 / command_frame_rate  * sample_rate #datapoints
        dropped_frame_inds = np.where(np.abs(frame_len - ideal_frame_len)>frame_slop)[0]
        if len(dropped_frame_inds)>0:
            print('Warning! Dropped ' + str(len(dropped_frame_inds)) + ' frame(s)')
        good_frame_inds = np.where(np.abs(frame_len - ideal_frame_len)<frame_slop)[0]
        measured_frame_len = np.mean(frame_len[good_frame_inds]) #datapoints
        frame_rate = 1 / (measured_frame_len / sample_rate) #Hz

        if plot_trace_flag:
            pylab.plot(time_vector,frame_monitor)
            pylab.plot(time_vector[frame_times],threshold * np.ones(frame_times.shape),'ko')
            pylab.plot(stimulus_start_times, threshold * np.ones(stimulus_start_times.shape),'go')
            pylab.plot(stimulus_end_times, threshold * np.ones(stimulus_end_times.shape),'ro')
            pylab.plot(frame_times[dropped_frame_inds] / sample_rate, 1 * np.ones(dropped_frame_inds.shape),'ro')
            pylab.show

        return {'frame_times':frame_times, 'stimulus_end_times':stimulus_end_times,
                'stimulus_start_times':stimulus_start_times, 'dropped_frame_inds':dropped_frame_inds,
                'frame_rate':frame_rate}

    def getHierarchy(self, file_path):
        with h5py.File(file_path, 'r') as experiment_file:
            hierarchy = recursively_load_dict_contents_from_group(experiment_file, '/')
        return hierarchy


def recursively_load_dict_contents_from_group(h5file, path):
    # https://codereview.stackexchange.com/questions/120802/recursively-save-python-dictionaries-to-hdf5-files-using-h5py
    exclusions = ['acquisition', 'Client', 'epochs', 'stimulus_timing']
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            pass
        elif isinstance(item, h5py._hl.group.Group):
            if np.any([x == key for x in exclusions]):
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
