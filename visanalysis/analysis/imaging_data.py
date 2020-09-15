"""
ImagingDataObject for visanalysis data files
Associated with a data file and series number


@author: mhturner
"""
import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.signal as signal
import functools

from visanalysis import plot_tools


class ImagingDataObject():
    def __init__(self, experiment_file_directory, experiment_file_name, series_number, kwargs=None):
        kwargs_passed = {'plot_trace_flag': False,
                         'minimum_epoch_separation': 2e3}

        if kwargs is not None:
            for key in kwargs:
                kwargs_passed[key] = kwargs[key]

        self.experiment_file_directory = experiment_file_directory
        self.experiment_file_name = experiment_file_name
        self.series_number = series_number
        self.file_path = os.path.join(self.experiment_file_directory, self.experiment_file_name + '.hdf5')

        # Retrieve from file: stimulus_parameters
        self.getRunParameters()
        self.getEpochParameters()
        # Retrieve:
        self.getFlyMetadata()
        self.getAcquisitionMetadata()
        # Retrieve: photodiode_trace, photodiode_time_vector, photodiode_sample_rate
        self.getPhotodiodeData()
        # Retrieve: response_timing
        self.getResponseTiming()
        # Calculate stimulus_timing
        self.computeEpochAndFrameTiming(plot_trace_flag=kwargs_passed['plot_trace_flag'], minimum_epoch_separation=kwargs_passed['minimum_epoch_separation'])

        self.colors = sns.color_palette("Set2", n_colors=20)

    def getRunParameters(self):
        with h5py.File(self.file_path, 'r') as experiment_file:
            find_partial = functools.partial(find_series, sn=self.series_number)
            epoch_run_group = experiment_file.visititems(find_partial)
            self.run_parameters = {}
            for attr_key in epoch_run_group.attrs:
                self.run_parameters[attr_key] = epoch_run_group.attrs[attr_key]

    def getEpochParameters(self):
        with h5py.File(self.file_path, 'r') as experiment_file:
            find_partial = functools.partial(find_series, sn=self.series_number)
            epoch_run_group = experiment_file.visititems(find_partial)
            self.epoch_parameters = []
            for epoch in epoch_run_group['epochs'].values():
                new_params = {}
                for attr_key in epoch.attrs:
                    new_params[attr_key] = epoch.attrs[attr_key]
                self.epoch_parameters.append(new_params)

    def getFlyMetadata(self):
        with h5py.File(self.file_path, 'r') as experiment_file:
            find_partial = functools.partial(find_series, sn=self.series_number)
            epoch_run_group = experiment_file.visititems(find_partial)
            fly_group = epoch_run_group.parent.parent
            self.fly_metadata = {}
            for attr_key in fly_group.attrs:
                self.fly_metadata[attr_key] = fly_group.attrs[attr_key]

    def getAcquisitionMetadata(self):
        with h5py.File(self.file_path, 'r') as experiment_file:
            find_partial = functools.partial(find_series, sn=self.series_number)
            epoch_run_group = experiment_file.visititems(find_partial)
            acquisition_group = epoch_run_group['acquisition']
            self.acquisition_metadata = {}
            for attr_key in acquisition_group.attrs:
                self.acquisition_metadata[attr_key] = acquisition_group.attrs[attr_key]

    def getPhotodiodeData(self):
        with h5py.File(self.file_path, 'r') as experiment_file:
            find_partial = functools.partial(find_series, sn=self.series_number)
            epoch_run_group = experiment_file.visititems(find_partial)
            stimulus_timing_group = epoch_run_group['stimulus_timing']

            self.photodiode_trace = stimulus_timing_group.get('frame_monitor')[:]
            if len(self.photodiode_trace.shape) < 2:
                self.photodiode_trace = self.photodiode_trace[np.newaxis, :] # dummy dim for single channel photodiode
            self.photodiode_time_vector = stimulus_timing_group.get('time_vector')[:]
            self.photodiode_sample_rate = stimulus_timing_group.attrs['sample_rate']

    def getResponseTiming(self):
        with h5py.File(self.file_path, 'r') as experiment_file:
            find_partial = functools.partial(find_series, sn=self.series_number)
            epoch_run_group = experiment_file.visititems(find_partial)
            acquisition_group = epoch_run_group['acquisition']

            self.response_timing = {}
            self.response_timing['time_vector'] = acquisition_group.get('time_points')[:]  # sec
            self.response_timing['sample_period'] = acquisition_group.attrs['sample_period']  # sec

    def computeEpochAndFrameTiming(self,
                                   plot_trace_flag=True,
                                   threshold=0.6,
                                   minimum_epoch_separation=2e3,  # datapoints
                                   frame_slop=20,  # datapoints +/- ideal frame duration
                                   command_frame_rate=120):
        """
        getEpochAndFrameTiming(self, time_vector, frame_monitor, sample_rate)
            returns stimulus timing information based on photodiode voltage trace from alternating frame tracker signal

        """
        frame_monitor_channels = self.photodiode_trace.copy()
        time_vector = self.photodiode_time_vector.copy()
        sample_rate = self.photodiode_sample_rate.copy()

        num_channels = frame_monitor_channels.shape[0]
        for ch in range(num_channels):
            frame_monitor = frame_monitor_channels[ch, :]

            # Low-pass filter frame_monitor trace
            b, a = signal.butter(4, 10*command_frame_rate, btype='low', fs=sample_rate)
            frame_monitor = signal.filtfilt(b, a, frame_monitor)

            # shift & normalize so frame monitor trace lives on [0 1]
            frame_monitor = frame_monitor - np.min(frame_monitor)
            frame_monitor = frame_monitor / np.max(frame_monitor)

            # find frame flip times
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
            ideal_frame_len = 1 / command_frame_rate * sample_rate  # datapoints
            dropped_frame_inds = np.where(np.abs(frame_len - ideal_frame_len)>frame_slop)[0]
            if len(dropped_frame_inds) > 0:
                print('Warning! Ch. {} Dropped {} frames'.format(ch, len(dropped_frame_inds)))
            good_frame_inds = np.where(np.abs(frame_len - ideal_frame_len) < frame_slop)[0]
            measured_frame_len = np.mean(frame_len[good_frame_inds])  # datapoints
            frame_rate = 1 / (measured_frame_len / sample_rate)  # Hz

            if plot_trace_flag:
                self.frame_monitor_figure = plt.figure(figsize=(16, 6))
                ax = self.frame_monitor_figure.add_subplot(111)
                ax.plot(time_vector, frame_monitor)
                ax.plot(time_vector[frame_times], threshold * np.ones(frame_times.shape),'ko')
                ax.plot(stimulus_start_times, threshold * np.ones(stimulus_start_times.shape),'go')
                ax.plot(stimulus_end_times, threshold * np.ones(stimulus_end_times.shape),'ro')
                ax.plot(frame_times[dropped_frame_inds] / sample_rate, 1 * np.ones(dropped_frame_inds.shape),'ro')
                ax.set_title('Ch. {}: Frame rate = {} Hz'.format(ch, frame_rate))
                plt.show()

        # for stimulus_timing just use one of the channels, both *should* be in sync
        self.stimulus_timing = {'frame_times': frame_times,
                                'stimulus_end_times': stimulus_end_times,
                                'stimulus_start_times': stimulus_start_times,
                                'dropped_frame_inds': dropped_frame_inds,
                                'frame_rate': frame_rate}

    def getRoiResponses(self, background_subtraction=False):
        self.roi = {}
        with h5py.File(self.file_path, 'r') as experiment_file:
            find_partial = functools.partial(find_series, sn=self.series_number)
            roi_parent_group = experiment_file.visititems(find_partial)['rois']
            for roi_set_name in roi_parent_group.keys():
                roi_set_group = roi_parent_group[roi_set_name]
                new_roi = {}
                new_roi['roi_response'] = list(roi_set_group.get("roi_response")[:])
                new_roi['roi_mask'] = list(roi_set_group.get("roi_mask")[:])
                new_roi['roi_image'] = roi_set_group.get("roi_image")[:]
                self.roi[roi_set_name] = new_roi

        if background_subtraction: # TODO: make this a function for different bgs computations, maybe subclass out
            # do background subtraction for each roi
            response_rois = [x for x in self.roi.keys() if 'bg' not in x]
            bg_all = self.roi.get('bg', None)
            for roi_set_name in response_rois:
                print(roi_set_name)
                if bg_all:
                    background = np.mean(self.roi.get('bg')['roi_response'], axis=0)
                else:
                    background = np.mean(self.roi.get(roi_set_name + '_bg')['roi_response'], axis=0)

                self.roi.get(roi_set_name)['roi_response'] = self.roi.get(roi_set_name)['roi_response'] - background
        else:
            response_rois = self.roi.keys()

        # get epoch response matrix for each roi
        for roi_set_name in response_rois:
            time_vector, response_matrix = self.getEpochResponseMatrix(self.roi.get(roi_set_name)['roi_response'])

            self.roi.get(roi_set_name)['epoch_response'] = response_matrix
            self.roi.get(roi_set_name)['time_vector'] = time_vector

    def getEpochResponseMatrix(self, roi_response, dff=True):
        """
        getEpochReponseMatrix(self)
            Takes in long stack response traces and splits them up into each stimulus epoch
            roi_response shape = list of roi responses

        Returns:
            time_vector (ndarray): in seconds. Time points of each frame acquisition within each epoch
            response_matrix (ndarray): response for each roi in each epoch.
                shape = (num rois, num epochs, num frames per epoch)
        """
        self.response = {}
        response_trace = np.vstack(roi_response)

        stimulus_start_times = self.stimulus_timing['stimulus_start_times']  # sec
        stimulus_end_times = self.stimulus_timing['stimulus_end_times']  # sec

        pre_time = self.run_parameters['pre_time']  # sec
        tail_time = self.run_parameters['tail_time']  # sec
        epoch_start_times = stimulus_start_times - pre_time
        epoch_end_times = stimulus_end_times + tail_time

        sample_period = self.response_timing['sample_period']  # sec
        stack_times = self.response_timing['time_vector']  # sec

        # Use measured stimulus lengths for stim time instead of epoch param
        # cut off a bit of the end of each epoch to allow for slop in how many frames were acquired
        epoch_time = 0.99 * np.mean(epoch_end_times - epoch_start_times)  # sec

        # find how many acquisition frames correspond to pre, stim, tail time
        epoch_frames = int(epoch_time / sample_period)  # in acquisition frames
        pre_frames = int(pre_time / sample_period)  # in acquisition frames
        time_vector = np.arange(0, epoch_frames) * sample_period  # sec

        no_trials = len(epoch_start_times)
        no_rois = response_trace.shape[0]
        response_matrix = np.empty(shape=(no_rois, no_trials, epoch_frames), dtype=float)
        response_matrix[:] = np.nan
        cut_inds = np.empty(0, dtype=int)
        for idx, val in enumerate(epoch_start_times):
            stack_inds = np.where(np.logical_and(stack_times < epoch_end_times[idx], stack_times >= epoch_start_times[idx]))[0]
            if len(stack_inds) == 0:  # no imaging acquisitions happened during this epoch presentation
                cut_inds = np.append(cut_inds,idx)
                continue
            if np.any(stack_inds > response_trace.shape[1]):
                cut_inds = np.append(cut_inds, idx)
                continue
            if idx != 0:
                if len(stack_inds) < epoch_frames:  # missed images for the end of the stimulus
                    cut_inds = np.append(cut_inds, idx)
                    print('Missed acquisition frames at the end of the stimulus!')
                    continue
            # pull out Roi values for these scans. shape of newRespChunk is (nROIs,nScans)
            new_resp_chunk = response_trace[:, stack_inds]

            if dff:
                # calculate baseline using pre frames
                baseline = np.mean(new_resp_chunk[:, 0:pre_frames], axis=1, keepdims=True)
                # to dF/F
                new_resp_chunk = (new_resp_chunk - baseline) / baseline

            response_matrix[:, idx, :] = new_resp_chunk[:, 0:epoch_frames]

        if len(cut_inds) > 0:
            print('Warning: cut {} epochs from epoch response matrix'.format(len(cut_inds)))
        response_matrix = np.delete(response_matrix, cut_inds, axis=1)
        return time_vector, response_matrix

    def generateRoiMap(self, roi_name, scale_bar_length=0, z=0):
        newImage = plot_tools.overlayImage(self.roi.get(roi_name).get('roi_image'), self.roi.get(roi_name).get('roi_mask'), 0.5, self.colors, z=z)

        fh = plt.figure(figsize=(4,4))
        ax = fh.add_subplot(111)
        ax.imshow(newImage)
        ax.set_aspect('equal')
        ax.set_axis_off()
        if scale_bar_length > 0:
            microns_per_pixel = float(self.metadata['micronsPerPixel_XAxis'])
            plot_tools.addImageScaleBar(ax, newImage, scale_bar_length, microns_per_pixel, 'lr')


def find_series(name, obj, sn):
    target_group_name = 'series_{}'.format(str(sn).zfill(3))
    if target_group_name in name:
        return obj
