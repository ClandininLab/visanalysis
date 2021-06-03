"""
ImagingDataObject for visanalysis data files.

Associated with an hdf5 data file and series number

https://github.com/ClandininLab/visanalysis
mhturner@stanford.edu
"""
import functools
import os
import h5py
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.signal as signal

from visanalysis import plot_tools


class ImagingDataObject():
    __slots__ = ["file_path", "series_number",  "colors", "quiet"]
    def __init__(self, file_path, series_number, quiet=False, kwargs=None):
        kwargs_passed = {'plot_trace_flag': False}

        if kwargs is not None:
            for key in kwargs:
                kwargs_passed[key] = kwargs[key]

        self.file_path = file_path
        self.series_number = series_number
        self.quiet = quiet

        self.colors = sns.color_palette("Set2", n_colors=20)

        # check to see if hdf5 file exists
        if not os.path.exists(self.file_path):
            raise Exception('No hdf5 file found at {}, check your filepath'.format(self.file_path))

        # check to see if series exists in this file:
        with h5py.File(self.file_path, 'r') as experiment_file:
            find_partial = functools.partial(find_series, series_number=self.series_number)
            epoch_run_group = experiment_file.visititems(find_partial)
            if epoch_run_group is None:
                raise Exception('No series {} found in {}'.format(self.series_number, self.file_path))


    def getRunParameters(self):
        """Return epoch run parameters as dict."""
        with h5py.File(self.file_path, 'r') as experiment_file:
            find_partial = functools.partial(find_series, series_number=self.series_number)
            epoch_run_group = experiment_file.visititems(find_partial)
            run_parameters = {}
            for attr_key in epoch_run_group.attrs:
                run_parameters[attr_key] = epoch_run_group.attrs[attr_key]

        return run_parameters

    def getEpochParameters(self):
        """Return list of epoch parameters, one dict for each trial."""
        with h5py.File(self.file_path, 'r') as experiment_file:
            find_partial = functools.partial(find_series, series_number=self.series_number)
            epoch_run_group = experiment_file.visititems(find_partial)
            epoch_parameters = []
            for epoch in epoch_run_group['epochs'].values():
                new_params = {}
                for attr_key in epoch.attrs:
                    new_params[attr_key] = epoch.attrs[attr_key]
                epoch_parameters.append(new_params)
        return epoch_parameters

    def getFlyMetadata(self):
        """Return fly metadata as dict."""
        with h5py.File(self.file_path, 'r') as experiment_file:
            find_partial = functools.partial(find_series, series_number=self.series_number)
            epoch_run_group = experiment_file.visititems(find_partial)
            fly_group = epoch_run_group.parent.parent
            fly_metadata = {}
            for attr_key in fly_group.attrs:
                fly_metadata[attr_key] = fly_group.attrs[attr_key]

        return fly_metadata

    def getAcquisitionMetadata(self):
        """Return imaging acquisition metadata as dict."""
        with h5py.File(self.file_path, 'r') as experiment_file:
            find_partial = functools.partial(find_series, series_number=self.series_number)
            epoch_run_group = experiment_file.visititems(find_partial)
            acquisition_group = epoch_run_group['acquisition']
            acquisition_metadata = {}
            for attr_key in acquisition_group.attrs:
                acquisition_metadata[attr_key] = acquisition_group.attrs[attr_key]
        return acquisition_metadata

    def getPhotodiodeData(self):
        """
        Get photodiode trace data.

        Returns:
            photodiode_trace: array, shape=(n photodiode channels, n time points)
            photodiode_time_vector: array
            photodiode_sample_rate: (Hz)
        """
        with h5py.File(self.file_path, 'r') as experiment_file:
            find_partial = functools.partial(find_series, series_number=self.series_number)
            epoch_run_group = experiment_file.visititems(find_partial)
            stimulus_timing_group = epoch_run_group['stimulus_timing']

            photodiode_trace = stimulus_timing_group.get('frame_monitor')[:]
            if len(photodiode_trace.shape) < 2:
                # dummy dim for single channel photodiode
                photodiode_trace = photodiode_trace[np.newaxis, :]
            photodiode_time_vector = stimulus_timing_group.get('time_vector')[:]
            photodiode_sample_rate = stimulus_timing_group.attrs['sample_rate']

        return photodiode_trace, photodiode_time_vector, photodiode_sample_rate

    def getResponseTiming(self):
        """
        Get imaging timing.

        Returns:
            response_timing: dict

        """
        with h5py.File(self.file_path, 'r') as experiment_file:
            find_partial = functools.partial(find_series, series_number=self.series_number)
            epoch_run_group = experiment_file.visititems(find_partial)
            acquisition_group = epoch_run_group['acquisition']

            response_timing = {}
            response_timing['time_vector'] = acquisition_group.get('time_points')[:]  # sec
            response_timing['sample_period'] = acquisition_group.attrs['sample_period']  # sec

            return response_timing

    def getStimulusTiming(self,
                          plot_trace_flag=False,
                          threshold=0.8,
                          frame_slop=20,  # datapoints +/- ideal frame duration
                          command_frame_rate=120):
        """
        Returns stimulus timing information based on photodiode voltage trace from frame tracker signal.



        """
        frame_monitor_channels, time_vector, sample_rate = self.getPhotodiodeData()
        run_parameters = self.getRunParameters()
        epoch_parameters = self.getEpochParameters()

        if len(frame_monitor_channels.shape) == 1:
            frame_monitor_channels = frame_monitor_channels[np.newaxis, :]

        minimum_epoch_separation = 0.9 * (run_parameters['pre_time'] + run_parameters['tail_time']) * sample_rate

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
            stimulus_end_frames = np.append(np.where(np.diff(frame_times) > minimum_epoch_separation)[0], len(frame_times)-1)
            stimulus_start_times = frame_times[stimulus_start_frames] / sample_rate  # datapoints -> sec
            stimulus_end_times = frame_times[stimulus_end_frames] / sample_rate  # datapoints -> sec

            stim_durations = stimulus_end_times - stimulus_start_times # sec

            ideal_frame_len = 1 / command_frame_rate * sample_rate  # datapoints
            frame_durations = []
            dropped_frame_times = []
            for s_ind, ss in enumerate(stimulus_start_frames):
                frame_len = np.diff(frame_times[stimulus_start_frames[s_ind]:stimulus_end_frames[s_ind]+1])
                dropped_frame_inds = np.where(np.abs(frame_len - ideal_frame_len)>frame_slop)[0] + 1 # +1 b/c diff
                if len(dropped_frame_inds) > 0:
                    dropped_frame_times.append(frame_times[ss]+dropped_frame_inds * ideal_frame_len) # time when dropped frames should have flipped
                    # print('Warning! Ch. {} Dropped {} frames in epoch {}'.format(ch, len(dropped_frame_inds), s_ind))
                good_frame_inds = np.where(np.abs(frame_len - ideal_frame_len) <= frame_slop)[0]
                frame_durations.append(frame_len[good_frame_inds]) # only include non-dropped frames in frame rate calc

            if len(dropped_frame_times) > 0:
                dropped_frame_times = np.hstack(dropped_frame_times) # datapoints
            else:
                dropped_frame_times = np.array(dropped_frame_times)

            frame_durations = np.hstack(frame_durations) # datapoints
            measured_frame_len = np.mean(frame_durations)  # datapoints
            frame_rate = 1 / (measured_frame_len / sample_rate)  # Hz

            if plot_trace_flag:
                frame_monitor_figure = plt.figure(figsize=(12, 8))
                gs1 = gridspec.GridSpec(2, 2)
                ax = frame_monitor_figure.add_subplot(gs1[1, :])
                ax.plot(time_vector, frame_monitor)
                # ax.plot(time_vector[frame_times], threshold * np.ones(frame_times.shape), 'ko')
                ax.plot(stimulus_start_times, threshold * np.ones(stimulus_start_times.shape), 'go')
                ax.plot(stimulus_end_times, threshold * np.ones(stimulus_end_times.shape), 'ro')
                ax.plot(dropped_frame_times / sample_rate, 1 * np.ones(dropped_frame_times.shape), 'rx')
                ax.set_title('Ch. {}: Frame rate = {:.2f} Hz'.format(ch, frame_rate), fontsize=12)

                ax = frame_monitor_figure.add_subplot(gs1[0, 0])
                ax.hist(frame_durations)
                ax.axvline(ideal_frame_len, color='k')
                ax.set_xlabel('Frame duration (datapoints)')

                ax = frame_monitor_figure.add_subplot(gs1[0, 1])
                ax.plot(stim_durations, 'b.')
                ax.axhline(y=run_parameters['stim_time'], xmin=0, xmax=run_parameters['num_epochs'], color='k', linestyle='-', marker='None', alpha=0.50)
                ymin = np.min([0.9 * run_parameters['stim_time'], np.min(stim_durations)])
                ymax = np.max([1.1 * run_parameters['stim_time'], np.max(stim_durations)])
                ax.set_ylim([ymin, ymax])
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Stim duration (sec)')

                frame_monitor_figure.tight_layout()
                plt.show()

            if self.quiet:
                pass
            else:
                # Print timing summary
                print('===================TIMING: Channel {}======================'.format(ch))
                print('{} Stims presented (of {} parameterized)'.format(len(stim_durations), len(epoch_parameters)))
                inter_stim_starts = np.diff(stimulus_start_times)
                print('Stim start to start: [min={:.3f}, median={:.3f}, max={:.3f}] / parameterized = {:.3f} sec'.format(inter_stim_starts.min(),
                                                                                                                         np.median(inter_stim_starts),
                                                                                                                         inter_stim_starts.max(),
                                                                                                                         run_parameters['stim_time'] + run_parameters['pre_time'] + run_parameters['tail_time']))
                print('Stim duration: [min={:.3f}, median={:.3f}, max={:.3f}] / parameterized = {:.3f} sec'.format(stim_durations.min(), np.median(stim_durations), stim_durations.max(), run_parameters['stim_time']))
                total_frames = len(frame_times)
                dropped_frames = len(dropped_frame_times)
                print('Dropped {} / {} frames ({:.2f}%)'.format(dropped_frames, total_frames, 100*dropped_frames/total_frames))
                print('==========================================================')

        # for stimulus_timing just use one of the channels, both *should* be in sync
        stimulus_timing = {'stimulus_end_times': stimulus_end_times,
                           'stimulus_start_times': stimulus_start_times,
                           'dropped_frame_times': dropped_frame_times,
                           'frame_rate': frame_rate}
        return stimulus_timing

    def getRoiSetNames(self):
        roi_set_names = []
        with h5py.File(self.file_path, 'r') as experiment_file:
            find_partial = functools.partial(find_series, series_number=self.series_number)
            roi_parent_group = experiment_file.visititems(find_partial)['rois']
            for roi_set_name in roi_parent_group.keys():
                roi_set_names.append(roi_set_name)

        return roi_set_names

    def getRoiResponses(self, roi_set_name, background_subtraction=False):
        """
        Get responses for indicated roi
        Params:
            -roi_set_name: (str) name of roi set to pull out
            -background_subtraction: (Bool) subtract background roi values.
                There must be a roi set for this series called 'bg'

        Returns:
            roi_data: dict, keys:
                        roi_response: ndarry, shape = (rois, time)
                        roi_mask: list of ndarray masks, one for each roi in roi set
                        roi_image: ndarray image showing roi overlay
                        epoch_response: ndarray, shape = (rois, epochs, time)
                        time_vector: 1d array, time values for epoch_response traces (sec)
        """
        roi_data = {}
        with h5py.File(self.file_path, 'r') as experiment_file:
            find_partial = functools.partial(find_series, series_number=self.series_number)
            roi_parent_group = experiment_file.visititems(find_partial)['rois']
            roi_set_group = roi_parent_group[roi_set_name]
            roi_data['roi_response'] = list(roi_set_group.get("roi_response")[:])
            roi_data['roi_mask'] = list(roi_set_group.get("roi_mask")[:])
            roi_data['roi_image'] = roi_set_group.get("roi_image")[:]

        if background_subtraction:
            with h5py.File(self.file_path, 'r') as experiment_file:
                find_partial = functools.partial(find_series, series_number=self.series_number)
                roi_parent_group = experiment_file.visititems(find_partial)['rois']
                bg_roi_group = roi_parent_group['bg']
                bg_roi_response = list(bg_roi_group.get("roi_response")[:])

            roi_data['roi_response'] = roi_data['roi_response'] - bg_roi_response

        time_vector, response_matrix = self.getEpochResponseMatrix(roi_data.get('roi_response'))

        roi_data['epoch_response'] = response_matrix
        roi_data['time_vector'] = time_vector

        return roi_data

    def getEpochResponseMatrix(self, roi_response, dff=True):
        """
        getEpochReponseMatrix(self, roi_response, dff=True)
            Takes in long stack response traces and splits them up into each stimulus epoch
            Params:
                roi_response: list of roi responses
                dff: (Bool) convert from raw intensity value to dF/F based on mean of pre_time

            Returns:
                time_vector (1d array): 1d array, time values for epoch_response traces (sec)
                response_matrix (ndarray): response for each roi in each epoch.
                    shape = (rois, epochs, frames per epoch)
        """
        roi_response = np.vstack(roi_response)

        run_parameters = self.getRunParameters()
        response_timing = self.getResponseTiming()
        stimulus_timing = self.getStimulusTiming()

        epoch_start_times = stimulus_timing['stimulus_start_times'] - run_parameters['pre_time']
        epoch_end_times = stimulus_timing['stimulus_end_times'] + run_parameters['tail_time']
        # Use measured stimulus lengths for stim time instead of epoch param
        # cut off a bit of the end of each epoch to allow for slop in how many frames were acquired
        epoch_time = 0.97*(run_parameters['pre_time'] +
                           run_parameters['stim_time'] +
                           run_parameters['tail_time']) # sec

        # find how many acquisition frames correspond to pre, stim, tail time
        epoch_frames = int(epoch_time / response_timing['sample_period'])  # in acquisition frames
        pre_frames = int(run_parameters['pre_time'] / response_timing['sample_period'])  # in acquisition frames
        time_vector = np.arange(0, epoch_frames) * response_timing['sample_period']  # sec

        no_trials = len(epoch_start_times)
        no_rois = roi_response.shape[0]
        response_matrix = np.empty(shape=(no_rois, no_trials, epoch_frames), dtype=float)
        response_matrix[:] = np.nan
        cut_inds = np.empty(0, dtype=int)
        for idx, val in enumerate(epoch_start_times):
            stack_inds = np.where(np.logical_and(response_timing['time_vector'] < epoch_end_times[idx],
                                                 response_timing['time_vector'] >= epoch_start_times[idx]))[0]
            if len(stack_inds) == 0:  # no imaging acquisitions happened during this epoch presentation
                cut_inds = np.append(cut_inds, idx)
                continue
            if np.any(stack_inds > roi_response.shape[1]):
                cut_inds = np.append(cut_inds, idx)
                continue
            if idx == no_trials:
                if len(stack_inds) < epoch_frames:  # missed images for the end of the stimulus
                    cut_inds = np.append(cut_inds, idx)
                    print('Missed acquisition frames at the end of the stimulus!')
                    continue
            # pull out Roi values for these scans. shape of newRespChunk is (nROIs,nScans)
            new_resp_chunk = roi_response[:, stack_inds]

            if dff:
                # calculate baseline using pre frames
                baseline = np.mean(new_resp_chunk[:, 0:pre_frames], axis=1, keepdims=True)
                # to dF/F
                new_resp_chunk = (new_resp_chunk - baseline) / baseline

            try:
                response_matrix[:, idx, :] = new_resp_chunk[:, 0:epoch_frames]
            except:
                print('Size mismatch idx = {}'.format(idx)) # the end of a response clipped off
                cut_inds = np.append(cut_inds, idx)

        if len(cut_inds) > 0:
            print('Warning: cut {} epochs from epoch response matrix'.format(len(cut_inds)))
        response_matrix = np.delete(response_matrix, cut_inds, axis=1)
        return time_vector, response_matrix

    def generateRoiMap(self, roi_name, scale_bar_length=0, z=0):
        """
        Make roi map image in a new figure.

        Params:
            roi_name: str
            scale_bar_length: (microns)
            z: index of z plane to display, for xyz images
        """
        roi_data = self.getRoiResponses(roi_name)
        new_image = plot_tools.overlayImage(roi_data.get('roi_image'),
                                            roi_data.get('roi_mask'), 0.5, self.colors, z=z)

        fh, ax = plt.subplots(1, 1, figsize=(4,4))
        ax.imshow(new_image)
        ax.set_aspect('equal')
        ax.set_axis_off()
        if scale_bar_length > 0:
            microns_per_pixel = float(self.getAcquisitionMetadata()['micronsPerPixel_XAxis'])
            plot_tools.addImageScaleBar(ax, new_image, scale_bar_length, microns_per_pixel, 'lr')


def find_series(name, obj, series_number):
    """return hdf5 group object if it corresponds to indicated series_number."""
    target_group_name = 'series_{}'.format(str(series_number).zfill(3))
    if target_group_name in name:
        return obj
    return None
