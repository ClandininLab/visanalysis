"""
40 Hour Fitness rig plugin.

https://github.com/ClandininLab/visanalysis
"""
import os
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import h5py
import skimage.io as io
import functools
import nibabel as nib
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.signal as signal

from visanalysis.plugin import base as base_plugin
from visanalysis.util import h5io

from visanalysis.analysis.imaging_data import ImagingDataObject
from visanalysis.plugin.twentyfourhourfitness import TwentyFourHourDataObject

class FortyHourFitnessPlugin(base_plugin.BasePlugin):
    def __init__(self):
        super().__init__()
        self.current_series = None
        self.current_series_number = 0

    def attachData(self, experiment_file_name, file_path, data_directory):
        '''
        Attach Jackfish DAQ data.
        '''
        for series_number in self.getSeriesNumbers(file_path):
            print('Attaching data to series {}'.format(series_number))
            # # # # Retrieve metadata from files in data directory # # #
            series_directory = os.path.join(data_directory, str(series_number))
            if not os.path.exists(series_directory):
                print(f"Series {str(series_number)} does not exist in directory strucutre. Skipping...")
                continue
            
            daq_data_candidates = [x for x in os.listdir(series_directory) if x.endswith('.jfdaqdata')]
            if len(daq_data_candidates) > 0:
                jackfish_daq_data_path = os.path.join(series_directory, daq_data_candidates[0]) # take the first candidate
            
                # Photodiode trace
                voltage_recording, input_ch_nicknames, time_vector, sample_rate = getVoltageRecording(jackfish_daq_data_path)

                # Pick frame monitor(s) out of voltage recording traces based on name
                frame_monitor_ch_idxs = [i for i,name in enumerate(input_ch_nicknames) if name.startswith('frame_monitor')]
                frame_monitor = voltage_recording[frame_monitor_ch_idxs, :]

                # Behavior Timing information
                cams_strobe_ch_idxs, cams_strobe_ch_nicknames = tuple(zip(*[(i,name) for (i,name) in enumerate(input_ch_nicknames) if name.startswith('cam_strobe')]))
                cams_strobe_ch_idxs = list(cams_strobe_ch_idxs)
                cams_strobe_ch_nicknames = list(cams_strobe_ch_nicknames)
                
                cams_timing = {cam_nickname:{} for cam_nickname in cams_strobe_ch_nicknames}
                cams_strobe = voltage_recording[cams_strobe_ch_idxs, :]
                cams_strobe_diff = np.diff(cams_strobe, axis=1)

                for diff,nickname in zip(cams_strobe_diff, cams_strobe_ch_nicknames):
                    # approximate times in seconds, converted from indices
                    exposure_onset = time_vector[np.nonzero(diff== 1)[0] + 1]
                    exposure_offset = time_vector[np.nonzero(diff==-1)[0] + 1]

                    if len(exposure_onset) == 0:
                        print(f"Camera {nickname} had no strobes / exposures.")
                        cams_timing.pop(nickname)
                    else:
                        assert abs(len(exposure_onset) - len(exposure_offset)) <= 1
                        if len(exposure_onset) > len(exposure_offset): # Last exposure got truncated
                            exposure_onset = exposure_onset[:-1]
                        elif len(exposure_onset) < len(exposure_offset): # First exposure got truncated
                            exposure_offset = exposure_offset[1:]

                        exposure_times = exposure_offset - exposure_onset
                        cams_timing[nickname]['exposure_onset'] = exposure_onset
                        cams_timing[nickname]['exposure_offset'] = exposure_offset
                        cams_timing[nickname]['exposure_time'] = np.mean(exposure_times)
                        cams_timing[nickname]['frame_rate'] = 1/np.mean(np.diff(cams_timing[nickname]['exposure_onset']))
                        
                        if np.all(exposure_times < 0): # Noticed some spontaneous inverted strobe activity from Grasshopper camera... exception for such cases. Might be that GrassHopper strobe is inverted by default...? We invert in initializaiton now.
                            print(f"Camera {nickname} strobes were all inverted. Proceed with caution. Mean frame rate: {cams_timing[nickname]['frame_rate']}")
                        else:
                            assert np.all(exposure_times > 0), f"Camera {nickname} strobes were not all positive. Proceed with caution. Mean exposure time: {cams_timing[nickname]['exposure_time']}"
                                    
                # # # # Attach metadata to epoch run group in data file # # #\
                with h5py.File(file_path, 'r+') as experiment_file:
                    find_partial = functools.partial(h5io.find_series, sn=series_number)
                    epoch_run_group = experiment_file.visititems(find_partial)

                    # make sure subgroups exist for stimulus and response timing
                    stimulus_timing_group = epoch_run_group.require_group('stimulus_timing')
                    h5io.overwriteDataSet(stimulus_timing_group, 'frame_monitor', frame_monitor)
                    h5io.overwriteDataSet(stimulus_timing_group, 'time_vector', time_vector)
                    stimulus_timing_group.attrs['sample_rate'] = sample_rate

                    behavior_group = epoch_run_group.require_group('behavior')
                    for nickname in cams_timing:
                        cam_group = behavior_group.require_group(nickname)
                        h5io.overwriteDataSet(cam_group,  'exposure_onset', cams_timing[nickname][ 'exposure_onset'])
                        h5io.overwriteDataSet(cam_group, 'exposure_offset', cams_timing[nickname]['exposure_offset'])
                        cam_group.attrs['exposure_time'] = cams_timing[nickname]['exposure_time']
                        cam_group.attrs['frame_rate'] = cams_timing[nickname]['frame_rate']

                print('Attached timing data to series {}'.format(series_number))
            else:
                print('WARNING! Required DAQ data files not found at {}'.format(data_directory))

            #### Fictrac
            fictrac_directory = os.path.join(series_directory, 'loco')
            if os.path.exists(fictrac_directory):
                fictrac_data_path = [os.path.join(fictrac_directory, x) for x in os.listdir(fictrac_directory) if x.endswith('.dat')]                
                log_path = [os.path.join(fictrac_directory, x) for x in os.listdir(fictrac_directory) if x=='log.txt']
                assert len(log_path) == 1, "log.txt must exist in Fictrac data directory."
                log_path = log_path[0]
            else:
                print(f'WARNING! Loco directory {fictrac_directory} not found.')
                fictrac_data_path = []
                log_path = None
            
            if len(fictrac_data_path) > 0:
                fictrac_data_path = fictrac_data_path[0]
            
                fictrac_data_header = ['frame_count', 
                        'rel_vec_cam_x', 'rel_vec_cam_y', 'rel_vec_cam_z', 'error',
                        'rel_vec_world_x', 'rel_vec_world_y', 'rel_vec_world_z',
                        'abs_vec_cam_x', 'abs_vec_cam_y', 'abs_vec_cam_z',
                        'abs_vec_world_x', 'abs_vec_world_y', 'abs_vec_world_z',
                        'integrated_xpos', 'integrated_ypos', 'integrated_heading',
                        'direction', 'speed', 'integrated_x_movement', 'integrated_y_movement',
                        'timestamp', 'sequence_number', 'delta_ts', 'timestamp_alt']
                # fictrac_data = pd.DataFrame(np.genfromtxt(fictrac_data_path, delimiter=","), columns=fictrac_data_header)
                # fictrac_data = fictrac_data.astype({'frame_count': int, 'sequence_number': int})
                # fictrac_data = fictrac_data.set_index('frame_count')
                
                fictrac_data = np.genfromtxt(fictrac_data_path, delimiter=",")
                if 'cam_strobe_Fictrac' in cams_timing and len(cams_timing['cam_strobe_Fictrac']['exposure_onset']) < len(fictrac_data):
                    print('There are more Fictrac data lines than strobes.')

                # log file
                with open(log_path, 'r') as lf:
                    log_lines = lf.readlines()
                
                with h5py.File(file_path, 'r+') as experiment_file:
                    find_partial = functools.partial(h5io.find_series, sn=series_number)
                    epoch_run_group = experiment_file.visititems(find_partial)
                    behavior_group = epoch_run_group.require_group('behavior')
                    h5io.overwriteDataSet(behavior_group,  'fictrac_data', fictrac_data)
                    behavior_group['fictrac_data'].attrs['fictrac_data_header'] = fictrac_data_header
                    log_group = behavior_group.require_group('log_lines')
                    for i,log_line in enumerate(log_lines):
                        line_json = json.loads(log_line)
                        line_group = log_group.require_group(f'line_{i:03d}')
                        line_group.attrs['ts'] = line_json.pop('ts')
                        for log_k, log_v in line_json.items():
                            item_group = line_group.require_group(log_k)
                            for k,v in log_v.items():
                                item_group.attrs[k] = v

                    print('Attached Fictrac data to series {}'.format(series_number))
            else:
                print(f'WARNING! No Fictrac data found in {fictrac_data_path}.')

                
    # %%
    ###########################################################################
    # Functions for timing and metadata
    #   Accessible outside of the plugin object
    ###########################################################################


def getVoltageRecording(filepath):
    """
    Get frame monitor voltage traces, associated timestamps, and frame rate
    params:
        :filepath: path to voltage recording file, with no suffix
    """    
    
    with open(filepath, 'r') as jf:
        header = jf.readline()
        if header == '\n': # Old format with a blank header; assume header
            header = None
            input_channels = {"AIN0": "frame_monitor_R", 
                              "AIN2": "frame_monitor_C", 
                              "AIN4": "frame_monitor_L", 
                              "FIO0": "cam_strobe_Fictrac", 
                              "FIO2": "cam_strobe_Top", 
                              "FIO4": "cam_strobe_Left"}
            daq_framerate = 5000
            n_input_chs = len(input_channels)

            daq_data = []
            for line in jf:
                if line.endswith('\n'):
                    line = line[:-1]
                if line.startswith('['):
                    line = line[1:]
                if line.endswith(']'):
                    line = line[:-1]

                line = [float(entry) for entry in line.split(',') if entry!='']
                assert len(line)%n_input_chs == 0, line
                daq_data.extend(line)
            daq_data_np = np.asarray(daq_data).reshape((-1, n_input_chs))
            
        else: # New format with a header
            header = json.loads(header)
            daq_framerate = header['scan_rate']
            input_channels = header['input_channels']
            
            daq_data_np = np.genfromtxt(jf)

    input_ch_addresses = list(input_channels.keys())
    input_ch_nicknames = list(input_channels.values())    

    time_vector = np.arange(daq_data_np.shape[0]) / daq_framerate # in seconds; approximate
        
    return daq_data_np.T, input_ch_nicknames, time_vector, daq_framerate




# %%
###########################################################################
# DataObject specific to FortyHourFitness data. Inherits ImagingDataObject
###########################################################################

class FortyHourDataObject(ImagingDataObject):
    """
    FortyHourDataObject inherits ImagingDataObject and alters the getStimulusTiming method.
    """
    
    def getStimulusTiming(self,
                          plot_trace_flag=False,
                          use_square_photodiodes=True):
        """
        Returns stimulus timing information based on photodiode voltage trace from frame tracker signal.



        """
        if not use_square_photodiodes:
            return TwentyFourHourDataObject.getStimulusTiming(self, plot_trace_flag=plot_trace_flag)
        
        frame_monitor_channels, time_vector, sample_rate = self.getVoltageData()
        run_parameters = self.getRunParameters()
        epoch_parameters = self.getEpochParameters()

        # If more than two voltage channels, just take the LAST two in the list as photodiodes
        if len(frame_monitor_channels) > 3:
            frame_monitor_channels = frame_monitor_channels[-3:]

        if len(frame_monitor_channels.shape) == 1:
            frame_monitor_channels = frame_monitor_channels[np.newaxis, :]

        minimum_epoch_separation = 0.9 * (run_parameters['pre_time'] + run_parameters['tail_time']) * sample_rate

        num_channels = frame_monitor_channels.shape[0]
        channel_timing = []
        for ch in range(num_channels):
            frame_monitor = frame_monitor_channels[ch, :]

            # Low-pass filter frame_monitor trace
            b, a = signal.butter(4, min(10*self.command_frame_rate, sample_rate/2-1), btype='low', fs=sample_rate)
            frame_monitor = signal.filtfilt(b, a, frame_monitor)

            # Remove extreme values
            extreme_thresholds = np.percentile(frame_monitor, [0.1, 99.9])
            frame_monitor[frame_monitor<extreme_thresholds[0]] = np.nan
            frame_monitor[frame_monitor>extreme_thresholds[1]] = np.nan
            
            # shift & normalize so frame monitor trace lives on [0 1]
            frame_monitor = frame_monitor - np.nanmin(frame_monitor)
            frame_monitor = frame_monitor / np.nanmax(frame_monitor)

            # find frame flip times
            # V_orig = frame_monitor[0:-2]
            # V_shift = frame_monitor[1:-1]
            # ups = np.where(np.logical_and(V_orig < self.threshold, V_shift >= self.threshold))[0] + 1
            # downs = np.where(np.logical_and(V_orig >= self.threshold, V_shift < self.threshold))[0] + 1
            ideal_frame_len = 1 / self.command_frame_rate * sample_rate  # datapoints
            ideal_frame_len_samples = int(np.round(1 / self.command_frame_rate * sample_rate))  # datapoints
            min_peak_distance = int(np.floor(ideal_frame_len * 1.8))  # datapoints
            ups,peak_params = signal.find_peaks(frame_monitor, height=self.threshold, threshold=None, distance=min_peak_distance, prominence=0.04, width=None, wlen=None, rel_height=0.5, plateau_size=None)
            
            downs = []
            for i in range(len(ups)):
                up_0 = ups[i]
                # up_1 = ups[i+1]
                # sig = frame_monitor[up_0:up_1+1]
                down = up_0 + ideal_frame_len_samples
                downs.append(down)
            downs = np.asarray(downs)
                
                
            # downs,_ = signal.find_peaks(1-frame_monitor, height=(0.1, 0.85), threshold=None, distance=min_peak_distance, prominence=None, width=None, wlen=None, rel_height=0.5, plateau_size=None)
            if plot_trace_flag:
                plt.figure()
                plt.plot(frame_monitor)
                plt.plot(ups, np.ones(ups.shape), 'go')
                plt.plot(downs, np.ones(downs.shape), 'rx')
                plt.show()

            frame_times = np.sort(np.append(ups, downs)) # datapoints

            # Use frame flip times to find stimulus start times
            stimulus_start_frames = np.append(0, np.where(np.diff(frame_times) > minimum_epoch_separation)[0] + 1)
            stimulus_end_frames = np.append(np.where(np.diff(frame_times) > minimum_epoch_separation)[0], len(frame_times)-1)
            stimulus_start_times = frame_times[stimulus_start_frames] / sample_rate  # datapoints -> sec
            stimulus_end_times = frame_times[stimulus_end_frames] / sample_rate  # datapoints -> sec

            stim_durations = stimulus_end_times - stimulus_start_times  # sec

            ideal_frame_len = 1 / self.command_frame_rate * sample_rate  # datapoints
            frame_durations = []
            dropped_frame_times = []
            good_frame_times = []
            for s_ind, ss in enumerate(stimulus_start_frames):
                frame_len = np.diff(frame_times[stimulus_start_frames[s_ind]:stimulus_end_frames[s_ind]+1])
                dropped_frame_inds = np.where(np.abs(frame_len - ideal_frame_len)>self.frame_slop)[0]  # +1 b/c diff
                if len(dropped_frame_inds) > 0:
                    stim_dropped_frame_times = frame_times[ss+dropped_frame_inds]  # time when dropped frames should have flipped
                    dropped_frame_times.append(stim_dropped_frame_times)
                    # print('Warning! Ch. {} Dropped {} frames in epoch {}'.format(ch, len(dropped_frame_inds), s_ind))
                good_frame_inds = np.where(np.abs(frame_len - ideal_frame_len) <= self.frame_slop)[0]
                if len(good_frame_inds) > 0:
                    stim_good_frame_times = frame_times[ss+good_frame_inds]
                    good_frame_times.append(stim_good_frame_times)
                    frame_durations.append(np.diff(stim_good_frame_times))  # only include non-dropped frames in frame rate calc

            if len(dropped_frame_times) > 0:
                dropped_frame_times = np.hstack(dropped_frame_times)  # datapoints
            else:
                dropped_frame_times = np.array(dropped_frame_times)
            if len(good_frame_times) > 0:
                good_frame_times = np.hstack(good_frame_times)  # datapoints
            else:
                good_frame_times = np.array(good_frame_times)
            
            frame_durations = np.hstack(frame_durations)  # datapoints
            measured_frame_len = np.mean(frame_durations)  # datapoints
            frame_rate = 1 / (measured_frame_len / sample_rate)  # Hz

            if plot_trace_flag:
                frame_monitor_figure = plt.figure(figsize=(12, 8))
                gs1 = gridspec.GridSpec(2, 2)
                ax = frame_monitor_figure.add_subplot(gs1[1, :])
                ax.plot(time_vector, frame_monitor)
                # ax.plot(time_vector[frame_times], self.threshold * np.ones(frame_times.shape), 'ko')
                ax.plot(stimulus_start_times, self.threshold * np.ones(stimulus_start_times.shape), 'go', label='Stim start')
                ax.plot(stimulus_end_times, self.threshold * np.ones(stimulus_end_times.shape)-0.05, 'ro', label='Stim end')
                ax.plot(good_frame_times / sample_rate, 1 * np.ones(good_frame_times.shape), 'go', markerfacecolor='none', label='Good frame')
                ax.plot(dropped_frame_times / sample_rate, 1 * np.ones(dropped_frame_times.shape), 'rx', label='Dropped frame')
                ax.legend()
                ax.set_xlabel('Time [s]')
                ax.set_title('Ch. {}: Frame rate = {:.2f} Hz'.format(ch, frame_rate), fontsize=12)

                ax = frame_monitor_figure.add_subplot(gs1[0, 0])
                ax.hist(frame_durations)
                ax.axvline(ideal_frame_len, color='k')
                ax.set_xlabel('Frame duration (datapoints)')

                ax = frame_monitor_figure.add_subplot(gs1[0, 1])
                ax.plot(stim_durations, 'b.')
                if 'stim_time' in run_parameters:
                    ax.axhline(y=run_parameters['stim_time'], xmin=0, xmax=run_parameters['num_epochs'], color='k', linestyle='-', marker='None', alpha=0.50)
                else:
                    ax.axhline(y=np.mean(stim_durations), xmin=0, xmax=run_parameters['num_epochs'], color='k', linestyle='-', marker='None', alpha=0.50)
                ymin = 0.9 * np.min(stim_durations)
                ymax = 1.1 * np.max(stim_durations)
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
                if len(inter_stim_starts) >= 1:
                    if 'stim_time' in run_parameters:
                        print('Stim start to start: [min={:.3f}, median={:.3f}, max={:.3f}] / parameterized = {:.3f} sec'.format(inter_stim_starts.min(),
                                                                                                                                np.median(inter_stim_starts),
                                                                                                                                inter_stim_starts.max(),
                                                                                                                                run_parameters['stim_time'] + run_parameters['pre_time'] + run_parameters['tail_time']))
                    else:
                        print('Stim start to start: [min={:.3f}, median={:.3f}, max={:.3f}] sec'.format(inter_stim_starts.min(),
                                                                                                                                np.median(inter_stim_starts),
                                                                                                                                inter_stim_starts.max()))
                if 'stim_time' in run_parameters:
                    print('Stim duration: [min={:.3f}, median={:.3f}, max={:.3f}] / parameterized = {:.3f} sec'.format(stim_durations.min(), np.median(stim_durations), stim_durations.max(), run_parameters['stim_time']))
                else:
                    print('Stim duration: [min={:.3f}, median={:.3f}, max={:.3f}] sec'.format(stim_durations.min(), np.median(stim_durations), stim_durations.max()))
                total_frames = len(frame_times)
                dropped_frames = len(dropped_frame_times)
                print('Dropped {} / {} frames ({:.2f}%)'.format(dropped_frames, total_frames, 100*dropped_frames/total_frames))
                print('==========================================================')

            new_dict = {'stimulus_end_times': stimulus_end_times,
                        'stimulus_start_times': stimulus_start_times,
                        'dropped_frame_times': dropped_frame_times,
                        'frame_rate': frame_rate}
            channel_timing.append(new_dict)

        return channel_timing[self.timing_channel_ind]
