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

from visanalysis.plugin import base as base_plugin
from visanalysis.util import h5io


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
                            assert np.all(exposure_times > 0)
                                    
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
                print('WARNING! Required data files not found at {}'.format(data_directory))

            #### Fictrac
            fictrac_directory = os.path.join(series_directory, 'loco')
            fictrac_data_path = [os.path.join(fictrac_directory, x) for x in os.listdir(fictrac_directory) if x.endswith('.dat')]
            log_path = [os.path.join(fictrac_directory, x) for x in os.listdir(fictrac_directory) if x=='log.txt']
            assert len(log_path) == 1, "log.txt must exist in Fictrac data directory."
            log_path = log_path[0]
            
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
                if len(cams_timing['cam_strobe_Fictrac']['exposure_onset']) < len(fictrac_data):
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

