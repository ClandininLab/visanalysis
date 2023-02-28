"""
40 Hour Fitness rig plugin.

https://github.com/ClandininLab/visanalysis
"""
import os
import numpy as np
import h5py
import functools
import json
from datetime import timezone
import dateutil

from visanalysis.plugin import base as base_plugin
from visanalysis.util import h5io

from visanalysis.analysis.imaging_data import ImagingDataObject

class TwentyFourHourFitnessPlugin(base_plugin.BasePlugin):
    def __init__(self):
        super().__init__()
        self.current_series = None
        self.current_series_number = 0

    def attachData(self, experiment_file_name, file_path, data_directory):
        '''
        Attach Fictrac data.
        '''
        for series_number in self.getSeriesNumbers(file_path):
            print('Attaching data to series {}'.format(series_number))
            # # # # Retrieve metadata from files in data directory # # #
            series_directory = os.path.join(data_directory, str(series_number))
            if not os.path.exists(series_directory):
                print(f"Series {str(series_number)} does not exist in directory strucutre. Skipping...")
                continue

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
                fictrac_data = np.genfromtxt(fictrac_data_path, delimiter=",")

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

class TwentyFourHourDataObject(ImagingDataObject):
    """
    TwentyFourHourDataObject inherits ImagingDataObject and alters the getStimulusTiming method.
    """
    __slots__ = ["file_path", "series_number",  "colors", "quiet",
                 "timing_channel_ind", "threshold", "frame_slop", "command_frame_rate"]
        
    def getStimulusTiming(self,
                          plot_trace_flag=False):
        """
        Returns stimulus timing information based on Visprotocol epoch times.
        """

        run_parameters = self.getRunParameters()
        epoch_parameters = self.getEpochParameters()

        stimulus_start_times = []
        stimulus_end_times = []
        
        with h5py.File(self.file_path, 'r') as experiment_file:
            find_partial = functools.partial(h5io.find_series, sn=self.series_number)
            epochs_group = experiment_file.visititems(find_partial)['epochs']
            n_epochs = len(epochs_group)
            for e, epoch in enumerate(epochs_group.values()):
                # Stimulus start time
                if 'epoch_unix_time' in epoch.attrs:
                    epoch_unix_time = epoch.attrs['epoch_unix_time']
                else:
                    # For older VP data when epoch_unix_time was not saved, get date and epoch_time and convert to unix time
                    date = experiment_file.attrs['date']
                    epoch_time = epoch.attrs['epoch_time']
                    epoch_unix_time = dateutil.parser.parse(date+' '+epoch_time).astimezone(timezone.utc).timestamp()
                stimulus_start_times.append(epoch_unix_time)
                
                # Stimulus end time
                if 'epoch_end_unix_time' in epoch.attrs:
                    epoch_end_unix_time = epoch.attrs['epoch_end_unix_time']
                else:
                    # For older VP data when epoch_end_unix_time was not saved, get stim_duration and add to epoch_unix_time
                    if 'stim_time' in run_parameters:
                        stim_duration = run_parameters['stim_time']
                    elif 'current_stim_time' in epoch_parameters:
                        stim_duration = epoch_parameters['current_stim_time']
                    else: # This will break when we only have 1 epoch, so hopefully we didn't enter this scenario
                        if e == n_epochs-1:
                            stim_duration = stimulus_end_times[-1] # Use previous epoch end time as a proxy
                        else:
                            iti = run_parameters['pre_time'] + run_parameters['tail_time']
                            stim_duration = epochs_group[f'epoch_{e+2:03d}'].attrs['epoch_unix_time'] - epoch_unix_time - iti
                    epoch_end_unix_time = epoch_unix_time + stim_duration
                stimulus_end_times.append(epoch_end_unix_time)
            
        stimulus_timing = {'stimulus_end_times': stimulus_end_times,
                           'stimulus_start_times': stimulus_start_times}

        return stimulus_timing
