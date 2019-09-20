# -*- coding: utf-8 -*-
"""
Created on Fri May 31 09:43:59 2019

@author: mhturner
"""
import os
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import h5py
import skimage.io as io
from registration import CrossCorr
from tifffile import imsave
import functools


from visanalysis import plugin, roi


##############################################################################
# Functions for image series data from bruker / PV
##############################################################################

class BrukerPlugin(plugin.base.BasePlugin):
    def __init__(self):
        super().__init__()
        self.current_series = None
        self.current_series_number = 0

    def getRoiImage(self, **kwargs):
        if kwargs.get('series_number') != self.current_series_number:
            self.current_series_number = kwargs.get('series_number')
            self.current_series = self.loadImageSeries(kwargs.get('experiment_file_name'),
                                                       kwargs.get('data_directory'),
                                                       kwargs.get('series_number'))
        if self.current_series is None:  # No image file found
            roi_image = None
        else:
            roi_image = np.mean(self.current_series, axis=0)  # avg across time
        return roi_image

    def getRoiDataFromPath(self, roi_path, data_directory, series_number, experiment_file_name, experiment_file_path):
        if series_number != self.current_series_number:
            self.current_series_number = series_number
            self.current_series = self.loadImageSeries(experiment_file_name, data_directory, series_number)
        roi_image = np.mean(self.current_series, axis=0)
        mask = roi.getRoiMaskFromPath(roi_image, roi_path)
        roi_response = (np.mean(self.current_series[:, mask], axis=1, keepdims=True) - np.min(self.current_series)).T
        return roi_response

    def loadImageSeries(self, experiment_file_name, data_directory, series_number):
        image_series_name = 'TSeries-' + experiment_file_name.replace('-','') + '-' + ('00' + str(series_number))[-3:]
        #  Check to see if this series has already been registered
        raw_file_path = os.path.join(data_directory, image_series_name) + '.tif'
        reg_file_path = os.path.join(data_directory, image_series_name) + '_reg.tif'

        if os.path.isfile(reg_file_path):
            image_series = io.imread(reg_file_path)
        elif os.path.isfile(raw_file_path):
            image_series = io.imread(raw_file_path)
            print('!! Warning: no registered series found !!')
        else:
            image_series = None
            print('File not found at {}'.format(raw_file_path))

        return image_series

    def registerStack(self, image_series, response_timing):
        """
        """

        reference_time_frame = 1  # sec, first frames to use as reference for registration
        reference_frame = np.where(response_timing['stack_times'] > reference_time_frame)[0][0]

        reference_image = np.squeeze(np.mean(image_series[0:reference_frame,:,:], axis = 0))
        register = CrossCorr()
        model = register.fit(image_series, reference=reference_image)

        registered_series = model.transform(image_series)
        if len(registered_series.shape) == 3:  # xyt
            registered_series = registered_series.toseries().toarray().transpose(2,0,1)  # shape t, y, x
        elif len(registered_series.shape) == 4:  # xyzt
            registered_series = registered_series.toseries().toarray().transpose(3,0,1,2)  # shape t, z, y, x

        return registered_series

    def registerAndSaveStacks(self, experiment_file_name, file_path, data_directory):
        print('Registering stacks...')
        for series_number in self.getSeriesNumbers(file_path):
            image_series_name = 'TSeries-' + experiment_file_name.replace('-','') + '-' + ('00' + str(series_number))[-3:]
            #  Check to see if this series has already been registered
            raw_file_path = os.path.join(data_directory, image_series_name) + '.tif'
            if os.path.isfile(raw_file_path):
                image_series = io.imread(raw_file_path)
            else:
                print('File not found at {}'.format(raw_file_path))
                continue

            response_timing = self.getAcquisitionTiming(experiment_file_name,
                                                        data_directory,
                                                        series_number)

            registered_series = self.registerStack(image_series, response_timing)
            save_path = raw_file_path.split('.')[0] + '_reg' + '.tif'
            print('Saved: ' + save_path)
            imsave(save_path, registered_series)
        print('Stacks registered')

    def attachData(self, experiment_file_name, file_path, data_directory):
        for series_number in self.getSeriesNumbers(file_path):
            # # # # Retrieve metadata from files in data directory # # #
            # Photodiode trace
            frame_monitor, time_vector, sample_rate = self.getPhotodiodeSignal(experiment_file_name,
                                                                               data_directory,
                                                                               series_number)

            # Imaging acquisition timing information
            response_timing = self.getAcquisitionTiming(experiment_file_name,
                                                        data_directory,
                                                        series_number)
            # Imaging metadata
            metadata = self.getMetaData(experiment_file_name,
                                        data_directory,
                                        series_number)
            # # # # Attach metadata to epoch run group in data file # # #\
            def find_series(name, obj, sn):
                target_group_name = 'series_{}'.format(str(sn).zfill(3))
                if target_group_name in name:
                    return obj

            with h5py.File(file_path, 'r+') as experiment_file:
                find_partial = functools.partial(find_series, sn=series_number)
                epoch_run_group = experiment_file.visititems(find_partial)

                #make sure subgroups exist for stimulus and response timing
                stimulus_timing_group = epoch_run_group.require_group('stimulus_timing')
                plugin.base.overwriteDataSet(stimulus_timing_group, 'frame_monitor', frame_monitor)
                plugin.base.overwriteDataSet(stimulus_timing_group, 'time_vector', time_vector)
                stimulus_timing_group.attrs['sample_rate'] = sample_rate

                acquisition_group = epoch_run_group.require_group('acquisition')
                plugin.base.overwriteDataSet(acquisition_group, 'time_points', response_timing['stack_times'])
                if 'frame_times' in response_timing:
                    plugin.base.overwriteDataSet(acquisition_group, 'frame_times', response_timing['frame_times'])
                acquisition_group.attrs['sample_period'] = response_timing['sample_period']

                for key in metadata:
                    acquisition_group.attrs[key] = metadata[key]

            print('Attached data to series {}'.format(series_number))


    # %%
    ##############################################################################
    # Functions for timing and metadata
    ##############################################################################

    def getAcquisitionTiming(self, experiment_file_name, data_directory, series_number):
        """
        Bruker imaging acquisition metadata based on the bruker metadata file (xml)
        """
        image_series_name = 'TSeries-' + experiment_file_name.replace('-','') + '-' + ('00' + str(series_number))[-3:]
        metaData = ET.parse(os.path.join(data_directory, image_series_name) + '.xml')
        root = metaData.getroot()

        if root.find('Sequence').get('type') == 'TSeries ZSeries Element':
            # volumetric xyz time series
            num_t = len(root.findall('Sequence'))
            num_z = len(root.find('Sequence').findall('Frame'))
            frame_times = np.ndarray(shape=(num_t, num_z), dtype=float)
            frame_times[:] = np.nan
            for t_ind, t_step in enumerate(root.findall('Sequence')):
                for z_ind, z_step in enumerate(t_step.findall('Frame')):
                    frame_times[t_ind, z_ind] = z_step.get('relativeTime')

            stack_times = frame_times[:, 0]
            sample_period = np.mean(np.diff(stack_times))

            response_timing = {'frame_times': frame_times,
                               'stack_times': stack_times,
                               'sample_period': sample_period}

        elif root.find('Sequence').get('type') == 'TSeries Timed Element':
            # Single-plane, xy time series
            stack_times = []
            for frame in root.find('Sequence').findall('Frame'):
                frTime = frame.get('relativeTime')
                stack_times.append(float(frTime))

            stack_times = np.array(stack_times)

            sample_period = np.mean(np.diff(stack_times))  # sec
            response_timing = {'stack_times': stack_times,
                               'sample_period': sample_period}

        return response_timing

    def getMetaData(self, experiment_file_name, data_directory, series_number):
        image_series_name = 'TSeries-' + experiment_file_name.replace('-','') + '-' + ('00' + str(series_number))[-3:]
        metaData = ET.parse(os.path.join(data_directory, image_series_name) + '.xml')
        root = metaData.getroot()

        metadata = {}
        for child in list(root.find('PVStateShard')):
            if child.get('value') is None:
                for subchild in list(child):
                    if subchild.get('value') is None:
                        for subsubchild in list(subchild):
                            new_key = child.get('key') + '_' + subchild.get('index') + subsubchild.get('subindex')
                            new_value = subsubchild.get('value')

                    else:
                        new_key = child.get('key') + '_' + subchild.get('index')
                        new_value = subchild.get('value')

            else:
                new_key = child.get('key')
                new_value = child.get('value')

            metadata[new_key] = new_value

        metadata['version'] = root.get('version')
        metadata['date'] = root.get('date')
        metadata['notes'] = root.get('notes')

        return metadata

    def getPhotodiodeSignal(self, experiment_file_name, data_directory, series_number,
                            v_rec_suffix = '_Cycle00001_VoltageRecording_001'):
        """
        """

        image_series_name = 'TSeries-' + experiment_file_name.replace('-','') + '-' + ('00' + str(series_number))[-3:]
        metadata = ET.parse(os.path.join(data_directory, image_series_name) + v_rec_suffix + '.xml')
        root = metadata.getroot()
        rate_node = root.find('Experiment').find('Rate')
        sample_rate = int(rate_node.text)

        active_channels = []
        signal_list = root.find('Experiment').find('SignalList').getchildren()
        for signal_node in signal_list:
            is_channel_active = signal_node.find('Enabled').text
            channel_name = signal_node.find('Name').text
            if is_channel_active == 'true':
                active_channels.append(channel_name)

        # Load frame tracker signal and pull frame/epoch timing info
        data_frame = pd.read_csv(os.path.join(data_directory, image_series_name) + v_rec_suffix + '.csv');

        time_vector = data_frame.get('Time(ms)').values / 1e3  # ->sec
        #for now takes first enabled channel.
        frame_monitor = data_frame.get(' ' + active_channels[0]).values

        return frame_monitor, time_vector, sample_rate
