"""
Bruker / Prairie View plugin.

https://github.com/ClandininLab/visanalysis
mhturner@stanford.edu
"""
import os
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import h5py
import skimage.io as io
import functools
import nibabel as nib

from visanalysis.plugin import base as base_plugin
from visanalysis.util import h5io


class BrukerPlugin(base_plugin.BasePlugin):
    def __init__(self):
        super().__init__()
        self.current_series = None
        self.mean_brain = None
        self.current_channel = 1  # 0, 1
        self.current_image_file_name = None

    def updateImageSeries(self, data_directory, image_file_name, series_number, channel):
        if image_file_name != self.current_image_file_name or channel != self.current_channel:  # only re-load if selected new image file
            self.current_image_file_name = image_file_name
            self.current_channel = channel
            self.loadImageSeries(data_directory=data_directory,
                                 image_file_name=image_file_name,
                                 channel=channel)
            print('Loaded image series from {}:{}, channel {}'.format(data_directory, image_file_name, channel))
        else:
            print('Series already loaded from {}:{}, channel {}'.format(data_directory, image_file_name, channel))

    def getRoiImage(self, data_directory, image_file_name, series_number, channel, z_slice):
        if image_file_name != self.current_image_file_name or channel != self.current_channel:
            self.current_image_file_name = image_file_name
            self.current_channel = channel
            self.loadImageSeries(data_directory=data_directory,
                                 image_file_name=image_file_name,
                                 channel=channel)
        else:
            pass  # don't need to re-load the entire series

        if self.current_series is None:  # No image file found
            roi_image = []
        else:
            roi_image = self.mean_brain[:, :, int(z_slice)]

        return roi_image

    def getRoiDataFromPath(self, roi_path):
        """
        Compute roi response from roi path objects.

        param:
            roi_path: list of path objects

        *Must first define self.current_series
        """
        mask = self.getRoiMaskFromPath(roi_path)

        roi_response = np.mean(self.current_series[mask, :], axis=0, keepdims=True)

        return roi_response

    def getRoiMaskFromPath(self, roi_path):
        """
        Compute roi mask from roi path objects.

        param:
            roi_path: list of path objects

        *Must first define self.current_series
        """
        x_dim, y_dim, z_dim, t_dim = self.current_series.shape

        pixX = np.arange(y_dim)
        pixY = np.arange(x_dim)
        xv, yv = np.meshgrid(pixX, pixY)
        roi_pix = np.vstack((xv.flatten(), yv.flatten())).T

        mask = np.zeros(shape=(x_dim, y_dim, z_dim))

        for path in roi_path:
            z_level = path.z_level
            xy_indices = np.reshape(path.contains_points(roi_pix, radius=0.5), (x_dim, y_dim))
            mask[xy_indices, z_level] = 1

        mask = mask == 1  # convert to boolean for masking

        return mask

    def attachData(self, experiment_file_name, file_path, data_directory):
        for series_number in self.getSeriesNumbers(file_path):
            # # # # Retrieve metadata from files in data directory # # #
            file_basename = 'TSeries-' + experiment_file_name.replace('-', '') + '-' + ('00' + str(series_number))[-3:]
            metadata_filepath = os.path.join(data_directory, file_basename)
            if os.path.exists(metadata_filepath + '.xml'):
                # Photodiode trace
                v_rec_suffix = '_Cycle00001_VoltageRecording_001'
                voltage_basepath = os.path.join(data_directory, file_basename + v_rec_suffix)
                voltage_recording, time_vector, sample_rate = getVoltageRecording(voltage_basepath)

                # TODO: pick frame monitor(s) out of voltage recording traces based on name, or alt by input number
                frame_monitor = voltage_recording

                # Metadata & timing information
                response_timing = getAcquisitionTiming(metadata_filepath)
                metadata = getMetaData(metadata_filepath)

                # # # # Attach metadata to epoch run group in data file # # #\
                with h5py.File(file_path, 'r+') as experiment_file:
                    find_partial = functools.partial(h5io.find_series, sn=series_number)
                    epoch_run_group = experiment_file.visititems(find_partial)

                    # make sure subgroups exist for stimulus and response timing
                    stimulus_timing_group = epoch_run_group.require_group('stimulus_timing')
                    h5io.overwriteDataSet(stimulus_timing_group, 'frame_monitor', frame_monitor)
                    h5io.overwriteDataSet(stimulus_timing_group, 'time_vector', time_vector)
                    stimulus_timing_group.attrs['sample_rate'] = sample_rate

                    acquisition_group = epoch_run_group.require_group('acquisition')
                    h5io.overwriteDataSet(acquisition_group, 'time_points', response_timing['stack_times'])
                    if 'frame_times' in response_timing:
                        h5io.overwriteDataSet(acquisition_group, 'frame_times', response_timing['frame_times'])
                    acquisition_group.attrs['sample_period'] = response_timing['sample_period']
                    for key in metadata:
                        acquisition_group.attrs[key] = metadata[key]

                print('Attached data to series {}'.format(series_number))
            else:
                print('WARNING! Required metadata files not found at {}'.format(metadata_filepath))

    def loadImageSeries(self, data_directory, image_file_name, channel=0):
        metadata_image_dims = self.ImagingDataObject.getAcquisitionMetadata().get('image_dims')  # xyztc
        image_file_path = os.path.join(data_directory, image_file_name)
        if '.tif' in image_file_name:  # tif assumed to be tyx series
            # native axes order is tyx: convert to xyzt, with z dummy axis
            self.current_series = io.imread(image_file_path)
            self.current_series = np.swapaxes(self.current_series, 0, 2)[:, :, np.newaxis, :]  # -> xyzt
            print('Loaded xyt image series {}'.format(image_file_path))
        elif '.nii' in image_file_name:
            nib_brain = np.squeeze(np.asanyarray(nib.load(image_file_path).dataobj).astype('uint16'))
            brain_dims = nib_brain.shape
            print('brain_dims = {}'.format(brain_dims))
            if len(brain_dims) == 3:  # xyt
                self.current_series = nib_brain[:, :, np.newaxis, :]  # -> xyzt
                print('Loaded xyt image series {}'.format(image_file_path))

            elif len(brain_dims) == 4:  # xyzt or xytc
                if brain_dims[-1] == metadata_image_dims[-1]:  # xytc
                    self.current_series = np.squeeze(nib_brain[:, :, :, channel])[:, :, np.newaxis, :]  # xytc -> xyzt
                    print('Loaded xytc image series {}'.format(image_file_path))
                else:  # xyzt
                    self.current_series = nib_brain  # xyzt
                    print('Loaded xyzt image series {}'.format(image_file_path))

            elif len(brain_dims) == 5:  # xyztc
                self.current_series = np.squeeze(nib_brain[:, :, :, :, channel])  # xyzt
                print('Loaded xyzt image series from xyztc {}: channel {}'.format(image_file_path, channel))

            else:
                print('Unrecognized image dimensions')
                self.current_series = None
        else:
            print('Unrecognized image format. Expects .tif or .nii')

        self.mean_brain = np.mean(self.current_series, axis=3)  # xyz

        print('Brain shape is {} (xyzt)'.format(self.current_series.shape))

    def saveRegionResponsesFromMask(self, file_path, series_number, response_set_name, mask, include_zero=False):
        """
        Save region responses from a mask to the data file

        args
            file_path: string, full path to hdf5 data file
            series_number: int, series in hdf5 data file
            channel: int, which pmt/channel to load
        """
        mask_values = np.unique(mask)
        if include_zero:
            pass
        else:  # Don't compute region responses for pixels where mask == 0
            mask_values = mask_values[mask_values != 0]

        region_responses = [np.mean(self.current_series[mask == label, :], axis=0) for label in mask_values]
        region_responses = np.vstack(region_responses)  # mask ID x Time

        with h5py.File(file_path, 'r+') as experiment_file:
            find_partial = functools.partial(h5io.find_series, sn=series_number)
            epoch_run_group = experiment_file.visititems(find_partial)
            parent_roi_group = epoch_run_group.require_group('aligned')
            current_roi_group = parent_roi_group.require_group(response_set_name)

            h5io.overwriteDataSet(current_roi_group, 'roi_mask', mask)
            h5io.overwriteDataSet(current_roi_group, 'roi_response', region_responses)
            h5io.overwriteDataSet(current_roi_group, 'roi_image', self.mean_brain)

    # %%
    ###########################################################################
    # Functions for timing and metadata
    #   Accessible outside of the plugin object
    ###########################################################################


def getVoltageRecording(filepath):
    """

    params:
        :filepath: path to voltage recording file, with no suffix
    """

    metadata = ET.parse(filepath + '.xml')
    root = metadata.getroot()
    rate_node = root.find('Experiment').find('Rate')
    sample_rate = int(rate_node.text)

    active_channels = []
    signal_list = list(root.find('Experiment').find('SignalList'))
    for signal_node in signal_list:
        is_channel_active = signal_node.find('Enabled').text
        channel_name = signal_node.find('Name').text
        if is_channel_active == 'true':
            active_channels.append(channel_name)

    # Load frame tracker signal and pull frame/epoch timing info
    data_frame = pd.read_csv(filepath + '.csv')

    time_vector = data_frame.get('Time(ms)').values / 1e3  # ->sec

    frame_monitor = []  # get responses in all active channels
    for ac in active_channels:
        frame_monitor.append(data_frame.get(' ' + ac).values)
    frame_monitor = np.vstack(frame_monitor)

    return frame_monitor, time_vector, sample_rate


def getMetaData(filepath):
    """

    params:
        filepath: path to photodiode file(s), with no suffix

    returns
        metadata: dict

    """
    metaData = ET.parse(filepath + '.xml')
    root = metaData.getroot()

    metadata = {}
    for child in list(root.find('PVStateShard')):
        if child.get('value') is None:
            for subchild in list(child):
                if subchild.get('value') is None:
                    for subsubchild in list(subchild):
                        new_key = child.get('key') + '_' + subchild.get('index') + subsubchild.get('subindex')
                        new_value = subsubchild.get('value')
                        metadata[new_key] = new_value

                else:
                    new_key = child.get('key') + '_' + subchild.get('index')
                    new_value = subchild.get('value')
                    metadata[new_key] = new_value

        else:
            new_key = child.get('key')
            new_value = child.get('value')
            metadata[new_key] = new_value

    # Get axis dims
    sequences = root.findall('Sequence')
    c_dim = len(sequences[0].findall('Frame')[0].findall('File'))  # number of channels
    x_dim = metadata['pixelsPerLine']
    y_dim = metadata['linesPerFrame']

    if root.find('Sequence').get('type') == 'TSeries Timed Element':  # Plane time series
        t_dim = len(sequences[0].findall('Frame'))
        z_dim = 1
    elif root.find('Sequence').get('type') == 'TSeries ZSeries Element':  # Volume time series
        t_dim = len(sequences)
        z_dim = len(sequences[0].findall('Frame'))
    elif root.find('Sequence').get('type') == 'ZSeries':  # Single Z stack (anatomical)
        t_dim = 1
        z_dim = len(sequences[0].findall('Frame'))
    else:
        print('!Unrecognized series type in PV metadata!')

    metadata['image_dims'] = [int(x_dim), int(y_dim), z_dim, t_dim, c_dim]

    metadata['version'] = root.get('version')
    metadata['date'] = root.get('date')
    metadata['notes'] = root.get('notes')

    return metadata


def getAcquisitionTiming(filepath):
    """
    Imaging acquisition metadata based on the bruker metadata file (xml)

    params:
        filepath: path to photodiode file(s), with no suffix

    returns
        response_timing: dict

    """
    metaData = ET.parse(filepath + '.xml')
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


def getMarkPointsMetadata(filepath):
    """
    Parse Bruker / PrairieView markpoints metadata from .xml file.

    params:
        filepath: path to photodiode file(s), with no suffix

    returns
        metadata: dict
    """
    metadata = {}

    root = ET.parse(filepath + '.xml').getroot()
    for key in root.keys():
        metadata[key] = root.get(key)

    point_element = root.find('PVMarkPointElement')
    for key in point_element.keys():
        metadata[key] = point_element.get(key)

    galvo_element = point_element[0]
    for key in galvo_element.keys():
        metadata[key] = galvo_element.get(key)

    points = list(galvo_element)
    for point_ind, point in enumerate(points):
        for key in point.keys():
            metadata['Point_{}_{}'.format(point_ind+1, key)] = point.get(key)

    return metadata
