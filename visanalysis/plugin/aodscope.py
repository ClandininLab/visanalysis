# -*- coding: utf-8 -*-
"""
Created on Fri May 31 09:43:59 2019

@author: mhturner
"""
import os
import xml.etree.ElementTree as ET
import numpy as np
from matplotlib import path
import h5py
import skimage.io as io
import functools
from nptdms import TdmsFile
import configparser

from visanalysis import plugin, plot_tools

##############################################################################
# Functions for random access poi data from AODscope / Karthala
##############################################################################

# TODO: poi picker / grouper
# TODO: special assignment for background pois?
# TODO: PMT 1 or 2 selection


class AodScopePlugin(plugin.base.BasePlugin):
    def __init__(self):
        super().__init__()

    def getRoiImage(self, **kwargs):
        series_number = kwargs.get('series_number')
        file_path = kwargs.get('file_path')
        pmt = kwargs.get('pmt')
        with h5py.File(file_path, 'r') as experiment_file:
            find_partial = functools.partial(find_series, sn=series_number)
            epoch_run_group = experiment_file.visititems(find_partial)
            acquisition_group = epoch_run_group['acquisition']

            snap_image = acquisition_group.get('snap_image')[:]
            poi_locations = acquisition_group.get('poi_locations')[:]

        poi_mask = []
        pixX = np.arange(snap_image.shape[1])
        pixY = np.arange(snap_image.shape[0])
        yv, xv = np.meshgrid(pixX, pixY)
        pix = np.vstack((yv.flatten(), xv.flatten())).T

        for poi_loc in poi_locations:
            center = poi_loc
            new_roi_path = path.Path.circle(center=center, radius=2)
            ind = new_roi_path.contains_points(pix, radius=0.5)

            array = np.zeros(snap_image.shape)
            lin = np.arange(array.size)
            newArray = array.flatten()
            newArray[lin[ind]] = 1
            poi_mask.append(newArray.reshape(array.shape))

        poi_overlay = plot_tools.overlayImage(snap_image, poi_mask, 1.0)

        return poi_overlay

    def getRoiDataFromPath(self, roi_path, data_directory, series_number, experiment_file_name):
        poi_data = self.getPoiData(poi_directory=data_directory,
                                   poi_series_number=series_number,
                                   pmt=1)
        # Imaging metadata
        # TODO: ctl+click to select multiple regions?
        # TODO: poi picker?
        # TODO: number of pois as attribute

        indices = roi_path.contains_points(poi_data['poi_locations'], radius=0.5)
        selected_poi_data = poi_data['poi_data_matrix'][indices, :]
        if selected_poi_data.shape[0] == 0:
            roi_response = None
        else:
            print('Selected {} pois'.format(selected_poi_data.shape[0]))
            roi_response = np.mean(selected_poi_data, axis=0)
        return roi_response

    def attachData(self, experiment_file_name, file_path, data_directory):
        for series_number in self.getSeriesNumbers(file_path):
            # # # # Retrieve metadata from files in data directory # # #
            # Photodiode trace
            frame_monitor, time_vector, sample_rate = self.getPhotodiodeSignal(data_directory,
                                                                               series_number)

            # Poi data
            poi_data = self.getPoiData(data_directory,
                                       series_number,
                                       pmt=1)

            # Imaging metadata
            metadata = self.getMetaData(data_directory,
                                        series_number)

            # # # # Attach metadata to epoch run group in data file # # #
            with h5py.File(file_path, 'r+') as experiment_file:
                find_partial = functools.partial(find_series, sn=series_number)
                epoch_run_group = experiment_file.visititems(find_partial)

                # make sure subgroups exist for stimulus and response timing
                stimulus_timing_group = epoch_run_group.require_group('stimulus_timing')
                plugin.base.overwriteDataSet(stimulus_timing_group, 'frame_monitor', frame_monitor)
                plugin.base.overwriteDataSet(stimulus_timing_group, 'time_vector', time_vector)
                stimulus_timing_group.attrs['sample_rate'] = sample_rate

                acquisition_group = epoch_run_group.require_group('acquisition')
                plugin.base.overwriteDataSet(acquisition_group, 'time_points', poi_data['time_points'])
                plugin.base.overwriteDataSet(acquisition_group, 'poi_data_matrix', poi_data['poi_data_matrix'])
                plugin.base.overwriteDataSet(acquisition_group, 'poi_xy', poi_data['poi_xy'])

                for outer_k in metadata.keys():
                    for inner_k in metadata[outer_k].keys():
                        acquisition_group.attrs[outer_k + '/' + inner_k] = metadata[outer_k][inner_k]

                plugin.base.overwriteDataSet(acquisition_group, "poi_locations", poi_data['poi_locations'])
                plugin.base.overwriteDataSet(acquisition_group, "snap_image", poi_data['snap_image'])

            print('Attached data to series {}'.format(series_number))

    def getPoiData(self, poi_directory, poi_series_number, pmt=1):
        poi_name = 'points' + ('0000' + str(poi_series_number))[-4:]
        full_file_path = os.path.join(poi_directory, 'points', poi_name, poi_name + '_pmt' + str(pmt) + '.tdms')

        try:
            tdms_file = TdmsFile(full_file_path)

            time_points = tdms_file.channel_data('PMT'+str(pmt),'POI time') #msec
            poi_data_matrix = np.ndarray(shape = (len(tdms_file.group_channels('PMT'+str(pmt))[1:]), len(time_points)))
            poi_data_matrix[:] = np.nan

            for poi_ind in range(len(tdms_file.group_channels('PMT'+str(pmt))[1:])): #first object is time points. Subsequent for POIs
                poi_data_matrix[poi_ind, :] = tdms_file.channel_data('PMT'+str(pmt), 'POI ' + str(poi_ind) + ' ')

            # get poi locations in raw coordinates:
            poi_x = [int(v) for v in tdms_file.channel_data('parameter', 'parameter')[21:]]
            poi_y = [int(v) for v in tdms_file.channel_data('parameter', 'value')[21:]]
            poi_xy = np.array(list(zip(poi_x, poi_y)))

            # get Snap image and poi locations in snap coordinates
            metadata = self.getMetaData(poi_directory,
                                        poi_series_number)
            snap_name = metadata['Image']['name'].replace('"', '')
            snap_ct = 0
            while (('points' in snap_name) and (snap_ct < 1000)):  # used snap image from a previous POI scan
                snap_ct += 1
                alt_dict = self.getMetaData(poi_directory, int(snap_name[6:]))
                temp_image = alt_dict.get('Image')
                if temp_image is not None:
                    snap_name = temp_image['name'].replace('"', '')

            snap_image, snap_settings, poi_locations = self.getSnapImage(poi_directory,
                                                                         snap_name,
                                                                         poi_xy,
                                                                         pmt=1)
        except:
            time_points = None
            poi_data_matrix = None
            print('No tdms file found at: ' + full_file_path)

        return {'time_points': time_points,
                'poi_data_matrix': poi_data_matrix,
                'poi_xy': poi_xy,
                'snap_image': snap_image,
                'snap_settings': snap_settings,
                'poi_locations': poi_locations}

    def getPhotodiodeSignal(self, data_directory, series_number):
        poi_name = 'points' + ('0000' + str(series_number))[-4:]
        full_file_path = os.path.join(data_directory, 'points', poi_name, poi_name + '-AnalogIN.tdms')

        if os.path.exists(full_file_path):
            tdms_file = TdmsFile(full_file_path)
            try:
                time_vector = tdms_file.object('external analogIN', 'AnalogGPIOBoard/ai0').time_track()
                frame_monitor = tdms_file.object('external analogIN', 'AnalogGPIOBoard/ai0').data
            except:
                time_vector = None
                frame_monitor = None
                print('Analog input file has unexpected structure: ' + full_file_path)
        else:
            time_vector = None
            frame_monitor = None
            print('No analog_input file found at: ' + full_file_path)

        sample_rate = 1e4  # TODO: figure this out from tdms

        return frame_monitor, time_vector, sample_rate

    def getMetaData(self, data_directory, series_number):
        poi_name = 'points' + ('0000' + str(series_number))[-4:]
        full_file_path = os.path.join(data_directory, 'points', poi_name, poi_name + '.ini')

        config = configparser.ConfigParser()
        config.read(full_file_path)

        metadata = config._sections

        return metadata

    def getSnapImage(self, poi_directory, snap_name, poi_xy, pmt=1):
        full_file_path = os.path.join(poi_directory, 'snap', snap_name, snap_name[9:] + '_' + snap_name[:8] + '-snap-' + 'pmt'+str(pmt) + '.tif')
        if os.path.exists(full_file_path):
            snap_image = io.imread(full_file_path)

            roi_para_file_path = os.path.join(poi_directory, 'snap', snap_name,
                                              snap_name[9:] + '_' + snap_name[:8] + 'para.roi')
            roi_root = ET.parse(roi_para_file_path).getroot()
            ArrayNode = roi_root.find('{http://www.ni.com/LVData}Cluster/{http://www.ni.com/LVData}Array')
            snap_dims = [int(x.find('{http://www.ni.com/LVData}Val').text) for x in
                         ArrayNode.findall('{http://www.ni.com/LVData}I32')]

            snap_para_file_path = os.path.join(poi_directory, 'snap', snap_name,
                                               snap_name[9:] + '_' + snap_name[:8] + 'para.xml')

            with open(snap_para_file_path) as strfile:
                xmlString = strfile.read()
            french_parser = ET.XMLParser(encoding="ISO-8859-1")
            snap_parameters = ET.fromstring(xmlString, parser=french_parser)

            resolution = [int(float(x.find('{http://www.ni.com/LVData}Val').text)) for x in
                          snap_parameters.findall(".//{http://www.ni.com/LVData}DBL") if
                          x.find('{http://www.ni.com/LVData}Name').text == 'Resolution'][0]
            full_resolution = [int(float(x.find('{http://www.ni.com/LVData}Val').text)) for x in
                               snap_parameters.findall(".//{http://www.ni.com/LVData}DBL") if
                               x.find('{http://www.ni.com/LVData}Name').text == 'Resolution full'][0]

            snap_settings = {'snap_dims': snap_dims, 'resolution': resolution, 'full_resolution': full_resolution}

            # Poi xy locations are in full resolution space. Need to map to snap space
            poi_xy_to_resolution = poi_xy / (snap_settings['full_resolution'] / snap_settings['resolution'])
            poi_locations = (poi_xy_to_resolution - snap_settings['snap_dims'][0:2]).astype(int)

        else:
            snap_image = 0
            snap_settings = []
            poi_locations = []
            print('Warning no snap image found at: ' + full_file_path)

        return snap_image, snap_settings, poi_locations


def find_series(name, obj, sn):
    target_group_name = 'series_{}'.format(str(sn).zfill(3))
    if target_group_name in name:
        return obj
