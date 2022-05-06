"""
Shared analysis tools.

https://github.com/ClandininLab/visanalysis
mhturner@stanford.edu
"""
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
from visanalysis.util import plot_tools
from collections import Sequence


def matchQuery(epoch_parameters, query):
    """

    params:
        epoch_parameters: single epoch_parameter dict
        query: dict, key-value pairs indicate matching parameter editable_values
            e.g. query = {'current_intensity': 0.75}

    Returns:
        Bool, True if all param values match, false otherwise
    """
    return np.all([epoch_parameters.get(key) == query[key] for key in query])


def filterTrials(epoch_response, ID, query, return_inds=False):
    matching_trials = np.where([matchQuery(ep, query) for ep in ID.getEpochParameters()])[0]

    if return_inds:
        return epoch_response[:, matching_trials, :], matching_trials
    else:
        return epoch_response[:, matching_trials, :]


def getUniqueParameterCombinations(param_keys, ID):
    ep_params = [[ep.get(x, None) for x in param_keys]for ep in ID.getEpochParameters()]
    return list({tuple(row) for row in ep_params})


def plotResponseByCondition(ImagingData, roi_name, condition, eg_ind=0):
    roi_data = ImagingData.getRoiResponses(roi_name)

    unique_parameter_values = np.unique([ep.get(condition) for ep in ImagingData.getEpochParameters()])
    fh, ax = plt.subplots(1, len(unique_parameter_values), figsize=(8, 2))
    [x.set_axis_off() for x in ax]
    [x.set_ylim([-0.25, 1.0]) for x in ax]
    for p_ind, param_value in enumerate(unique_parameter_values):
        query = {condition: param_value}
        trials = filterTrials(roi_data.get('epoch_response'), ImagingData, query)
        ax[p_ind].plot(roi_data.get('time_vector'), np.mean(trials[eg_ind, :, :], axis=0), linestyle='-', color=ImagingData.colors[0])

        if p_ind == 0:  # scale bar
            plot_tools.addScaleBars(ax[p_ind], dT=1, dF=0.5, F_value=-0.25, T_value=-0.2)


def plotRoiResponses(ImagingData, roi_name):
    roi_data = ImagingData.getRoiResponses(roi_name)

    fh, ax = plt.subplots(1, int(roi_data.get('epoch_response').shape[0]+1), figsize=(6, 2))
    [x.set_axis_off() for x in ax]
    [x.set_ylim([-0.25, 1]) for x in ax]

    for r_ind in range(roi_data.get('epoch_response').shape[0]):
        time_vector = roi_data.get('time_vector')
        no_trials = roi_data.get('epoch_response')[r_ind, :, :].shape[0]
        current_mean = np.mean(roi_data.get('epoch_response')[r_ind, :, :], axis=0)
        current_std = np.std(roi_data.get('epoch_response')[r_ind, :, :], axis=0)
        current_sem = current_std / np.sqrt(no_trials)

        ax[r_ind].plot(time_vector, current_mean, 'k')
        ax[r_ind].fill_between(time_vector,
                               current_mean - current_sem,
                               current_mean + current_sem,
                               alpha=0.5)
        ax[r_ind].set_title(int(r_ind))

        if r_ind == 0:  # scale bar
            plot_tools.addScaleBars(ax[r_ind], 1, 1, F_value=-0.1, T_value=-0.2)


def filterDataFiles(data_directory,
                    target_fly_metadata={},
                    target_series_metadata={},
                    target_roi_series=[],
                    target_groups=[],
                    quiet=False):
    """
    Searches through a directory of visprotocol datafiles and finds datafiles/series that match the search values
    Can search based on any number of fly metadata params or run parameters

    Params
        -data_directory: directory of visprotocol data files to search through
        -target_fly_metadata: (dict) key-value pairs of target parameters to search for in the fly metadata
        -target_series_metadata: (dict) key-value pairs of target parameters to search for in the series run (run parameters)
        -target_roi_series: (list) required roi_series names
        -target_groups: (list) required names of groups under series group

    Returns
        -matching_series: List of matching series dicts with all fly & run params as well as file name and series number
    """
    fileNames = glob.glob(data_directory + "/*.hdf5")
    if not quiet:
        print('Found {} files in {}'.format(len(fileNames), data_directory))

    # collect key/value pairs for all series in data directory
    all_series = []
    for ind, fn in enumerate(fileNames):

        with h5py.File(fn, 'r') as data_file:
            for fly in data_file.get('Flies'):
                fly_metadata = {}
                for f_key in data_file.get('Flies').get(fly).attrs.keys():
                    fly_metadata[f_key] = data_file.get('Flies').get(fly).attrs[f_key]

                for epoch_run in data_file.get('Flies').get(fly).get('epoch_runs'):
                    series_metadata = {}
                    for s_key in data_file.get('Flies').get(fly).get('epoch_runs').get(epoch_run).attrs.keys():
                        series_metadata[s_key] = data_file.get('Flies').get(fly).get('epoch_runs').get(epoch_run).attrs[s_key]

                    new_series = {**fly_metadata, **series_metadata}
                    new_series['series'] = int(epoch_run.split('_')[1])
                    new_series['file_name'] = fn.split('\\')[-1].split('.')[0]

                    existing_roi_sets = list(data_file.get('Flies').get(fly).get('epoch_runs').get(epoch_run).get('rois').keys())
                    new_series['rois'] = existing_roi_sets
                    existing_groups = list(data_file.get('Flies').get(fly).get('epoch_runs').get(epoch_run).keys())
                    new_series['groups'] = existing_groups

                    all_series.append(new_series)

    # search in all series for target key/value pairs
    match_dict = {**target_fly_metadata, **target_series_metadata}
    matching_series = []
    for series in all_series:
        if checkAgainstTargetDict(match_dict, series):
            if np.all([r in series.get('rois') for r in target_roi_series]):
                if np.all([r in series.get('groups') for r in target_groups]):
                    matching_series.append(series)

    matching_series = sorted(matching_series, key=lambda d: d['file_name'] + '-' + str(d['series']).zfill(3))
    if not quiet:
        print('Found {} matching series'.format(len(matching_series)))
    return matching_series


def checkAgainstTargetDict(target_dict, test_dict):
    for key in target_dict:
        if key in test_dict:
            if not areValsTheSame(target_dict[key], test_dict[key]):
                return False  # Different values
        else:
            return False  # Target key not in this series at all

    return True


def areValsTheSame(target_val, test_val):

    if isinstance(target_val, str):
        return target_val.casefold() == test_val.casefold()
    elif isinstance(target_val, bool):
        if isinstance(test_val, str):
            return str(target_val).casefold() == test_val.casefold()

        return target_val == test_val

    elif isinstance(target_val, (int, float)):  # Scalar
        if isinstance(test_val, (int, float)):
            return float(target_val) == float(test_val)  # Ignore type for int vs. float here
        else:
            return False
    elif isinstance(target_val, (Sequence, np.ndarray)):  # Note already excluded possibility of string by if ordering
        if isinstance(test_val, (Sequence, np.ndarray)):
            # Ignore order of arrays, and ignore float vs. int
            return np.all(np.sort(target_val, axis=0).astype(float) == np.sort(test_val, axis=0).astype(float))
        else:
            return False

    else:
        print('----')
        print('Unable to match ')
        print('Target {} ({})'.format(target_val, type(target_val)))
        print('Test {} ({})'.format(test_val, type(test_val)))
        print('----')
        return False
