"""
Shared analysis tools for visanalysis

https://github.com/ClandininLab/visanalysis
mhturner@stanford.edu
"""
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
from visanalysis import plot_tools


def getTraceMatrixByStimulusParameter(response_matrix, parameter_values):
    """
    parameter values is nTrials x nParams  numpy array
    returns:
      uniqueParameterValues (nConditions x nParams)
      meanTraceMatrix and semTraceMatrix (nRois x nConditions, time)
    """
    unique_parameter_values = np.unique(parameter_values, axis=0)

    no_rois = response_matrix.shape[0]
    epoch_len = response_matrix.shape[2]
    no_conditions = len(unique_parameter_values)

    mean_trace_matrix = np.empty(shape=(no_rois, no_conditions, epoch_len), dtype=float)
    mean_trace_matrix[:] = np.nan

    sem_trace_matrix = np.empty(shape=(no_rois, no_conditions, epoch_len), dtype=float)
    sem_trace_matrix[:] = np.nan

    individual_traces = []
    for vInd, V in enumerate(unique_parameter_values):
        pull_inds = np.where((parameter_values == V).all(axis=1))[0]
        current_responses = response_matrix[:, pull_inds, :]
        mean_trace_matrix[:, vInd, :] = np.mean(current_responses, axis=1)
        sem_trace_matrix[:, vInd, :] = np.std(current_responses, axis=1) / np.sqrt(len(pull_inds))
        individual_traces.append(current_responses)

    return unique_parameter_values, mean_trace_matrix, sem_trace_matrix, individual_traces


def plotResponseByCondition(ImagingData, roi_name, condition, eg_ind=0):
    roi_data = ImagingData.getRoiResponses(roi_name)

    conditioned_param = []
    for ep in ImagingData.getEpochParameters():
        conditioned_param.append(ep[condition])

    parameter_values = np.array([conditioned_param]).T
    unique_parameter_values, mean_trace_matrix, sem_trace_matrix, individual_traces = getTraceMatrixByStimulusParameter(roi_data.get('epoch_response'), parameter_values)

    unique_params = np.unique(np.array(conditioned_param))

    # plot stuff
    fig_handle = plt.figure(figsize=(10,2))
    grid = plt.GridSpec(1, len(unique_params), wspace=0.2, hspace=0.15)
    fig_axes = fig_handle.get_axes()
    ax_ind = 0

    plot_y_max = np.max((1.1*np.max(mean_trace_matrix), 3))
    plot_y_min = -1

    for ind_v, val in enumerate(unique_params):
        pull_ind = np.where((unique_parameter_values == val).all(axis = 1))[0][0]
        ax_ind += 1
        if len(fig_axes) > 1:
            new_ax = fig_axes[ax_ind]
            new_ax.clear()
        else:
            new_ax = fig_handle.add_subplot(grid[0,ind_v])

        new_ax.plot(roi_data.get('time_vector'), mean_trace_matrix[eg_ind,pull_ind,:].T)
        new_ax.set_ylim([plot_y_min, plot_y_max])
        new_ax.set_axis_off()
        new_ax.set_title(val)
        if ind_v == 0:  # scale bar
            plot_tools.addScaleBars(new_ax, 1, 1, F_value = -1, T_value = -0.2)
    fig_handle.canvas.draw()


def plotRoiResponses(ImagingData, roi_name):
    plot_y_min = -0.5
    plot_y_max = 2

    roi_data = ImagingData.getRoiResponses(roi_name)

    fh, ax = plt.subplots(1, int(roi_data.get('epoch_response').shape[0]+1), figsize=(6, 2))

    for r_ind in range(roi_data.get('epoch_response').shape[0]):
        time_vector = roi_data.get('time_vector')
        no_trials = roi_data.get('epoch_response')[r_ind, :, :].shape[0]
        current_mean = np.mean(roi_data.get('epoch_response')[r_ind,:,:], axis=0)
        current_std = np.std(roi_data.get('epoch_response')[r_ind,:,:], axis=0)
        current_sem = current_std / np.sqrt(no_trials)

        ax[r_ind].plot(time_vector, current_mean, 'k')
        ax[r_ind].fill_between(time_vector,
                            current_mean - current_sem,
                            current_mean + current_sem,
                            alpha=0.5)
        ax[r_ind].set_ylim([plot_y_min, plot_y_max])
        ax[r_ind].set_axis_off()
        ax[r_ind].set_title(int(r_ind))

        if r_ind == 0: # scale bar
            plot_tools.addScaleBars(ax[r_ind], 1, 1, F_value = -0.1, T_value = -0.2)

    fh.canvas.draw()

def filterDataFiles(data_directory, target_fly_metadata={}, target_series_metadata={}):
    """
    Searches through a directory of visprotocol datafiles and finds datafiles/series that match the search values
    Can search based on any number of fly metadata params or run parameters

    Params
        -data_directory: directory of visprotocol data files to search through
        -target_fly_metadata: (dict) key-value pairs of target parameters to search for in the fly metadata
        -target_series_metadata: (dict) key-value pairs of target parameters to search for in the series run (run parameters)

    Returns
        -matching_series: List of matching series dicts with all fly & run params as well as file name and series number
    """
    fileNames = glob.glob(data_directory + "/*.hdf5")
    print('Found {} files in {}'.format(len(fileNames), data_directory))
    # print(fileNames)

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
                    all_series.append(new_series)

    # search in all series for target key/value pairs
    match_dict = {**target_fly_metadata, **target_series_metadata}
    matching_series = []
    for series in all_series:
        if all([series[key] == match_dict[key] for key in match_dict]):
            matching_series.append(series)

    return matching_series
