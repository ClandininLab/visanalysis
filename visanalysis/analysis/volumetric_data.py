from visanalysis.analysis import imaging_data
import numpy as np
from scipy import stats
import nibabel as nib


class VolumetricDataObject(imaging_data.ImagingDataObject):
    def __init__(self, file_path, series_number, quiet=False):
        super().__init__(file_path, series_number, quiet=quiet)

    def getTrialAlignedVoxelResponses(self, voxels, dff=False):
        n_voxels, t_dim = voxels.shape

        # zero values are from registration. Replace with nan
        voxels[np.where(voxels == 0)] = np.nan
        # set to minimum
        voxels[np.where(np.isnan(voxels))] = np.nanmin(voxels)

        stimulus_start_times = self.getStimulusTiming()['stimulus_start_times']  # sec
        pre_time = self.getRunParameters()['pre_time']  # sec
        stim_time = self.getRunParameters()['stim_time']  # sec
        tail_time = self.getRunParameters()['tail_time']  # sec
        epoch_start_times = stimulus_start_times - pre_time
        epoch_end_times = stimulus_start_times + stim_time + tail_time
        epoch_time = pre_time + stim_time + tail_time # sec

        sample_period = self.getResponseTiming()['sample_period']  # sec
        stack_times = self.getResponseTiming()['time_vector']  # sec

        # find how many acquisition frames correspond to pre, stim, tail time
        epoch_frames = int(epoch_time / sample_period)  # in acquisition frames
        pre_frames = int(pre_time / sample_period)  # in acquisition frames
        tail_frames = int(tail_time / sample_period)
        time_vector = np.arange(0, epoch_frames) * sample_period  # sec

        no_trials = len(epoch_start_times)
        voxel_trial_matrix = np.ndarray(shape=(n_voxels, epoch_frames, no_trials), dtype='float32') #n_voxels, time_vector, trials
        voxel_trial_matrix[:] = np.nan
        cut_inds = np.empty(0, dtype=int)
        for idx, val in enumerate(epoch_start_times):
            stack_inds = np.where(np.logical_and(stack_times < epoch_end_times[idx], stack_times >= epoch_start_times[idx]))[0]
            if len(stack_inds) == 0:  # no imaging acquisitions happened during this epoch presentation
                cut_inds = np.append(cut_inds, idx)
                continue
            if np.any(stack_inds > voxels.shape[1]):
                cut_inds = np.append(cut_inds, idx)
                continue
            if idx == no_trials:
                if len(stack_inds) < epoch_frames:  # missed images for the end of the stimulus
                    cut_inds = np.append(cut_inds, idx)
                    continue

            # Get voxel responses for this epoch
            new_resp_chunk = voxels[:, stack_inds]  # voxel X time

            if dff:
                # calculate baseline using pre frames and last half of tail frames
                baseline_pre = new_resp_chunk[:, 0:pre_frames]
                baseline_tail = new_resp_chunk[:, -int(tail_frames/2):]
                baseline = np.mean(np.concatenate((baseline_pre, baseline_tail), axis=1), axis=1, keepdims=True)
                # to dF/F
                new_resp_chunk = (new_resp_chunk - baseline) / baseline;

            try:
                voxel_trial_matrix[:, :, idx] = new_resp_chunk[:, 0:epoch_frames]
            except:
                print('Size mismatch idx = {}'.format(idx)) # the end of a response clipped off
                cut_inds = np.append(cut_inds, idx)

        voxel_trial_matrix = np.delete(voxel_trial_matrix, cut_inds, axis=2) # shape = (voxel, time, trial)

        return time_vector, voxel_trial_matrix

    def getMeanBrainByStimulus(self, voxel_trial_matrix, parameter_key=None):
        run_parameters = self.getRunParameters()
        response_timing = self.getResponseTiming()
        epoch_parameters = self.getEpochParameters()

        if parameter_key is None:
            parameter_values = [list(pd.values()) for pd in self.getEpochParameterDicts()]
        elif type(parameter_key) is dict: #for composite stims like panglom suite
            parameter_values = []
            for ind_e, ep in enumerate(epoch_parameters):
                component_stim_type = ep.get('component_stim_type')
                e_params = [component_stim_type]
                param_keys = parameter_key[component_stim_type]
                for pk in param_keys:
                    e_params.append(ep.get(pk))

                parameter_values.append(e_params)
        else:
            parameter_values = [ep.get(parameter_key) for ep in epoch_parameters]

        unique_parameter_values = np.unique(parameter_values)
        n_stimuli = len(unique_parameter_values)

        pre_frames = int(run_parameters['pre_time'] / response_timing.get('sample_period'))
        stim_frames = int(run_parameters['stim_time'] / response_timing.get('sample_period'))
        tail_frames = int(run_parameters['tail_time'] / response_timing.get('sample_period'))

        n_voxels, t_dim, trials = voxel_trial_matrix.shape

        mean_voxel_response = np.ndarray(shape=(n_voxels, t_dim, n_stimuli)) # voxels x time x stim condition
        p_values = np.ndarray(shape=(n_voxels, n_stimuli))
        response_amp = np.ndarray(shape=(n_voxels, n_stimuli)) # mean voxel resp for each stim condition (voxel x stim)
        trial_response_amp = [] # list (len=n_stimuli), each entry is ndarray of mean response amplitudes (voxels x trials)
        trial_response_by_stimulus = [] # list (len=n_stimuli), each entry is ndarray of trial response (voxel x time x trial)

        for p_ind, up in enumerate(unique_parameter_values):
            pull_inds = np.where([up == x for x in parameter_values])[0]

            if np.any(pull_inds >= voxel_trial_matrix.shape[2]):
                tmp = np.where(pull_inds >= voxel_trial_matrix.shape[2])[0]
                print('Epoch(s) {} not included in voxel_trial_matrix'.format(pull_inds[tmp]))
                pull_inds = pull_inds[pull_inds < voxel_trial_matrix.shape[2]]

            baseline_pts = np.concatenate((voxel_trial_matrix[:, 0:pre_frames, pull_inds],
                                           voxel_trial_matrix[:, -int(tail_frames/2):, pull_inds]), axis=1)
            response_pts = voxel_trial_matrix[:, pre_frames:(pre_frames+stim_frames), pull_inds]

            _, p_values[:, p_ind] = stats.ttest_ind(np.reshape(baseline_pts, (n_voxels, -1)),
                                                    np.reshape(response_pts, (n_voxels, -1)), axis=1)

            trial_response_amp.append(np.nanmean(response_pts, axis=1))  # each list entry = timee average. (voxels x trials)

            response_amp[:, p_ind] = np.mean(response_pts, axis=(1, 2))

            mean_voxel_response[:, :, p_ind] = (np.mean(voxel_trial_matrix[:, :, pull_inds], axis=2))
            trial_response_by_stimulus.append(voxel_trial_matrix[:, :, pull_inds])

        return mean_voxel_response, unique_parameter_values, p_values, response_amp, trial_response_amp, trial_response_by_stimulus


def loadFunctionalBrain(file_path, x_lim=[0, None], y_lim=[0, None], z_lim=[0, None], t_lim=[0, None], channel=1):
    brain = nib.load(file_path).get_fdata()
    if len(brain.shape) > 4:  # multi-channel xyztc
        brain = brain[x_lim[0]:x_lim[1], y_lim[0]:y_lim[1], z_lim[0]:z_lim[1], t_lim[0]:t_lim[1], channel]
        # print('Loaded channel {} of xyztc brain {}'.format(channel, file_path))
    else:  # single channel xyzt
        brain = brain[x_lim[0]:x_lim[1], y_lim[0]:y_lim[1], z_lim[0]:z_lim[1], t_lim[0]:t_lim[1]]
        # print('Loaded single channel xyzt brain {}'.format(file_path))

    return brain
