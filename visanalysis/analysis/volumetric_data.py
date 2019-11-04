from visanalysis.analysis import imaging_data
import numpy as np
from scipy import stats


class VolumetricDataObject(imaging_data.ImagingDataObject):
    def __init__(self, experiment_file_directory, experiment_file_name, series_number):
        super().__init__(experiment_file_directory, experiment_file_name, series_number)

    def getTrialAlignedVoxelResponses(self, brain, dff=False):
        # brain is shape (x,y,z,t)
        x_dim, y_dim, z_dim, t_dim = brain.shape

        # zero values are from registration. Replace with nan
        brain[np.where(brain == 0)] = np.nan
        # set to minimum
        brain[np.where(np.isnan(brain))] = np.nanmin(brain)

        stimulus_start_times = self.stimulus_timing['stimulus_start_times']  # sec
        stimulus_end_times = self.stimulus_timing['stimulus_end_times']  # sec
        pre_time = self.run_parameters['pre_time']  # sec

        tail_time = self.run_parameters['tail_time']  # sec
        epoch_start_times = stimulus_start_times - pre_time
        epoch_end_times = stimulus_end_times + tail_time

        sample_period = self.response_timing['sample_period']  # sec
        stack_times = self.response_timing['time_vector']  # sec

        # Use measured stimulus lengths for stim time instead of epoch param
        # cut off a bit of the end of each epoch to allow for slop in how many frames were acquired
        epoch_time = 0.99 * np.mean(epoch_end_times - epoch_start_times)  # sec

        # find how many acquisition frames correspond to pre, stim, tail time
        epoch_frames = int(epoch_time / sample_period)  # in acquisition frames
        pre_frames = int(pre_time / sample_period)  # in acquisition frames
        tail_frames = int(tail_time / sample_period)
        time_vector = np.arange(0, epoch_frames) * sample_period  # sec

        no_trials = len(epoch_start_times)
        brain_trial_matrix = np.ndarray(shape=(x_dim, y_dim, z_dim, epoch_frames, no_trials), dtype='float32') #x, y, z, trials, time_vector
        brain_trial_matrix[:] = np.nan
        cut_inds = np.empty(0, dtype=int)
        for idx, val in enumerate(epoch_start_times):
            stack_inds = np.where(np.logical_and(stack_times < epoch_end_times[idx], stack_times >= epoch_start_times[idx]))[0]
            if len(stack_inds) == 0:  # no imaging acquisitions happened during this epoch presentation
                cut_inds = np.append(cut_inds, idx)
                continue
            if np.any(stack_inds > brain.shape[3]):
                cut_inds = np.append(cut_inds, idx)
                continue
            if idx is not 0:
                if len(stack_inds) < epoch_frames:  # missed images for the end of the stimulus
                    cut_inds = np.append(cut_inds, idx)
                    continue

            # Get voxel responses for this epoch
            new_resp_chunk = brain[:, :, :, stack_inds]  # xyzt

            if dff:
                # calculate baseline using pre frames and last half of tail frames
                baseline_pre = new_resp_chunk[:, :, :, 0:pre_frames]
                baseline_tail = new_resp_chunk[:, :, :, -int(tail_frames/2):]
                baseline = np.mean(np.concatenate((baseline_pre, baseline_tail), axis=3), axis=3, keepdims=True)
                # to dF/F
                new_resp_chunk = (new_resp_chunk - baseline) / baseline;

            brain_trial_matrix[:, :, :, :, idx] = new_resp_chunk[:, :, :, 0:epoch_frames]

        brain_trial_matrix = np.delete(brain_trial_matrix, cut_inds, axis=4)

        return time_vector, brain_trial_matrix

    def getMeanBrainByStimulus(self, brain_trial_matrix, parameter_keys):
        parameter_values = []
        for ind_e, ep in enumerate(self.epoch_parameters):
            component_stim_type = ep.get('component_stim_type')
            e_params = [component_stim_type]
            param_keys = parameter_keys[component_stim_type]
            for pk in param_keys:
                e_params.append(ep.get(pk))

            parameter_values.append(e_params)
        unique_parameter_values = np.unique(parameter_values)

        n_stimuli = len(unique_parameter_values)

        pre_frames = int(self.run_parameters['pre_time'] / self.response_timing.get('sample_period'))
        stim_frames = int(self.run_parameters['stim_time'] / self.response_timing.get('sample_period'))
        tail_frames = int(self.run_parameters['tail_time'] / self.response_timing.get('sample_period'))

        x_dim = brain_trial_matrix.shape[0]
        y_dim = brain_trial_matrix.shape[1]
        z_dim = brain_trial_matrix.shape[2]
        t_dim = brain_trial_matrix.shape[3]

        mean_brain_response = np.ndarray(shape=(x_dim, y_dim, z_dim, t_dim, n_stimuli))
        p_values = np.ndarray(shape=(x_dim, y_dim, z_dim, n_stimuli))
        for p_ind, up in enumerate(unique_parameter_values):
            pull_inds = np.where([up == x for x in parameter_values])[0]

            baseline_pts = np.concatenate((brain_trial_matrix[:, :, :, 0:pre_frames, pull_inds],
                                           brain_trial_matrix[:, :, :, -int(tail_frames/2):, pull_inds]), axis=3)
            response_pts = brain_trial_matrix[:, :, :, pre_frames:(pre_frames+stim_frames), pull_inds]

            _, p_values[:, :, :, p_ind] = stats.ttest_ind(np.reshape(baseline_pts, (x_dim, y_dim, z_dim, -1)),
                                                          np.reshape(response_pts, (x_dim, y_dim, z_dim, -1)), axis=3)

            mean_brain_response[:, :, :, :, p_ind] = (np.mean(brain_trial_matrix[:, :, :, :, pull_inds], axis=4))

        return mean_brain_response, unique_parameter_values, p_values

#     def getConcatenatedMeanVoxelResponses(self, brain_trial_matrix, parameter_keys):
#         parameter_values = np.ndarray((len(self.epoch_parameters), len(parameter_keys)))
#         for ind_e, ep in enumerate(self.epoch_parameters):
#             for ind_k, k in enumerate(parameter_keys):
#                 new_val = ep.get(k)
#                 parameter_values[ind_e, ind_k] = new_val
#
#         unique_parameter_values = np.unique(parameter_values, axis=0)
#
#         pre_frames = int(self.run_parameters['pre_time'] / self.response_timing.get('sample_period'))
#         stim_frames = int(self.run_parameters['stim_time'] / self.response_timing.get('sample_period'))
#         tail_frames = int(self.run_parameters['tail_time'] / self.response_timing.get('sample_period'))
#
#         x_dim = brain_trial_matrix.shape[0]
#         y_dim = brain_trial_matrix.shape[1]
#         z_dim = brain_trial_matrix.shape[2]
#
#         mean_resp = []
#         std_resp = []
#         p_values = []
#         for up in unique_parameter_values:
#             pull_inds = np.where((up == parameter_values).all(axis=1))[0]
#
#             # get baseline timepoints for each voxel
#             baseline_pre = brain_trial_matrix[:, :, :, pull_inds, 0:pre_frames]
#             baseline_tail = brain_trial_matrix[:, :, :, pull_inds, -int(tail_frames/2):]
#             baseline_points = np.concatenate((baseline_pre, baseline_tail), axis=4)
#
#             # dF/F
#             baseline = np.mean(baseline_points, axis = (3,4))
#             baseline = np.expand_dims(np.expand_dims(baseline, axis = 3), axis = 4)
#             response_dff = (brain_trial_matrix[:,:,:,pull_inds,:] - baseline) / baseline
#             baseline_dff = (baseline_points - baseline) / baseline
# #
# #            #set any nans (baseline 0) to 0. 0s come from registration
# #            baseline_dff[np.isnan(baseline_dff)] = 0
# #            response_dff[np.isnan(response_dff)] = 0
# #
#             _, p = stats.ttest_ind(np.reshape(baseline_dff, (x_dim, y_dim, z_dim, -1)),
#                                    np.reshape(response_dff[:,:,:,:,pre_frames:(pre_frames+stim_frames)], (x_dim, y_dim, z_dim, -1)), axis = 3)
#
#             p_values.append(p)
#
#             mean_resp.append(np.mean(response_dff, axis = 3))
#             std_resp.append(np.std(response_dff, axis = 3))
#
#         p_values = np.stack(p_values, axis = 3) #x y z stimulus
#         mean_resp = np.concatenate(mean_resp, axis = 3)
#         std_resp = np.concatenate(std_resp, axis = 3)
#
#         return mean_resp, std_resp, p_values, unique_parameter_values