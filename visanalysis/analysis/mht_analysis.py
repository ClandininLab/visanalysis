import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


from visanalysis import plot_tools
from visanalysis.analysis import shared_analysis



def doMovingSquareMappingAnalysis(ImagingData, roi_name, plot_ind = 0):
    plot_colors = sns.color_palette("deep",n_colors = 10)
    
    
    
    time_vector = ImagingData.roi.get(roi_name).get('time_vector')
    response_matrix = ImagingData.roi.get(roi_name).get('epoch_response')

    axis = []; location = []
    for ep in ImagingData.epoch_parameters:
        if ep['current_search_axis'] == 'azimuth':
            axis.append(1)
            location.append(ep['current_location'])
        elif ep['current_search_axis'] == 'elevation':
            axis.append(2)
            location.append(ep['current_location'])
    
    parameter_values = np.array([axis, location]).T
    
    unique_parameter_values, mean_trace_matrix, sem_trace_matrix, individual_traces = shared_analysis.getTraceMatrixByStimulusParameter(response_matrix,parameter_values)


    plot_y_max = 6.0
    plot_y_min = -0.5

    azimuth_inds = np.where(unique_parameter_values[:,0] == 1)[0]
    elevation_inds = np.where(unique_parameter_values[:,0] == 2)[0]

    azimuth_locations = np.unique(np.array(location)[np.where(np.array(axis) == 1)])
    elevation_locations = np.unique(np.array(location)[np.where(np.array(axis) == 2)])

    fig_handle = plt.figure(figsize=(9,6))
    grid = plt.GridSpec(len(elevation_locations)+1, len(azimuth_locations)+1, wspace=0.4, hspace=0.3)
    for ind_a, a in enumerate(azimuth_inds):
#        current_loc = unique_parameter_values[a,1]
        current_mean = mean_trace_matrix[:,a,:]
        current_sem = sem_trace_matrix[:,a,:]

        new_ax = fig_handle.add_subplot(grid[0,ind_a+1])
            
        new_ax.plot(time_vector, current_mean[plot_ind,:], alpha=1, color = plot_colors[plot_ind], linewidth=2)
        new_ax.fill_between(time_vector,
                            current_mean[plot_ind,:].T - current_sem[plot_ind,:].T,
                            current_mean[plot_ind,:].T + current_sem[plot_ind,:].T, alpha = 0.2)
        
        

        new_ax.set_ylim([plot_y_min, plot_y_max])
        new_ax.set_axis_off()
        if ind_a == 0:
            plot_tools.addScaleBars(new_ax, 1, 1, F_value = -0.4, T_value = -0.4)
        fig_handle.canvas.draw()
        
    for ind_e, e in enumerate(elevation_inds):
#        current_loc = unique_parameter_values[e,1]
        current_mean = mean_trace_matrix[:,e,:]
        current_sem = sem_trace_matrix[:,e,:]

        new_ax = fig_handle.add_subplot(grid[ind_e+1,0])
            
        new_ax.plot(time_vector, current_mean[plot_ind,:], alpha=1, color = plot_colors[plot_ind], linewidth=2)
        new_ax.fill_between(time_vector,
                            current_mean[plot_ind,:].T - current_sem[plot_ind,:].T,
                            current_mean[plot_ind,:].T + current_sem[plot_ind,:].T, alpha = 0.2)
        
        new_ax.set_ylim([plot_y_min, plot_y_max])
        new_ax.set_axis_off()
        if ind_e == 0:
            plot_tools.addScaleBars(new_ax, 1, 1, F_value = -0.4, T_value = -0.4)
        fig_handle.canvas.draw()
     

    # single heat mat to go with example traces
    resp_az = np.max(mean_trace_matrix[:,azimuth_inds,:], axis = 2)
    resp_el = np.max(mean_trace_matrix[:,elevation_inds,:], axis = 2)
    
    resp_mat = []
    for rr in range(current_mean.shape[0]):
        new_resp_mat = np.outer(resp_az[rr,:],resp_el[rr,:]).T
        resp_mat.append(new_resp_mat)
    
    heat_map_ax = fig_handle.add_subplot(grid[1:len(elevation_locations)+1, 1:len(azimuth_locations)+1])
    
    extent=[azimuth_locations[0], azimuth_locations[-1],
                    elevation_locations[-1], elevation_locations[0]]
        
    heat_map = resp_mat[plot_ind]
    heat_map_ax.imshow(heat_map, extent=extent, cmap=plt.cm.Reds,interpolation='none')
    heat_map_ax.yaxis.tick_right()
    heat_map_ax.tick_params(axis='x', which='major', labelsize=12)
    heat_map_ax.tick_params(axis='y', which='major', labelsize=12)
    
        