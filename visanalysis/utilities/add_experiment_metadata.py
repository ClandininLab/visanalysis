import h5py
import os

flystim_data_directory = '/Users/mhturner/Dropbox/ClandininLab/CurrentData/FlystimData'
file_name = '2018-09-05'


fly_metadata = {'fly:fly_id':'Fly1',
                'fly:sex':'Female',
                'fly:age':6, 
                'fly:prep':'Left optic lobe',
                'fly:driver_1':'LC26 (VT007747; R85H06)',
                'fly:indicator_1':'GCaMP6f', 
                'fly:driver_2':'',
                'fly:indicator_2':'',
                'fly:genotype':''}


with h5py.File(os.path.join(flystim_data_directory, file_name) + '.hdf5','r+') as experiment_file:
    epochRuns = experiment_file['/epoch_runs']
    
    for run_no in epochRuns:
        if int(run_no) >17 :
            for key in fly_metadata:
                epochRuns.get(str(run_no)).attrs[key] = fly_metadata[key]
        
# %%
import h5py
import os

          
flystim_data_directory = '/Users/mhturner/Dropbox/ClandininLab/CurrentData/FlystimData'
file_name = '2018-xx-xx'

expt_file = h5py.File(os.path.join(flystim_data_directory, file_name) + '.hdf5','r+')

