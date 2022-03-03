"""
Example script: attach region responses to datafile using a mask
    instead of hand-drawing rois

https://github.com/ClandininLab/visanalysis
mhturner@stanford.edu
"""
from visanalysis.plugin import bruker
from visanalysis.analysis import imaging_data
import os
import numpy as np
import matplotlib.pyplot as plt
import skimage.morphology as morphology

experiment_file_directory = '/Users/mhturner/GitHub/visanalysis/examples/example_data/responses/bruker'
experiment_file_name = '2021-07-07'
series_number = 1

file_path = os.path.join(experiment_file_directory, experiment_file_name + '.hdf5')

# %% Make a mask for the image data, and save mask region responses to data file

# Make a bruker plugin object
bruker_data_object = bruker.BrukerPlugin()
# Associate the Imaging File...
bruker_data_object.updateImagingDataObject(experiment_file_directory,
                                           experiment_file_name,
                                           series_number)
# Associate the raw image data...
bruker_data_object.updateImageSeries(data_directory=experiment_file_directory,
                                     image_file_name='TSeries-20210707-001_reg.nii',
                                     series_number=series_number,
                                     channel=1)

# Make a test mask with 3 unique values: 0, 1, 2
#    By default doesn't compute region responses for mask=0. See include_zero below...
mask = np.zeros(bruker_data_object.current_series.shape[:3])
ball_1 = morphology.ball(1)
cube_2 = morphology.cube(3)
mask[:ball_1.shape[0], :ball_1.shape[1], :ball_1.shape[2]] = ball_1 * 1
mask[-cube_2.shape[0]:, -cube_2.shape[1]:, -cube_2.shape[2]:] = cube_2 * 2

# Save region responses and mask to data file
bruker_data_object.saveRegionResponsesFromMask(
                                               file_path=file_path,
                                               series_number=series_number,
                                               response_set_name='mask_1',
                                               mask=mask,
                                               include_zero=False)

# %% Retrieve saved mask region responses from data file

ID = imaging_data.ImagingDataObject(file_path,
                                    series_number,
                                    quiet=False)

# Mask-aligned roi data gets saved under /aligned
# Hand-drawn roi data gets saved under /rois
ID.getRoiSetNames(roi_prefix='aligned')

# You can access the aligned region response data just as with hand-drawn rois, using the 'aligned' prefix argument
roi_data = ID.getRoiResponses('mask_1', roi_prefix='aligned')

# %% Plot region responses and masks
z_slice = 2
fh, ax = plt.subplots(2, 2, figsize=(8, 3),
                      gridspec_kw={'width_ratios': [1, 4]})
[x.set_axis_off() for x in ax.ravel()]

colors = 'rgb'
for r_ind, response in enumerate(roi_data['roi_response']):
    ax[r_ind, 0].imshow((roi_data['roi_mask'][:, :, z_slice] == (r_ind+1)).T)
    ax[r_ind, 1].plot(response, color=colors[r_ind])


# %%
