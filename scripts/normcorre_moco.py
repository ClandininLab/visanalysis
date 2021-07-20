import os
import nibabel as nib
import numpy as np
import time
from tifffile.tifffile import imsave
import caiman as cm
import caiman.source_extraction.cnmf as cnmf

data_dir = '/oak/stanford/groups/trc/data/Max/ImagingData/Bruker'

fn = 'TSeries-20210217-007'

# %% Load brains to Oak and save to SCRATCH as .tifs

t0 = time.time()


# Load red brain (Ch1)
red_fp = os.path.join(data_dir, date_str, '{}_channel_1.nii'.format(fn))
red_brain = np.asanyarray(nib.load(red_fp).dataobj).astype('uint32')  # xyzt

# Load green brain (Ch2)
date_str = fn.split('-')[1]
green_fp = os.path.join(data_dir, date_str, '{}_channel_2.nii'.format(fn))
green_brain = np.asanyarray(nib.load(green_fp).dataobj).astype('uint32') # xyzt

# reshape to t, x, y, z (caiman / normcorre wants T first)
green_brain = np.moveaxis(green_brain, (0, 1, 2, 3), (1, 2, 3, 0))
red_brain = np.moveaxis(red_brain, (0, 1, 2, 3), (1, 2, 3, 0))

# save to TIF files
green_tif_filepath = os.path.join(os.environ.get('SCRATCH'), 'tmp_images', 'tmp_green_brain.tif')
imsave(green_tif_filepath, green_brain)

red_tif_filepath = os.path.join(os.environ.get('SCRATCH'), 'tmp_images', 'tmp_red_brain.tif')
imsave(red_tif_filepath, red_brain)

# Measure original dynamic range for each frame
green_min = green_brain.min(axis=(1, 2, 3))
green_max = green_brain.max(axis=(1, 2, 3))

red_min = red_brain.min(axis=(1, 2, 3))
red_max = red_brain.max(axis=(1, 2, 3))

del green_brain, red_brain # for memory saving

#%% start a cluster for parallel processing
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)

# %% Set motion correction parameters
mc_opts_dict = {'strides': (24, 24, 6),    # start a new patch for pw-rigid motion correction every x pixels
                'overlaps': (12, 12, 2),   # overlap between patches (size of patch strides+overlaps)
                'max_shifts': (4, 4, 2),   # maximum allowed rigid shifts (in pixels)
                'max_deviation_rigid': 5,  # maximum shifts deviation allowed for patch with respect to rigid shifts
                'pw_rigid': False,         # flag for performing non-rigid motion correction
                'is3D': True}

mc_opts = cnmf.params.CNMFParams(params_dict=mc_opts_dict)

# %% Mem-map channels and do motion correction based on RED channel

# Create memmap objects for raw frames
file_basename = os.path.join(os.environ.get('SCRATCH'), 'memmaps', 'redraw')
red_mmap = cm.save_memmap([red_tif_filepath], base_name=file_basename, is_3D=True, order='C')

file_basename = os.path.join(os.environ.get('SCRATCH'), 'memmaps', 'greenraw')
green_mmap = cm.save_memmap([green_tif_filepath], base_name=file_basename, is_3D=True, order='C')

# Create a motion correction object for red channel
mc = cm.motion_correction.MotionCorrect(red_mmap, dview=dview, **mc_opts.get_group('motion'))

# Run motion correction on red channel
mc.motion_correct(save_movie=True)

# Save mc red as memmap
file_basename = os.path.join(os.environ.get('SCRATCH'), 'memmaps', 'redmc')
red_mc_mmap = cm.save_memmap(mc.mmap_file, base_name=file_basename, order='C',
                             border_to_0=0, dview=dview)

# Apply mc from red channel to green channel
green_mc = mc.apply_shifts_movie(green_mmap)

# Save green as memmap
file_basename = os.path.join(os.environ.get('SCRATCH'), 'memmaps', 'greenmc')
green_mc_mmap = cm.save_memmap([green_mc], base_name=file_basename, order='C',
                             border_to_0=0, dview=dview)

# %% Save motion corrected movie as merged .nii file

# Save mc frames to Oak as merged nii file


# load the files from memmapped
Y_grn, dims, T = cm.load_memmap(green_mc_mmap)

# Scale registered frames back to original dynamic range
m = (green_max - green_min) / (Y_grn.max(axis=0) - Y_grn.min(axis=0))
b = green_min - m * Y_grn.min(axis=0)
Y_grn_scaled= m * Y_grn + b

ch2 = np.reshape(Y_grn_scaled.T, [T] + list(dims), order='F').astype('uint16')
# Re-order to xyzt
ch2 = np.moveaxis(ch2, (1, 2, 3, 0), (0, 1, 2, 3))


Y_rd, dims, T = cm.load_memmap(red_mc_mmap)

# Scale registered frames back to original dynamic range
m = (red_max - red_min) / (Y_rd.max(axis=0) - Y_rd.min(axis=0))
b = red_min - m * Y_rd.min(axis=0)
Y_red_scaled= m * Y_rd + b

ch1 = np.reshape(Y_red_scaled.T, [T] + list(dims), order='F').astype('uint16')
# Re-order to xyzt
ch1 = np.moveaxis(ch1, (1, 2, 3, 0), (0, 1, 2, 3))

merged = np.stack([ch1, ch2], axis=-1) # xyztc

# Save registered, merged .nii to Oak
fp = os.path.join(data_dir, date_str, '{}_normcorred.nii'.format(fn))
nifti1_limit = (2**16 / 2)
if np.any(np.array(merged.shape) >= nifti1_limit): # Need to save as nifti2
    nib.save(nib.Nifti2Image(merged, np.eye(4)), fp)
else: # Nifti1 is OK
    nib.save(nib.Nifti1Image(merged, np.eye(4)), fp)
print('Saved registered brain to {}. Total time = {:.1f}'.format(fp, time.time()-t0))
