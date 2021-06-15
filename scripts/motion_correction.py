"""
Example motion correction script.

https://github.com/ClandininLab/visanalysis
mhturner@stanford.edu
"""
import sys
import time
import nibabel as nib
import numpy as np
from visanalysis import registration

t0 = time.time()

# first arg: path to image series base, without .suffix
#   e.g. /oak/stanford/groups/trc/data/Max/ImagingData/Bruker/20210611/TSeries-20210611-001
file_base_path = sys.argv[1]
print('Registering brain file from {}'.format(file_base_path))

# Load metadata from bruker .xml file
metadata = registration.get_bruker_metadata(file_base_path + '.xml')
print('Loaded metadata from {}'.format(file_base_path + '.xml'))

# Load brain images
ch1 = registration.get_ants_brain(file_base_path + '_channel_1.nii', metadata, channel=0)
print('Loaded {}, shape={}'.format(file_base_path + '_channel_1.nii', ch1.shape))
ch2 = registration.get_ants_brain(file_base_path + '_channel_2.nii', metadata, channel=0)
print('Loaded {}, shape={}'.format(file_base_path + '_channel_2.nii', ch2.shape))

# Register both channels to channel 1
merged = registration.register_two_channels_to_red(ch1, ch2, spatial_dims=len(ch1.shape) - 1)

# Save registered, merged .nii
nib.save(nib.Nifti1Image(merged, np.eye(4)), file_base_path + '_reg.nii')
print('Saved registered brain to {}. Total time = {:.1f}'.format(file_base_path + '_reg.nii', time.time()-t0))
