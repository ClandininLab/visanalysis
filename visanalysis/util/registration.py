"""
Tools for brain loading, registration. Relies on ANTSpy.

---ANTSpy registration params---
Ref: https://rstudio-pubs-static.s3.amazonaws.com/295353_14e261742cae4e7fb237de43b74d9d0c.html
flowSigma - this will regularize the similarity metric graident,
    which we follow to get a good registration …. higher sigma focuses on coarser features
totalSigma - this will regularize the total deformation field.
    usually zero, higher values will restrict the amount of deformation allowed

https://github.com/ClandininLab/visanalysis
mhturner@stanford.edu
"""
import time

import ants
import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter
from scipy.signal import medfilt


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # #  Load image files # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def get_ants_brain(filepath, metadata, channel=0):
    """Load .nii brain file as ANTs image."""
    nib_brain = np.asanyarray(nib.load(filepath).dataobj).astype('uint32')
    spacing = [float(metadata.get('micronsPerPixel_XAxis', 0)),
               float(metadata.get('micronsPerPixel_YAxis', 0)),
               float(metadata.get('micronsPerPixel_ZAxis', 0)),
               float(metadata.get('sample_period', 0))]
    spacing = [spacing[x] for x in range(4) if metadata['image_dims'][x] > 1]

    if len(nib_brain.shape) > 4:  # multiple channels
        # trim to single channel
        return ants.from_numpy(np.squeeze(nib_brain[..., channel]), spacing=spacing)
    else:
        # return ants.from_numpy(np.squeeze(nib_brain[..., :300]), spacing=spacing) # TESTING
        return ants.from_numpy(np.squeeze(nib_brain), spacing=spacing)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # #  Image processing # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def get_smooth_brain(brain, smoothing_sigma=[1.0, 1.0, 0.0, 2.0]):
    """
    Gaussian smooth brain.

    brain: ants brain. shape = (spatial..., t)
    smoothing_sigma: Gaussian smoothing kernel. len = rank of brain
        Spatial dims come first. T last. Default dim is [x, y, z, t]

    returns smoothed brain, ants. Same dims as input brain
    """
    smoothed = gaussian_filter(brain.numpy(), sigma=smoothing_sigma)

    return ants.from_numpy(smoothed, spacing=brain.spacing)  # xyz


def get_time_averaged_brain(brain, frames=None):
    """
    Time average brain.

    brain: (spatial, t) ants brain
    frames: average from 0:n frames. Note None -> average over all time

    returns time-averaged brain, ants. Dim =  (spatial)
    """
    spacing = list(np.array(brain.spacing)[..., :-1])
    return ants.from_numpy(brain[..., 0:frames].mean(axis=len(brain.shape)-1), spacing=spacing)


def merge_channels(ch1, ch2):
    """
    Merge two channel brains into single array.

    ch1, ch2: np array single channel brain (dims)

    return
        merged np array, 2 channel brain (dims, c)

    """
    return np.stack([ch1, ch2], axis=-1)  # c is last dimension


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # Transformations # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def get_transform_matrix(transform):
    """
    Return np matrix of ANTs transform parameters.

    transform: list of ANTs transforms
    """
    transform_matrix = []
    for _, t in enumerate(transform):
        for x in t:
            if '.mat' in x:
                temp = ants.read_transform(x)
                transform_matrix.append(temp.parameters)
    transform_matrix = np.array(transform_matrix)
    return transform_matrix


def filter_transform_matrix(transform_matrix):
    """
    Median filter ANTs transform parameters for sudden jumps.

    transform_matrix: matrix of ANTs transform parameters (time, parameters)

    return
        filtered_transform_matrix (time, parameters)
    """
    med_filtered = [medfilt(transform_matrix[:, x], kernel_size=9) for x in range(transform_matrix.shape[1])]

    return np.vstack(med_filtered).T  # (time, parameters)


def compute_transform(brain, reference, type_of_transform='Rigid', flow_sigma=3, total_sigma=0):
    """
    Compute transform for time series brain to reference brain.

    brain: xyzt ants brain
    reference: xyz ants brain
    type_of_transform: 'Rigid' or 'Translation' are good
    flow_sigma:
    total_sigma:

    return
        transform_matrix: matrix of ANTs transform parameters (time, parameters)
        fixed_matrix: matrix of ANTs fixed transform parameters (time, parameters)
    """
    t0 = time.time()
    frame_spacing = list(np.array(brain.spacing)[..., :-1])

    transform_matrix = []
    fixed_matrix = []
    # corr_out = [] # uncomment and return this to get directly corrected brain
    for brain_frame in range(brain.shape[-1]):  # for time steps
        reg = ants.registration(reference,
                                ants.from_numpy(brain[..., brain_frame], spacing=frame_spacing),
                                type_of_transform=type_of_transform,
                                flow_sigma=flow_sigma,
                                total_sigma=total_sigma)
        # corr_out.append(reg['warpedmovout'].numpy())
        trans = reg['fwdtransforms']
        for x in trans:
            if '.mat' in x:
                temp = ants.read_transform(x)
                transform_matrix.append(temp.parameters)
                fixed_matrix.append(temp.fixed_parameters)

    # corr_out = np.moveaxis(np.asarray(corr_out), 0, 3).astype('uint16')

    print('Transform computed: ({:.2f} sec)'.format(time.time()-t0))

    return np.array(transform_matrix), np.array(fixed_matrix)


def apply_transform(brain_list, reference_list, transform_matrix, fixed_matrix):
    """
    Apply transforms to brain_list.

    brain_list: list of xyzt ants brains to transform. Must each have the same time dimension
    reference_list: list of xyz ANTs brains to act as references, just define spaces to transform into
    transform_matrix: matrix of ANTs transform parameters
    fixed_matrix: matrix of ANTs fixed transform parameters

    return
        transformed_brain_list: list of transformed brains (np ndarrays)
    """
    t0 = time.time()

    transformed_brain_list = []
    for b_ind, brain in enumerate(brain_list):
        frame_spacing = list(np.array(brain.spacing)[..., :-1])
        corrected = []
        for brain_frame in range(brain.shape[-1]):  # for time steps
            tx = ants.create_ants_transform(transform_type='AffineTransform', dimension=len(frame_spacing))
            tx.set_parameters(transform_matrix[brain_frame, :])
            tx.set_fixed_parameters(fixed_matrix[brain_frame, :])

            ants_frame = ants.from_numpy(brain[..., brain_frame], spacing=frame_spacing)
            corrected_frame = tx.apply_to_image(ants_frame, reference_list[b_ind])

            corrected.append(corrected_frame.numpy())

        corrected = np.stack(corrected, -1)

        # swap axes back to match original xyzt and cast back to uint16
        transformed_brain_list.append(corrected.astype('uint16'))

    print('Transform applied: ({:.2f} sec)'.format(time.time()-t0))

    return transformed_brain_list


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # #  Common motion correction functions # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def registerToReferenceChannel_FilterTransforms(reference_channel, moving_channel, spatial_dims=3, reference_frames=100):
    """
    Register 2 channels to channel 1 (red). Separate compute & apply steps to fliter transform fixed_matrix
       in between. Filtering useful for removing sudden artifacts that sometimes arise with sparse, noisy volumes
    Inputs:
        reference_channel, moving_channel: Ants images
        spatial_dims: 2 or 3

    returns:
        Merged two-channel ndarray
    """
    if spatial_dims == 3:
        smoothing_sigma = [1.0, 1.0, 0.0, 2.0]  # xyzt
    elif spatial_dims == 2:
        smoothing_sigma = [1.0, 1.0, 2.0]  # xyt

    reference = get_time_averaged_brain(get_smooth_brain(reference_channel, smoothing_sigma=smoothing_sigma), frames=reference_frames)
    transform_mat, fixed_mat = compute_transform(brain=get_smooth_brain(reference_channel, smoothing_sigma=smoothing_sigma),
                                                 reference=reference,
                                                 type_of_transform='Rigid',
                                                 flow_sigma=3,
                                                 total_sigma=0)

    # filter transforms to remove single frame artifacts
    filtered_transform_mat = filter_transform_matrix(transform_mat)

    # apply transforms to ch1 + ch2 brains
    transformed_brain_list = apply_transform(brain_list=[reference_channel, moving_channel],
                                             reference_list=[get_time_averaged_brain(reference_channel), get_time_averaged_brain(moving_channel)],
                                             transform_matrix=filtered_transform_mat,
                                             fixed_matrix=fixed_mat)

    merged = merge_channels(transformed_brain_list[0], transformed_brain_list[1])

    return merged


def registerToSelf_FilterTransforms(brain, spatial_dims=3, reference_frames=100):
    """
    Register 1 channel to itself
    Inputs:
        brain: Ants image
        spatial_dims: 2 or 3

    returns:
        xyzt ndarray
    """
    if spatial_dims == 3:
        smoothing_sigma = [1.0, 1.0, 0.0, 2.0] # xyz
    elif spatial_dims == 2:
        smoothing_sigma = [1.0, 1.0, 2.0] # xyt

    reference = get_time_averaged_brain(get_smooth_brain(brain, smoothing_sigma=smoothing_sigma), frames=reference_frames)
    transform_mat, fixed_mat = compute_transform(brain=get_smooth_brain(brain, smoothing_sigma=smoothing_sigma),
                                                reference=reference,
                                                type_of_transform='Rigid',
                                                flow_sigma=3,
                                                total_sigma=0)

    # filter transforms to remove single frame artifacts
    filtered_transform_mat = filter_transform_matrix(transform_mat)

    # apply transforms
    transformed_brain_list = apply_transform(brain_list=[brain],
                                            reference_list=[get_time_averaged_brain(brain)],
                                            transform_matrix=filtered_transform_mat,
                                            fixed_matrix=fixed_mat)

    return transformed_brain_list[0]


def registerToReferenceChannel(reference_channel, moving_channel, spatial_dims=3, reference_frames=100,
                               type_of_transform='Rigid', flow_sigma=3, total_sigma=0):
    t0 = time.time()
    if spatial_dims == 3:
        smoothing_sigma = [1.0, 1.0, 0.0, 2.0]  # xyzt
    elif spatial_dims == 2:
        smoothing_sigma = [1.0, 1.0, 2.0]  # xyt

    reference_brain = get_time_averaged_brain(get_smooth_brain(reference_channel, smoothing_sigma=smoothing_sigma),
                                              frames=reference_frames)

    reference_corrected = []
    moving_corrected = []
    for brain_frame in range(reference_channel.shape[-1]):  # for time steps
        reg = ants.registration(reference_brain,
                                ants.from_numpy(reference_channel[..., brain_frame], spacing=reference_brain.spacing),
                                type_of_transform=type_of_transform,
                                flow_sigma=flow_sigma,
                                total_sigma=total_sigma)

        transformlist = reg['fwdtransforms']

        reference_corrected.append(reg['warpedmovout'].numpy())
        moving_corrected.append(ants.apply_transforms(reference_brain,
                                                      ants.from_numpy(moving_channel[..., brain_frame], spacing=reference_brain.spacing),
                                                      transformlist).numpy())

    # Shape = (xyzt). Cast back to 16bit unsigned integers
    reference_corrected = np.stack(reference_corrected, -1).astype('uint16')
    moving_corrected = np.stack(moving_corrected, -1).astype('uint16')

    merged = merge_channels(reference_corrected, moving_corrected)

    return merged

    print('Two channel brain registered: ({:.2f} sec)'.format(time.time()-t0))
