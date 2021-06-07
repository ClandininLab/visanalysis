"""
Tools for brain volume loading, registration, Bruker stuff.

---ANTSpy registration params---
Ref: https://rstudio-pubs-static.s3.amazonaws.com/295353_14e261742cae4e7fb237de43b74d9d0c.html
flowSigma - this will regularize the similarity metric graident, which we follow to get a good registration …. higher sigma focuses on coarser features
totalSigma - this will regularize the total deformation field. usually zero, higher values will restrict the amount of deformation allowed

@author: mhturner
"""
import ants
import nibabel as nib
import numpy as np
import xml.etree.ElementTree as ET
import time
from scipy.ndimage import gaussian_filter
from scipy.signal import medfilt


def getMarkPointsMetaData(file_path):
    metadata = {}

    root = ET.parse(file_path).getroot()
    for key in root.keys():
        metadata[key] = root.get(key)

    point_element = root.find('PVMarkPointElement')
    for key in point_element.keys():
        metadata[key] = point_element.get(key)

    galvo_element = point_element[0]
    for key in galvo_element.keys():
        metadata[key] = galvo_element.get(key)

    points = list(galvo_element)
    for point_ind, point in enumerate(points):
        for key in point.keys():
            metadata['Point_{}_{}'.format(point_ind+1, key)] = point.get(key)

    return metadata

def getBrukerMetaData(file_path, get_frame_times=False):
    """
    Parse Bruker / PrairieView metadata from .xml file.

    file_path: .xml filepath
    get_frame_times: bool, return frame timestamps from metadata as well
    returns
        metadata: dict
    """
    root = ET.parse(file_path).getroot()

    metadata = {}
    for child in list(root.find('PVStateShard')):
        if child.get('value') is None:
            for subchild in list(child):
                new_key = child.get('key') + '_' + subchild.get('index')
                new_value = subchild.get('value')
                metadata[new_key] = new_value

        else:
            new_key = child.get('key')
            new_value = child.get('value')
            metadata[new_key] = new_value

    metadata['version'] = root.get('version')
    metadata['date'] = root.get('date')
    metadata['notes'] = root.get('notes')

    if get_frame_times:
        if root.find('Sequence').get('type') == 'TSeries Timed Element': # Plane time series
            frame_times = [float(fr.get('relativeTime')) for fr in root.find('Sequence').findall('Frame')]

        elif root.find('Sequence').get('type') == 'TSeries ZSeries Element': # Volume time series
            middle_frame = int(len(root.find('Sequence').findall('Frame')) / 2)
            frame_times = [float(seq.findall('Frame')[middle_frame].get('relativeTime')) for seq in root.findall('Sequence')]

        metadata['frame_times'] = frame_times

    return metadata


def getAntsBrain(fn, metadata, channel=0):
    """Load .nii brain file as ANTs image."""
    nib_brain = np.asanyarray(nib.load(fn).dataobj).astype('uint32')
    spacing = (float(metadata['micronsPerPixel_XAxis']), float(metadata['micronsPerPixel_YAxis']), float(metadata['micronsPerPixel_ZAxis']), nib_brain.shape[2] * float(metadata['framePeriod']))
    if len(nib_brain.shape) == 5: # xyztc
        # trim to single channel: xyzt
        return ants.from_numpy(nib_brain[:, :, :, :, channel], spacing=spacing)

    elif len(nib_brain.shape) == 4: # xyzt, single channel
        # return ants.from_numpy(nib_brain[:, :, :, :300], spacing=spacing) # TESTING
        return ants.from_numpy(nib_brain, spacing=spacing)


def getSmoothBrain(brain, smoothing_sigma=[1.0, 1.0, 0.0, 2.0]):
    """
    Gaussian smooth brain.

    brain: xyzt ants brain
    smoothing_sigma: [x, y, z, t] Gaussian smoothing kernel

    returns smoothed brain, xyzt ants
    """
    spacing = brain.spacing

    return ants.from_numpy(gaussian_filter(brain[:, :, :, :], sigma=smoothing_sigma), spacing=spacing)  # xyz


def getTimeAveragedBrain(brain, frames=None):
    """
    Time average brain.

    brain: xyzt ants brain
    frames: average from 0:n frames. Note None -> average over all time

    returns timeaverage brain, xyz ants
    """
    spacing = brain.spacing
    return ants.from_numpy(brain[:, :, :, 0:frames].mean(axis=3), spacing=spacing[0:3])  # xyz


def getTransformMatrix(transform):
    """
    Return np matrix of ANTs transform parameters.

    transform: list of ANTs transforms
    """
    transform_matrix = []
    for i, t in enumerate(transform):
        for x in t:
            if '.mat' in x:
                temp = ants.read_transform(x)
                transform_matrix.append(temp.parameters)
    transform_matrix = np.array(transform_matrix)
    return transform_matrix


def filterTransformMatrix(transform_matrix):
    """
    Median filter ANTs transform parameters for sudden jumps.

    transform_matrix: matrix of ANTs transform parameters (time, parameters)

    return
        filtered_transform_matrix (time, parameters)
    """
    med_filtered = np.vstack([medfilt(transform_matrix[:, x], kernel_size=9) for x in range(transform_matrix.shape[1])]).T

    return med_filtered # (time, parameters)

def computeTransform(brain, reference, type_of_transform='Rigid', flow_sigma=3, total_sigma=0):
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
    spacing = brain.spacing

    transform_matrix = []
    fixed_matrix = []
    # corr_out = [] # uncomment and return this to get directly corrected brain
    for brain_frame in range(brain.shape[3]):
        reg = ants.registration(reference, ants.from_numpy(brain[:, :, :, brain_frame], spacing=spacing[0:3]), type_of_transform=type_of_transform, flow_sigma=flow_sigma, total_sigma=total_sigma)
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

def applyTransform(brain_list, reference_list, transform_matrix, fixed_matrix):
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
        spacing = brain.spacing
        corrected = []
        for brain_frame in range(brain.shape[3]):
            tx = ants.create_ants_transform(transform_type='AffineTransform')
            tx.set_parameters(transform_matrix[brain_frame, :])
            tx.set_fixed_parameters(fixed_matrix[brain_frame, :])

            corrected_frame = tx.apply_to_image(ants.from_numpy(brain[:, :, :, brain_frame], spacing=spacing[0:3]), reference_list[b_ind])

            corrected.append(corrected_frame.numpy())

        # swap axes back to match original xyzt and cast back to uint16
        transformed_brain_list.append(np.moveaxis(np.asarray(corrected), 0, 3).astype('uint16'))

    print('Transform applied: ({:.2f} sec)'.format(time.time()-t0))

    return transformed_brain_list


def mergeChannels(ch1, ch2):
    """
    Merge two channel brains into single array.

    ch1, ch2: np array single channel brain (xyzt)

    return
        merged np array, 2 channel brain (xyztc)

    """
    return np.stack([ch1, ch2], axis=4) # xyztc
