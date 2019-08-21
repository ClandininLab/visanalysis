import numpy as np
import h5py



def saveRoiSet(file_path, roi_set_path, roi_response, roi_image, roi_path, roi_mask):
    with h5py.File(file_path, 'r+') as experiment_file:
        roi_group = experiment_file.require_group(roi_set_path)

        if roi_group.get("roi_mask"):  # roi dataset exists
            del roi_group["roi_mask"]
        if roi_group.get("roi_response"):
            del roi_group["roi_response"]
        if roi_group.get("roi_image"):
            del roi_group["roi_image"]

        for dataset_key in roi_group.keys():
            if 'path_vertices' in dataset_key:
                del roi_group[dataset_key]

        roi_group.create_dataset("roi_mask", data=roi_mask)
        roi_group.create_dataset("roi_response", data=roi_response)
        roi_group.create_dataset("roi_image", data=roi_image)
        for p_ind, p in enumerate(roi_path):
            roi_group.create_dataset("path_vertices_" + str(p_ind), data=p.vertices)


def loadRoiSet(file_path, roi_set_path):
    roi_set_name = roi_set_path.split('/')[-1]
    with h5py.File(file_path, 'r') as experiment_file:
        if 'series' in roi_set_name:  # roi set from a different series
            series_no = roi_set_name.split(':')[0].split('series')[1]
            roi_name = roi_set_name.split(':')[1]
            #TODO: FIXME
            roi_set_group = experiment_file['/epoch_runs'].get(series_no).get('rois').get(roi_name)
            roi_response = [getRoiDataFromMask(x) for x in roi_mask]

        else:  # from this series
            roi_set_group = experiment_file[roi_set_path]
            roi_response = list(roi_set_group.get("roi_response")[:])
            roi_mask = list(roi_set_group.get("roi_mask")[:])
            roi_image = roi_set_group.get("roi_image")[:]

            roi_path = []
            new_path = roi_set_group.get("path_vertices_0")
            ind = 0
            while new_path is not None:
                roi_path.append(new_path)
                ind += 1
                new_path = roi_set_group.get("path_vertices_" + str(ind))
            roi_path = [x[:] for x in roi_path]

    return roi_response, roi_image, roi_path, roi_mask


def getRoiMask(image, indices):
    array = np.zeros((image.shape[0], image.shape[1]))
    lin = np.arange(array.size)
    newRoiArray = array.flatten()
    newRoiArray[lin[indices]] = 1
    newRoiArray = newRoiArray.reshape(array.shape)
    mask = newRoiArray == 1 #convert to boolean for masking
    return mask

def getRoiDataFromMask(current_series, mask):
    roi_response = (np.mean(current_series[:, mask], axis=1, keepdims=True) - np.min(current_series)).T
    return roi_response
