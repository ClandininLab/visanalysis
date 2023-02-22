def nansem(a, axis=0, ddof=1, nan_policy='omit'):
    return sem(a, axis, ddof, nan_policy)

def uneven_list2d_to_np(v, fillval=np.nan):
    lens = np.array([len(item) for item in v])
    if len(np.unique(lens)) == 1:
        return np.asarray(v)
    mask = lens[:,None] > np.arange(lens.max())
    out = np.full(mask.shape,fillval)
    out[mask] = np.concatenate(v)
    return out

def generate_standard_timestamp(timestamps):
    '''
    timestamps: 2d numpy array with nan padding for uneven timestamp lengths

    Finds mean framerate and generates a single timestamp series starting from 0 evenly spaced to the max timestamp.
    '''
    if not isinstance(timestamps, np.ndarray):
        timestamps = uneven_list2d_to_np(timestamps)
    mean_diff = np.nanmean(np.diff(timestamps))
    min_time = np.nanmin(timestamps)
    max_time = np.nanmax(timestamps)

    return np.arange(min_time, max_time, mean_diff)
