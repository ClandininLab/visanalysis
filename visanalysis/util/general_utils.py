import numpy as np
from scipy.stats import sem
import collections

def nansem(a, axis=0, ddof=1, nan_policy='omit'):
    return sem(a, axis, ddof, nan_policy)

def mean_and_error(signals_aligned, mean_fxn=np.nanmean, error_fxn=nansem, axis=0):
    '''
    Input:
        timestamp: 1d array of timestamps
        signals_aligned: 2d array of signals, trials by time
        do_plot
    Returns:
        mean
        error (stdev or sem)
    '''

    mean = mean_fxn(signals_aligned, axis=axis)
    error = error_fxn(signals_aligned, axis=axis)

    return mean, error

def uneven_list2d_to_np(v, fillval=np.nan):
    lens = np.array([len(item) for item in v])
    if len(np.unique(lens)) == 1:
        return np.asarray(v)
    mask = lens[:,None] > np.arange(lens.max())
    out = np.full(mask.shape,fillval)
    out[mask] = np.concatenate(v)
    return out

def generate_standard_timestamp(timestamps, trim=False, min_time=None, max_time=None):
    '''
    timestamps: 2d numpy array with nan padding for uneven timestamp lengths

    Finds mean framerate and generates a single timestamp series starting from 0 evenly spaced to the max timestamp.
    
    If trim=True, finds the largest of the leftmost timestamps and the smallest of the rightmost timestamps.
    If min_time or max_time is defined, that value is used regardless of trim.
    '''
    if not isinstance(timestamps, np.ndarray):
        timestamps = uneven_list2d_to_np(timestamps)
    mean_diff = np.nanmean(np.diff(timestamps))
    if trim:
        min_time = np.nanmax(np.nanmin(timestamps,axis=1)) if min_time is None else min_time
        max_time = np.nanmin(np.nanmax(timestamps,axis=1)) if max_time is None else max_time
    else:
        min_time = np.nanmin(timestamps) if min_time is None else min_time
        max_time = np.nanmax(timestamps) if max_time is None else max_time

    if min_time <= 0 and max_time >= 0:
        left = np.flip(np.arange(0, min_time, -mean_diff))
        right = np.arange(0, max_time, mean_diff)
        ts_standard = np.concatenate((left[:-1], right))
    else:
        ts_standard = np.arange(min_time, max_time, mean_diff)

    return ts_standard

def interpolate_to_new_timestamp(y, t, nt):
    '''
    y: 1d data, length same as t
    t: original timestamp
    nt: new timestamp to interpolate to
    Returns ny, linearly interpolated data at nt
    '''
    not_nan = ~np.isnan(y)
    return np.interp(nt, t[not_nan], y[not_nan], left=np.nan, right=np.nan)


def align_traces_to_standardized_timestamp(ts, xs, ts_standard=None, trim=False, min_time=None, max_time=None):
    if ts_standard is None:
        ts_standard = generate_standard_timestamp(ts, trim=trim, min_time=min_time, max_time=max_time)
    xs_standardized = np.array([interpolate_to_new_timestamp(xs[i], ts[i], ts_standard) for i in range(len(xs))])

    return ts_standard, xs_standardized

def convert_iterables_to_tuples(l, recursive=False, exclude_strings=True):
    '''
    l: iterable of items, including iterables

    Converts all iterables inside to tuples
    '''
    if recursive:
        return [tuple(convert_iterables_to_tuples(i, recursive)) if isinstance(i, collections.Iterable) else i for i in l]
    else:
        if exclude_strings:
            return [tuple(i) if isinstance(i, collections.Iterable) and not isinstance(i, str) else i for i in l]
        else:
            return [tuple(i) if isinstance(i, collections.Iterable) else i for i in l]
