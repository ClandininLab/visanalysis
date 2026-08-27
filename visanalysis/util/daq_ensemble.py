"""
Ensemble-aware segmentation of a single Jackfish DAQ recording into per-series blocks.

Stimpack "ensemble" recordings run two or more protocol runs back-to-back without user
interruption.  Stimpack writes each run as its own hdf5 series and FicTrac writes each run
as its own .dat, but Jackfish is NOT split up: one .jfdaqdata file covers every run in the
ensemble.  This module decides where that one file should be cut and which series owns each
piece, using several independent signals and refusing to guess when they disagree.

The module is deliberately hdf5-write-free and does not parse the 1 GB DAQ body itself, so
the segmentation logic can be unit tested against small synthetic arrays.

Signals used, strongest first:
    1. Camera strobe gaps.  A camera that is restarted between runs leaves a multi-second
       dead interval in an otherwise metronomic pulse train.  On the reference recording the
       gap is 3.0084 s against a 3.3 ms frame period - a separation ratio of 1253.
    2. FicTrac .dat line counts.  FicTrac writes exactly one line per delivered camera frame,
       so the per-series line counts *are* the per-block strobe counts, up to a small surplus.
    3. Stimpack run_start/run_end/epoch unix times, which say how many series should share a
       DAQ file and how long each block should be.
    4. The FicTrac nanosecond clock, which is continuous across the whole ensemble and so
       independently predicts the boundary position.
    5. The photodiode stimulus count, used only to audit a cut after the fact.

https://github.com/ClandininLab/visanalysis
"""
import glob
import hashlib
import json
import os
from collections import OrderedDict
from dataclasses import dataclass, field

import h5py
import numpy as np

from visanalysis.util import h5io


###############################################################################
# Exceptions
###############################################################################

class EnsembleError(RuntimeError):
    """Base class. Carries .report, a dict of every measured number for the failing check."""

    def __init__(self, message, report=None):
        super().__init__(message)
        self.report = report if report is not None else {}


class EnsembleGroupingError(EnsembleError):
    """Which series share a DAQ file could not be decided."""


class EnsembleCutError(EnsembleError):
    """Where to cut the DAQ file could not be decided."""


class EnsembleAssociationError(EnsembleError):
    """Which block belongs to which series could not be decided."""


class EnsembleValidationError(EnsembleError):
    """A post-plan check failed."""


###############################################################################
# Thresholds.  Every one of these is overridable via attachData(thresholds={...}).
###############################################################################

# --- ensemble grouping, from hdf5 metadata only ---
ENSEMBLE_TURNAROUND_MAX_S = 5.0    # run_end[k] -> run_start[k+1]; measured 0.0181 s
ENSEMBLE_TURNAROUND_MIN_S = -0.5   # small negative slop for clock jitter
DAQ_CAPACITY_SLOP_S = 1.0          # DAQ may fall short of the metadata span by this much
DAQ_EXCESS_NOTE_S = 60.0           # unused DAQ beyond this is reported, never fatal

# --- gap detection, on merged rise+fall edges ---
GAP_ABS_MIN_S = 0.25
GAP_RATIO_P99 = 10.0               # x p99 of the merged-edge interval distribution
GAP_MIN_PERIODS = 50.0             # x median rise-to-rise interval; keeps the floor rate-relative
SEGMENTER_MIN_PULSES = 100
SEGMENTER_MIN_RATE_HZ = 20.0
MIN_SEPARATION_RATIO = 10.0
MIN_BLOCK_S = 5.0
MIN_BLOCK_PULSES = 100
CUT_SNAP_WINDOW_MULT = 2.0

# --- counts ---
SURPLUS_MAX = 2                    # a block may exceed its .dat line count by this many strobes
SURPLUS_MIN = 0                    # a block may never have FEWER strobes than .dat lines

# --- association / validation ---
BLOCK_DURATION_TOL_S = 0.50
BLOCK_DURATION_TOL_FRAC = 0.002
UNIQUENESS_FACTOR = 4.0
GAP_MATCH_TOL_S = 0.50
T0_SPREAD_MAX_S = 0.25
FICTRAC_CLOCK_GAP_TOL_SAMP = 2.0
FICTRAC_CLOCK_SPAN_TOL_S = 0.100
FICTRAC_CLOCK_SPAN_TOL_PPM = 100e-6

# --- photodiode audit; corroboration only ---
PD_MATCH_TOL_S = 0.50
PD_OVERSHOOT_MAX = 2               # detected > n_epochs + this is the mis-slice signature -> abort

DEFAULT_THRESHOLDS = {
    k: v for k, v in list(globals().items())
    if k.isupper() and not k.startswith('_') and isinstance(v, (int, float))
}


def resolveThresholds(thresholds=None):
    """Merge a user override dict over the module defaults. Unknown keys raise."""
    out = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        unknown = set(thresholds) - set(out)
        if unknown:
            raise ValueError('Unknown threshold(s): {}. Known: {}'.format(
                sorted(unknown), sorted(out)))
        out.update(thresholds)
    return out


###############################################################################
# Data classes
###############################################################################

@dataclass(frozen=True)
class SeriesEvidence:
    """Everything known about one series, from the hdf5 and from the filesystem."""
    series_number: int
    subject_id: str
    n_epoch_groups: int
    num_epochs: int = None
    num_epochs_completed: int = None
    run_status: str = None
    run_start_unix: float = None
    run_end_unix: float = None
    first_epoch_unix: float = None
    last_epoch_end_unix: float = None
    epoch_onset_unix: np.ndarray = None
    epoch_offset_unix: np.ndarray = None
    pre_time: float = None
    tail_time: float = None
    do_loco: bool = True
    series_dir: str = None
    daq_path: str = None
    fictrac_dat_path: str = None
    n_fictrac_lines: int = None       # None means UNKNOWN. Never coerce to 0.
    log_path: str = None
    n_log_lines: int = None

    @property
    def epoch_span_s(self):
        if self.first_epoch_unix is None or self.last_epoch_end_unix is None:
            return None
        return self.last_epoch_end_unix - self.first_epoch_unix


@dataclass(frozen=True)
class Block:
    """One contiguous piece of the DAQ sample axis, [sample_lo, sample_hi)."""
    sample_lo: int
    sample_hi: int
    strobe_lo: int
    strobe_hi: int
    gap_before_s: float = None
    gap_after_s: float = None

    @property
    def n_samples(self):
        return self.sample_hi - self.sample_lo

    @property
    def n_strobes(self):
        return self.strobe_hi - self.strobe_lo


@dataclass
class DaqPlan:
    """A complete, validated decision about how one DAQ file is split and assigned."""
    daq_path: str
    daq_relpath: str
    sample_rate: float
    n_samples: int
    channel_nicknames: list
    segmenter: str = None
    method: str = 'single'
    blocks: list = field(default_factory=list)
    series_for_block: list = field(default_factory=list)
    cut_samples: list = field(default_factory=list)
    cut_uncertainty_samples: int = 0
    cut_snap_distance_samples: list = field(default_factory=list)
    separation_ratio: float = float('inf')
    t0_unix: float = None
    t0_spread_s: float = None
    t0_anchors: list = field(default_factory=list)
    checks: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    quality: str = 'validated'
    diagnostics: dict = field(default_factory=dict)

    def addCheck(self, name, status, measured=None, tolerance=None, message=''):
        self.checks.append({'name': name, 'status': status, 'measured': measured,
                            'tolerance': tolerance, 'message': message})

    def warn(self, message):
        if message not in self.warnings:
            self.warnings.append(message)

    @property
    def n_blocks(self):
        return len(self.blocks)

    @property
    def failed_checks(self):
        return [c for c in self.checks if c['status'] == 'fail']


###############################################################################
# Cheap evidence gathering.  Nothing here parses the DAQ body.
###############################################################################

def countLines(path, chunk=1 << 23):
    """Count lines without loading the file. Adds 1 when the file lacks a trailing newline."""
    n = 0
    last = b'\n'
    with open(path, 'rb') as fh:
        while True:
            buf = fh.read(chunk)
            if not buf:
                break
            n += buf.count(b'\n')
            last = buf[-1:]
    if last not in (b'\n', b''):
        n += 1
    return n


def probeDaqFile(path):
    """Read the header line and count samples, without parsing the body."""
    with open(path, 'r') as jf:
        header = jf.readline()
    has_header = header.strip().startswith('{')
    if has_header:
        meta = json.loads(header)
        sample_rate = float(meta['scan_rate'])
        nicknames = list(meta['input_channels'].values())
    else:
        # Legacy blank-header format; getVoltageRecording hard-codes this same map.
        sample_rate = 5000.0
        nicknames = ['frame_monitor_R', 'frame_monitor_C', 'frame_monitor_L',
                     'cam_strobe_Fictrac', 'cam_strobe_Top', 'cam_strobe_Left']
    n_lines = countLines(path)
    n_samples = n_lines - 1 if has_header else None
    return {'sample_rate': sample_rate,
            'input_ch_nicknames': nicknames,
            'n_samples': n_samples,
            'duration_s': None if n_samples is None else n_samples / sample_rate,
            'has_header': has_header}


def _attr(group, key, default=None):
    """Read an hdf5 attribute defensively; a missing attr yields default, never KeyError."""
    if key not in group.attrs:
        return default
    val = group.attrs[key]
    if isinstance(val, bytes):
        return val.decode('utf-8', 'replace')
    return val


def _scalar(val, default=None):
    """Coerce a possibly array-valued hdf5 attribute to a float, or default if not scalar."""
    if val is None:
        return default
    arr = np.ravel(val)
    if arr.size != 1:
        return default
    try:
        return float(arr[0])
    except (TypeError, ValueError):
        return default


def readSeriesEvidence(file_path, data_directory, series_numbers):
    """Collect hdf5 + filesystem evidence for each series in one read-only pass."""
    evidence = {}
    with h5py.File(file_path, 'r') as experiment_file:
        for subject_id in h5io.getSubjectIds(experiment_file):
            # Stimpack renamed 'epoch_runs' to 'series'; h5io resolves whichever is present.
            runs_group = h5io.getSeriesParentGroup(experiment_file, subject_id)
            if runs_group is None:
                continue
            for series_name in runs_group.keys():
                series_number = int(series_name.split('_')[-1])
                if series_number not in series_numbers:
                    continue
                group = runs_group[series_name]
                epochs = group['epochs'] if 'epochs' in group else {}
                epoch_names = sorted(epochs.keys()) if len(epochs) else []

                pre_time = _scalar(_attr(group, 'pre_time'), 0.0)
                tail_time = _scalar(_attr(group, 'tail_time'), 0.0)

                onsets, offsets = [], []
                for name in epoch_names:
                    ea = epochs[name].attrs
                    e_pre = _scalar(ea.get('pre_time'), pre_time)
                    e_tail = _scalar(ea.get('tail_time'), tail_time)
                    e_stim = _scalar(ea.get('stim_time'), None)
                    start = ea.get('epoch_unix_time')
                    if start is None:
                        onsets.append(np.nan)
                        offsets.append(np.nan)
                        continue
                    start = float(start)
                    end = ea.get('epoch_end_unix_time')
                    if end is None and e_stim is not None:
                        # Same fallback as twentyfourhourfitness.getStimulusTiming.
                        end = start + e_pre + e_stim + e_tail
                    onsets.append(start + (e_pre or 0.0))
                    offsets.append(np.nan if end is None else float(end) - (e_tail or 0.0))

                first_epoch = None
                last_epoch_end = None
                if epoch_names:
                    fa = epochs[epoch_names[0]].attrs
                    la = epochs[epoch_names[-1]].attrs
                    if 'epoch_unix_time' in fa:
                        first_epoch = float(fa['epoch_unix_time'])
                    if 'epoch_end_unix_time' in la:
                        last_epoch_end = float(la['epoch_end_unix_time'])
                    elif not np.isnan(offsets[-1]):
                        last_epoch_end = offsets[-1] + (_scalar(la.get('tail_time'), tail_time) or 0.0)

                series_dir = os.path.join(data_directory, str(series_number))
                if not os.path.isdir(series_dir):
                    series_dir = None

                daq_path = None
                if series_dir is not None:
                    found = sorted(glob.glob(os.path.join(series_dir, '*.jfdaqdata')))
                    if len(found) > 1:
                        raise EnsembleGroupingError(
                            'Series {} holds {} .jfdaqdata files; expected at most one: {}'.format(
                                series_number, len(found), [os.path.basename(f) for f in found]),
                            {'series_number': series_number, 'candidates': found})
                    daq_path = found[0] if found else None

                dat_path, n_lines, log_path, n_log = None, None, None, None
                if series_dir is not None:
                    loco_dir = os.path.join(series_dir, 'loco')
                    if os.path.isdir(loco_dir):
                        dats = sorted(glob.glob(os.path.join(loco_dir, '*.dat')))
                        if dats:
                            dat_path = dats[0]
                            n_lines = countLines(dat_path)
                        candidate_log = os.path.join(loco_dir, 'log.txt')
                        if os.path.exists(candidate_log):
                            log_path = candidate_log
                            n_log = countLines(log_path)

                do_loco = _attr(group, 'do_loco', True)
                if isinstance(do_loco, (str, np.str_)):
                    do_loco = do_loco.strip().lower() not in ('false', '0', 'no')
                do_loco = bool(do_loco)

                evidence[series_number] = SeriesEvidence(
                    series_number=series_number,
                    subject_id=str(subject_id),
                    n_epoch_groups=len(epoch_names),
                    num_epochs=_scalar(_attr(group, 'num_epochs')),
                    num_epochs_completed=_scalar(_attr(group, 'num_epochs_completed')),
                    run_status=_attr(group, 'run_status'),
                    run_start_unix=_scalar(_attr(group, 'run_start_unix_time')),
                    run_end_unix=_scalar(_attr(group, 'run_end_unix_time')),
                    first_epoch_unix=first_epoch,
                    last_epoch_end_unix=last_epoch_end,
                    epoch_onset_unix=np.asarray(onsets, dtype=float),
                    epoch_offset_unix=np.asarray(offsets, dtype=float),
                    pre_time=pre_time,
                    tail_time=tail_time,
                    do_loco=do_loco,
                    series_dir=series_dir,
                    daq_path=daq_path,
                    fictrac_dat_path=dat_path,
                    n_fictrac_lines=n_lines,
                    log_path=log_path,
                    n_log_lines=n_log)
    return evidence


def _fileFingerprint(path, nbytes=1 << 20):
    """(realpath, size, hash of head+tail) - enough to spot the same file reached two ways."""
    size = os.path.getsize(path)
    digest = hashlib.sha1()
    with open(path, 'rb') as fh:
        digest.update(fh.read(nbytes))
        if size > nbytes:
            fh.seek(max(0, size - nbytes))
            digest.update(fh.read(nbytes))
    return (os.path.realpath(path), size, digest.hexdigest())


def findDaqCandidates(data_directory, series_numbers):
    """Every distinct .jfdaqdata reachable from the date directory, deduped by content."""
    paths = []
    for series_number in sorted(series_numbers):
        paths.extend(sorted(glob.glob(
            os.path.join(data_directory, str(series_number), '*.jfdaqdata'))))
    paths.extend(sorted(glob.glob(os.path.join(data_directory, '*.jfdaqdata'))))

    seen, candidates = {}, []
    for path in paths:
        fp = _fileFingerprint(path)
        if fp in seen:
            continue
        seen[fp] = path
        candidates.append({'path': path,
                           'relpath': os.path.relpath(path, data_directory),
                           'fingerprint': fp})
    return candidates


###############################################################################
# Grouping: which series share a DAQ file
###############################################################################

def buildDaqOwnership(evidence, candidates, thresholds=None):
    """Grow ensembles forward from each DAQ file using the hdf5 turnaround + capacity gates.

    Returns (ownership {daq_path: [series...]}, series_without_daq, notes).

    Built FROM the metadata rather than validated against it afterwards, so that a genuinely
    missing DAQ file degrades to the existing "no DAQ data" warning for that one series
    instead of failing the whole date.
    """
    th = resolveThresholds(thresholds)
    notes = []
    series_numbers = sorted(evidence)

    starts = [(sn, evidence[sn].run_start_unix) for sn in series_numbers
              if evidence[sn].run_start_unix is not None]
    for (sn_a, t_a), (sn_b, t_b) in zip(starts, starts[1:]):
        if t_b < t_a:
            raise EnsembleGroupingError(
                'Series numbers do not ascend with run_start_unix_time: series {} starts '
                '{:.3f} s after series {}.'.format(sn_a, t_a - t_b, sn_b),
                {'series': [sn_a, sn_b], 'run_start_unix': [t_a, t_b]})

    probes = {}
    for cand in candidates:
        probes[cand['path']] = probeDaqFile(cand['path'])

    ownership = OrderedDict()
    series_without_daq = []
    open_anchor = None

    for idx, sn in enumerate(series_numbers):
        ev = evidence[sn]
        if ev.series_dir is None:
            # Preserve the existing message so log-scrapers keep working.
            notes.append('Series {} does not exist in directory strucutre. Skipping...'.format(sn))
            open_anchor = None
            continue

        if ev.daq_path is not None:
            ownership[ev.daq_path] = [sn]
            open_anchor = (ev.daq_path, sn)
            continue

        joined = False
        if open_anchor is not None:
            daq_path, anchor_sn = open_anchor
            prev_sn = ownership[daq_path][-1]
            prev, anchor = evidence[prev_sn], evidence[anchor_sn]

            turnaround = None
            if ev.run_start_unix is not None and prev.run_end_unix is not None:
                turnaround = ev.run_start_unix - prev.run_end_unix
            g1 = (turnaround is not None
                  and th['ENSEMBLE_TURNAROUND_MIN_S'] <= turnaround <= th['ENSEMBLE_TURNAROUND_MAX_S'])

            duration = probes[daq_path]['duration_s']
            needed = None
            if (ev.last_epoch_end_unix is not None and anchor.first_epoch_unix is not None):
                needed = ev.last_epoch_end_unix - anchor.first_epoch_unix
            g2 = (needed is not None and duration is not None
                  and needed <= duration + th['DAQ_CAPACITY_SLOP_S'])

            if g1 and g2:
                ownership[daq_path].append(sn)
                joined = True
                notes.append(
                    'Series {} joins {} (turnaround {:.4f} s <= {:.1f}; needs {:.3f} s of '
                    '{:.3f} s available).'.format(sn, os.path.basename(daq_path), turnaround,
                                                  th['ENSEMBLE_TURNAROUND_MAX_S'], needed, duration))
            elif turnaround is not None:
                notes.append(
                    'Series {} does NOT join {}: turnaround {:.4f} s (gate {:.1f} s), '
                    'capacity ok={}.'.format(sn, os.path.basename(daq_path), turnaround,
                                             th['ENSEMBLE_TURNAROUND_MAX_S'], g2))

        if not joined:
            open_anchor = None
            series_without_daq.append(sn)

    return ownership, series_without_daq, notes


###############################################################################
# Edges and gaps
###############################################################################

def findStrobeEdges(voltage_recording, input_ch_nicknames, prefix='cam_strobe'):
    """Rising/falling edge sample indices per camera channel, computed once on the full trace.

    Uses arithmetic identical to the existing plugin loop so that the counts written to the
    hdf5 are guaranteed consistent with what a reader would recompute.
    """
    edges = OrderedDict()
    for idx, nickname in enumerate(input_ch_nicknames):
        if not nickname.startswith(prefix):
            continue
        diff = np.diff(voltage_recording[idx, :])
        rise = np.nonzero(diff == 1)[0] + 1
        fall = np.nonzero(diff == -1)[0] + 1
        edges[nickname] = (rise, fall)
    return edges


def gapThreshold(rise, fall, sample_rate, thresholds=None):
    """Interval above which a merged-edge gap counts as a camera stoppage.

    Three terms: an absolute floor, a distribution-relative term for jittery channels, and a
    rate-relative term so a slow camera dropping a few frames is not mistaken for a stoppage.
    """
    th = resolveThresholds(thresholds)
    merged = np.sort(np.concatenate([rise, fall]))
    if merged.size < 3:
        return float('inf'), {'n_edges': int(merged.size)}
    intervals = np.diff(merged)
    p99 = float(np.percentile(intervals, 99))
    median_period = float(np.median(np.diff(rise))) if rise.size > 1 else 0.0
    threshold = max(th['GAP_ABS_MIN_S'] * sample_rate,
                    th['GAP_RATIO_P99'] * p99,
                    th['GAP_MIN_PERIODS'] * median_period)
    diag = {'merged_median': float(np.median(intervals)), 'merged_p99': p99,
            'median_rise_period': median_period, 'threshold_samples': float(threshold),
            'max_interval': float(intervals.max())}
    return threshold, diag


def findEdgeGaps(rise, fall, sample_rate, thresholds=None):
    """Locate camera stoppages on the MERGED, sorted rise+fall edge array.

    Merged edges are used rather than rise-to-rise because this rig has a documented
    inverted-strobe camera; for an inverted channel `rise` is the exposure *end*, and the
    obvious "close of the previous pulse" expression lands inside the next block's first
    exposure. The merged form is inversion-agnostic and cannot index off the end of `fall`.
    """
    threshold, diag = gapThreshold(rise, fall, sample_rate, thresholds)
    merged = np.sort(np.concatenate([rise, fall]))
    gaps = []
    if merged.size >= 3 and np.isfinite(threshold):
        intervals = np.diff(merged)
        for i in np.nonzero(intervals > threshold)[0]:
            end_prev, start_next = int(merged[i]), int(merged[i + 1])
            gaps.append({'edge_index': int(i),
                         'end_prev': end_prev,
                         'start_next': start_next,
                         'dark_samples': start_next - end_prev,
                         'dark_s': (start_next - end_prev) / sample_rate,
                         'strobe_index': int(np.searchsorted(rise, start_next)),
                         'cut_sample': (end_prev + start_next) // 2})
    diag['n_gaps'] = len(gaps)
    return gaps, diag


def chooseSegmenter(edges, sample_rate, expected_lines_total, n_series, thresholds=None):
    """Pick the camera channel whose pulse train defines the block boundaries.

    Count match is decisive: only the camera with a 1:1 frame contract against FicTrac can
    have a pulse count equal to the summed .dat line counts. A free-running camera
    self-identifies by having no gaps and is never a segmenter.
    """
    th = resolveThresholds(thresholds)
    diag = {'channels': {}, 'expected_lines_total': expected_lines_total}
    eligible = []

    for nickname, (rise, fall) in edges.items():
        info = {'n_rise': int(rise.size), 'n_fall': int(fall.size)}
        if rise.size < th['SEGMENTER_MIN_PULSES']:
            info['eligible'] = False
            info['reason'] = 'fewer than {} pulses'.format(int(th['SEGMENTER_MIN_PULSES']))
            diag['channels'][nickname] = info
            continue
        median_period = float(np.median(np.diff(rise)))
        rate = sample_rate / median_period if median_period > 0 else 0.0
        info['rate_hz'] = rate
        if rate < th['SEGMENTER_MIN_RATE_HZ']:
            info['eligible'] = False
            info['reason'] = 'rate {:.2f} Hz below {:.1f} Hz'.format(rate, th['SEGMENTER_MIN_RATE_HZ'])
            diag['channels'][nickname] = info
            continue
        if abs(rise.size - fall.size) > 1:
            info['eligible'] = False
            info['reason'] = 'rise/fall counts differ by {}'.format(abs(rise.size - fall.size))
            diag['channels'][nickname] = info
            continue
        gaps, gdiag = findEdgeGaps(rise, fall, sample_rate, thresholds)
        info.update(gdiag)
        info['eligible'] = True
        diag['channels'][nickname] = info
        eligible.append((nickname, rise, fall, gaps, info))

    if not eligible:
        raise EnsembleCutError('No camera channel is eligible to segment this DAQ file.', diag)

    # Primary: pulse count matches the summed FicTrac line counts within the surplus budget.
    if expected_lines_total is not None:
        budget = th['SURPLUS_MAX'] * max(1, n_series)
        matches = [e for e in eligible
                   if th['SURPLUS_MIN'] <= (e[1].size - expected_lines_total) <= budget]
        for nickname, rise, _, _, info in eligible:
            info['count_delta'] = int(rise.size - expected_lines_total)
        if len(matches) == 1:
            diag['rule'] = 'count_match'
            return matches[0][0], diag
        if len(matches) > 1:
            named = [m[0] for m in matches if 'fictrac' in m[0].lower()]
            diag['rule'] = 'count_match_tiebreak_name'
            diag['tied'] = [m[0] for m in matches]
            if named:
                return named[0], diag
            return matches[0][0], diag

    # Fallback, only when a .dat is missing: the channel whose gap count fits the group and
    # whose separation is cleanest.
    gapped = [e for e in eligible if len(e[3]) == max(0, n_series - 1) and len(e[3]) > 0]
    if len(gapped) >= 1:
        if len(gapped) > 1:
            spans = []
            for nickname, _, _, gaps, _ in gapped:
                spans.append((nickname, min(g['end_prev'] for g in gaps),
                              max(g['start_next'] for g in gaps)))
            for (n_a, lo_a, hi_a), (n_b, lo_b, hi_b) in zip(spans, spans[1:]):
                if hi_a < lo_b or hi_b < lo_a:
                    raise EnsembleCutError(
                        'Gapped channels disagree on where the boundary is: {} vs {}.'.format(
                            n_a, n_b), diag)
        best = max(gapped, key=lambda e: e[4].get('max_interval', 0) / max(1.0, e[4].get('merged_p99', 1.0)))
        diag['rule'] = 'gap_count'
        return best[0], diag

    if n_series <= 1:
        preferred = [e for e in eligible if 'fictrac' in e[0].lower()]
        diag['rule'] = 'single_series_default'
        return (preferred or eligible)[0][0], diag

    raise EnsembleCutError(
        'Could not choose a segmenter channel for a {}-series group.'.format(n_series), diag)


def snapCut(raw_cut, strobe_rows, window):
    """Move the cut to the nearest sample where EVERY camera channel is low on both sides.

    A cut landing inside a high pulse gives one block a fall with no rise, which for an
    interior block produces interleaved pairs that pass the existing +/-1 trim and yield
    negative exposure times. Snapping makes len(onset) == len(offset) exact, not approximate.
    """
    if not len(strobe_rows):
        return int(raw_cut), 0, True
    n = strobe_rows[0].size
    for distance in range(int(window) + 1):
        for candidate in ((raw_cut - distance, raw_cut + distance) if distance else (raw_cut,)):
            if candidate <= 0 or candidate >= n:
                continue
            if all(row[candidate - 1] == 0 and row[candidate] == 0 for row in strobe_rows):
                return int(candidate), int(distance), True
    return int(raw_cut), int(window), False


def _admissibleInteriorCuts(line_counts, n_strobes, thresholds=None):
    """Interior strobe indices consistent with the per-series FicTrac line counts.

    For N series the cut vector C satisfies SURPLUS_MIN <= (C[j+1]-C[j]) - L[j] <= SURPLUS_MAX
    for every block j. Returns the per-cut admissible ranges as (lo, hi) inclusive.
    """
    th = resolveThresholds(thresholds)
    n = len(line_counts)
    if n < 2 or any(c is None for c in line_counts):
        return None
    lo_bounds, hi_bounds = [], []
    cum = 0
    for j in range(n - 1):
        cum += line_counts[j]
        lo_bounds.append(cum + th['SURPLUS_MIN'] * (j + 1))
        hi_bounds.append(cum + th['SURPLUS_MAX'] * (j + 1))
    # Tighten from the right: whatever remains must still satisfy the trailing blocks.
    tail = 0
    for j in range(n - 1, 0, -1):
        tail += line_counts[j]
        idx = j - 1
        hi_bounds[idx] = min(hi_bounds[idx], n_strobes - tail - th['SURPLUS_MIN'] * (n - j))
        lo_bounds[idx] = max(lo_bounds[idx], n_strobes - tail - th['SURPLUS_MAX'] * (n - j))
    return list(zip([int(x) for x in lo_bounds], [int(x) for x in hi_bounds]))


def planCuts(edges, segmenter, sample_rate, n_samples, line_counts, strobe_rows,
             daq_path='', daq_relpath='', channel_nicknames=None,
             allow_counts_only=False, forced_cut_samples=None, thresholds=None):
    """Decide the block boundaries: counts propose, the merged-edge gap disposes."""
    th = resolveThresholds(thresholds)
    plan = DaqPlan(daq_path=daq_path, daq_relpath=daq_relpath, sample_rate=sample_rate,
                   n_samples=n_samples, channel_nicknames=list(channel_nicknames or []),
                   segmenter=segmenter)

    rise, fall = edges[segmenter] if segmenter in edges else (np.array([], int), np.array([], int))
    n_strobes = int(rise.size)
    n_series = len(line_counts)
    known_counts = [c for c in line_counts if c is not None]
    total_lines = sum(known_counts) if len(known_counts) == n_series else None

    if total_lines is not None:
        surplus_total = n_strobes - total_lines
        if surplus_total < th['SURPLUS_MIN'] * n_series:
            raise EnsembleCutError(
                'Segmenter {} has {} strobes but the FicTrac .dat files hold {} lines; there '
                'are fewer camera exposures than logged frames.'.format(
                    segmenter, n_strobes, total_lines),
                {'n_strobes': n_strobes, 'total_lines': total_lines})
        if surplus_total > th['SURPLUS_MAX'] * n_series:
            plan.warn('strobe_surplus_unexplained: {} strobes vs {} FicTrac lines (surplus {}, '
                      'budget {}).'.format(n_strobes, total_lines, surplus_total,
                                           int(th['SURPLUS_MAX'] * n_series)))

    # Explicit operator override wins over every inference.
    if forced_cut_samples:
        interior = sorted(int(c) for c in forced_cut_samples)
        plan.method = 'manual'
        plan.cut_samples = interior
        plan.cut_snap_distance_samples = [0] * len(interior)
        _finalizeBlocks(plan, interior, rise, sample_rate, n_samples, [])
        return plan

    # Single series: return before any count gate, so today's behaviour is untouched.
    if n_series <= 1:
        plan.method = 'single'
        _finalizeBlocks(plan, [], rise, sample_rate, n_samples, [])
        return plan

    gaps, gdiag = findEdgeGaps(rise, fall, sample_rate, thresholds)
    plan.diagnostics['segmenter_gaps'] = gdiag
    ranges = _admissibleInteriorCuts(line_counts, n_strobes, thresholds)
    plan.diagnostics['admissible_cut_ranges'] = ranges

    n_needed = n_series - 1
    chosen = None

    def admissible(strobe_indices):
        if ranges is None:
            return True
        return all(lo <= idx <= hi for idx, (lo, hi) in zip(strobe_indices, ranges))

    if len(gaps) == n_needed and n_needed > 0:
        if admissible([g['strobe_index'] for g in gaps]):
            chosen, plan.method = gaps, 'counts+gap'
        else:
            raise EnsembleCutError(
                'The camera gap and the FicTrac line counts disagree. Gap puts the boundary at '
                'strobe index {}, counts admit {}.'.format(
                    [g['strobe_index'] for g in gaps], ranges),
                {'gaps': gaps, 'admissible_ranges': ranges})
    elif len(gaps) > n_needed:
        import itertools
        survivors = [combo for combo in itertools.combinations(gaps, n_needed)
                     if admissible([g['strobe_index'] for g in combo])]
        if len(survivors) == 1:
            chosen, plan.method = list(survivors[0]), 'gap_subset'
            rejected = [g for g in gaps if g not in chosen]
            plan.warn('gap_subset: {} candidate gaps found, {} used; rejected dark intervals at '
                      't = {}.'.format(len(gaps), n_needed,
                                       [round(g['end_prev'] / sample_rate, 3) for g in rejected]))
        else:
            raise EnsembleCutError(
                '{} candidate camera gaps for {} boundaries; {} subsets are consistent with the '
                'FicTrac line counts (need exactly 1).'.format(len(gaps), n_needed, len(survivors)),
                {'gaps': gaps, 'admissible_ranges': ranges, 'n_survivors': len(survivors)})
    elif len(gaps) == 0 and allow_counts_only and ranges is not None:
        interior, uncertainty = [], 0
        for (lo, hi) in ranges:
            if lo < 1 or hi >= n_strobes:
                raise EnsembleCutError(
                    'counts_bracket: admissible strobe range [{}, {}] falls outside the pulse '
                    'train (0..{}).'.format(lo, hi, n_strobes - 1), {'ranges': ranges})
            interior.append(int((rise[lo - 1] + rise[hi]) // 2))
            uncertainty = max(uncertainty, int((rise[hi] - rise[lo - 1]) // 2))
        plan.method = 'counts_bracket'
        plan.cut_uncertainty_samples = uncertainty
        plan.warn('counts_bracket: no camera gap found; the cut is derived from FicTrac line '
                  'counts alone, with +/-{} samples ({:.4f} s) of uncertainty.'.format(
                      uncertainty, uncertainty / sample_rate))
        snapped, distances = _snapAll(interior, strobe_rows, edges, sample_rate, thresholds)
        plan.cut_samples = snapped
        plan.cut_snap_distance_samples = distances
        _finalizeBlocks(plan, snapped, rise, sample_rate, n_samples, [])
        return plan
    else:
        raise EnsembleCutError(
            'Found {} camera gap(s) on {} but need {} boundaries for {} series. Use '
            "allow_counts_only=True if every camera free-ran, or ensemble='off' to attach as "
            'one block.'.format(len(gaps), segmenter, n_needed, n_series),
            {'gaps': gaps, 'n_series': n_series, 'diag': gdiag})

    raw_cuts = [g['cut_sample'] for g in chosen]
    snapped, distances = _snapAll(raw_cuts, strobe_rows, edges, sample_rate, thresholds)
    plan.cut_samples = snapped
    plan.cut_snap_distance_samples = distances

    max_interval = gdiag.get('merged_p99', 1.0)
    smallest_gap = min(g['dark_samples'] for g in chosen)
    plan.separation_ratio = float(smallest_gap / max(1.0, max_interval))

    _finalizeBlocks(plan, snapped, rise, sample_rate, n_samples, chosen)
    return plan


def _snapAll(raw_cuts, strobe_rows, edges, sample_rate, thresholds=None):
    """Snap every candidate cut to an all-cameras-low sample; abort if none is reachable."""
    periods = [float(np.median(np.diff(r))) for r, _ in edges.values() if r.size > 1]
    window = int(np.ceil(CUT_SNAP_WINDOW_MULT * (max(periods) if periods else 1.0)))
    snapped, distances = [], []
    for raw in raw_cuts:
        cut, distance, ok = snapCut(raw, strobe_rows, window)
        if not ok:
            raise EnsembleCutError(
                'No sample within +/-{} of {} ({:.4f} s) has every camera strobe low; cutting '
                'there would split an exposure.'.format(window, raw, raw / sample_rate),
                {'raw_cut': int(raw), 'window': window})
        snapped.append(cut)
        distances.append(distance)
    return snapped, distances


def _finalizeBlocks(plan, interior_cuts, rise, sample_rate, n_samples, gaps):
    """Turn interior cuts into Blocks. Block 0 always starts at 0; the last ends at n_samples."""
    boundaries = [0] + list(interior_cuts) + [int(n_samples)]
    blocks = []
    for k in range(len(boundaries) - 1):
        lo, hi = boundaries[k], boundaries[k + 1]
        strobe_lo = int(np.searchsorted(rise, lo)) if rise.size else 0
        strobe_hi = int(np.searchsorted(rise, hi)) if rise.size else 0
        blocks.append(Block(sample_lo=lo, sample_hi=hi,
                            strobe_lo=strobe_lo, strobe_hi=strobe_hi,
                            gap_before_s=gaps[k - 1]['dark_s'] if (gaps and k > 0) else None,
                            gap_after_s=gaps[k]['dark_s'] if (gaps and k < len(gaps)) else None))
    plan.blocks = blocks


###############################################################################
# Slicing
###############################################################################

def sliceCamsTiming(edges, lo, hi, sample_rate):
    """Per-camera exposure times for one block, re-zeroed to the block start.

    Edge indices are selected by an absolute-sample mask, never by a positional slice of the
    edge array, and the same integer `lo` re-zeroes every camera and the photodiode trace.
    """
    cams_timing = {}
    for nickname, (rise, fall) in edges.items():
        onset_abs = rise[(rise >= lo) & (rise < hi)]
        offset_abs = fall[(fall >= lo) & (fall < hi)]

        if onset_abs.size == 0:
            continue

        n_straddling = 0
        # Legacy +/-1 trim, kept verbatim: it is what produces the documented inverted-camera
        # behaviour. With a snapped cut it is a no-op.
        if onset_abs.size > offset_abs.size:
            onset_abs = onset_abs[:-1]
            n_straddling += 1
        elif onset_abs.size < offset_abs.size:
            offset_abs = offset_abs[1:]
            n_straddling += 1

        if onset_abs.size == 0:
            continue

        exposure_onset = (onset_abs - lo) / sample_rate
        exposure_offset = (offset_abs - lo) / sample_rate
        exposure_times = exposure_offset - exposure_onset

        cams_timing[nickname] = {
            'exposure_onset': exposure_onset,
            'exposure_offset': exposure_offset,
            'exposure_time': float(np.mean(exposure_times)),
            'frame_rate': float(1 / np.mean(np.diff(exposure_onset))) if exposure_onset.size > 1 else float('nan'),
            'n_pulses': int(onset_abs.size),
            'first_pulse_sample': int(onset_abs[0]),
            'last_pulse_sample': int(offset_abs[-1]) if offset_abs.size else int(onset_abs[-1]),
            'n_straddling_pulses': int(n_straddling),
            'daq_sample_offset': int(lo),
        }
    return cams_timing


###############################################################################
# Association and validation
###############################################################################

def _blockStrobeSpan(edges, segmenter, block, sample_rate):
    rise, fall = edges[segmenter]
    r = rise[(rise >= block.sample_lo) & (rise < block.sample_hi)]
    f = fall[(fall >= block.sample_lo) & (fall < block.sample_hi)]
    if r.size == 0 or f.size == 0:
        return None, None, None
    return (f[-1] - r[0]) / sample_rate, int(r[0]), int(f[-1])


def associateBlocks(plan, evidence, series_candidates, edges, thresholds=None):
    """Map blocks to series. Both lists are time-ordered, so a count mismatch means the
    segmentation is wrong, not the ordering - hence an abort rather than a re-search."""
    th = resolveThresholds(thresholds)
    checks = []

    loco_series = [sn for sn in series_candidates if evidence[sn].do_loco]
    non_loco = [sn for sn in series_candidates if not evidence[sn].do_loco]
    if non_loco:
        first, last = series_candidates.index(loco_series[0]) if loco_series else 0, 0
        interior = [sn for sn in non_loco
                    if loco_series and min(loco_series) < sn < max(loco_series)]
        if interior:
            raise EnsembleAssociationError(
                'Series {} have do_loco=False but sit between loco series {}; their strobe-free '
                'interval would be swallowed into a neighbouring block.'.format(
                    interior, loco_series),
                {'non_loco': non_loco, 'loco': loco_series})

    if plan.n_blocks != len(loco_series):
        raise EnsembleAssociationError(
            'Segmentation produced {} camera block(s) but {} series in this group record '
            'locomotion ({}).'.format(plan.n_blocks, len(loco_series), loco_series),
            {'n_blocks': plan.n_blocks, 'loco_series': loco_series})

    series_for_block = list(loco_series)

    # A2, primary: per-block strobe count vs FicTrac .dat line count.
    #
    # Only meaningful when there is actually a cut to validate. With a single block the count
    # is whatever the DAQ happened to record - a camera that free-ran before the run starts or
    # after it ends produces a large surplus, which the pre-ensemble code simply printed and
    # moved past. Enforcing it here would stop legacy single-series files from attaching.
    counts_known = all(evidence[sn].n_fictrac_lines is not None for sn in series_for_block)
    if counts_known and plan.n_blocks == 1:
        sn = series_for_block[0]
        surplus = plan.blocks[0].n_strobes - evidence[sn].n_fictrac_lines
        checks.append({'name': 'A2_counts_series_{}'.format(sn), 'status': 'info',
                       'measured': surplus, 'tolerance': None,
                       'message': '{} strobes vs {} FicTrac lines (single block; not gated)'.format(
                           plan.blocks[0].n_strobes, evidence[sn].n_fictrac_lines)})
        if not (th['SURPLUS_MIN'] <= surplus <= th['SURPLUS_MAX']):
            plan.warn('Series {}: {} strobes vs {} FicTrac lines (surplus {:+d}) in a single '
                      'block.'.format(sn, plan.blocks[0].n_strobes,
                                      evidence[sn].n_fictrac_lines, surplus))
        return series_for_block, checks

    if counts_known:
        for block, sn in zip(plan.blocks, series_for_block):
            surplus = block.n_strobes - evidence[sn].n_fictrac_lines
            ok = th['SURPLUS_MIN'] <= surplus <= th['SURPLUS_MAX']
            checks.append({'name': 'A2_counts_series_{}'.format(sn),
                           'status': 'pass' if ok else 'fail',
                           'measured': surplus,
                           'tolerance': [th['SURPLUS_MIN'], th['SURPLUS_MAX']],
                           'message': '{} strobes vs {} FicTrac lines'.format(
                               block.n_strobes, evidence[sn].n_fictrac_lines)})
        if any(c['status'] == 'fail' for c in checks):
            raise EnsembleAssociationError(
                'Per-block strobe counts do not match the FicTrac line counts: {}'.format(
                    [(c['name'], c['measured']) for c in checks if c['status'] == 'fail']),
                {'checks': checks})
        return series_for_block, checks

    # A3/A4, fallback: duration fingerprint, with a uniqueness guard.
    plan.warn('At least one FicTrac .dat is missing; falling back to the duration fingerprint '
              'for block-to-series association.')
    spans = []
    for block in plan.blocks:
        span, _, _ = _blockStrobeSpan(edges, plan.segmenter, block, plan.sample_rate)
        spans.append(span)
    for block_idx, (span, sn) in enumerate(zip(spans, series_for_block)):
        expected = evidence[sn].epoch_span_s
        if span is None or expected is None:
            checks.append({'name': 'A3_span_series_{}'.format(sn), 'status': 'skipped',
                           'measured': None, 'tolerance': None,
                           'message': 'no strobe span or no epoch span available'})
            continue
        tol = max(th['BLOCK_DURATION_TOL_S'], th['BLOCK_DURATION_TOL_FRAC'] * expected)
        delta = abs(span - expected)
        checks.append({'name': 'A3_span_series_{}'.format(sn),
                       'status': 'pass' if delta <= tol else 'fail',
                       'measured': delta, 'tolerance': tol,
                       'message': 'strobe span {:.4f} s vs epoch span {:.4f} s'.format(span, expected)})
        for other in series_for_block:
            if other == sn:
                continue
            other_span = evidence[other].epoch_span_s
            if other_span is None:
                continue
            if abs(span - other_span) <= th['UNIQUENESS_FACTOR'] * tol:
                raise EnsembleAssociationError(
                    'Block {} ({:.3f} s) is ambiguous between series {} ({:.3f} s) and series '
                    '{} ({:.3f} s).'.format(block_idx, span, sn, expected, other, other_span),
                    {'checks': checks})
    if any(c['status'] == 'fail' for c in checks):
        raise EnsembleAssociationError(
            'Block durations do not match the series epoch spans: {}'.format(
                [(c['name'], c['measured']) for c in checks if c['status'] == 'fail']),
            {'checks': checks})
    return series_for_block, checks


def computeT0(evidence, plan, edges):
    """Estimate unix time of DAQ sample 0 from every block edge, and report the spread."""
    anchors = []
    rise, fall = edges[plan.segmenter]
    for block, sn in zip(plan.blocks, plan.series_for_block):
        ev = evidence.get(sn)
        if ev is None:
            continue
        r = rise[(rise >= block.sample_lo) & (rise < block.sample_hi)]
        f = fall[(fall >= block.sample_lo) & (fall < block.sample_hi)]
        if r.size and ev.first_epoch_unix is not None:
            anchors.append(ev.first_epoch_unix - r[0] / plan.sample_rate)
        if f.size and ev.last_epoch_end_unix is not None:
            anchors.append(ev.last_epoch_end_unix - f[-1] / plan.sample_rate)
    if not anchors:
        return None, None, []
    anchors = [float(a) for a in anchors]
    return float(np.mean(anchors)), float(max(anchors) - min(anchors)), anchors


def readFictracBoundaryClock(evidence, series_a, series_b):
    """Seconds between the last FicTrac frame of one series and the first of the next.

    FicTrac's nanosecond timestamp column is monotonic across an entire ensemble, so this is
    an independent prediction of the camera dead time, from a clock the DAQ never sees.
    """
    ts_col = 21  # 'timestamp', nanoseconds
    path_a = evidence[series_a].fictrac_dat_path
    path_b = evidence[series_b].fictrac_dat_path
    if not path_a or not path_b:
        return None
    try:
        with open(path_a, 'rb') as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - 65536))
            last_line = fh.read().splitlines()[-1].decode()
        with open(path_b, 'r') as fh:
            first_line = fh.readline()
        end_a = float(last_line.split(',')[ts_col])
        start_b = float(first_line.split(',')[ts_col])
    except (IndexError, ValueError, OSError):
        return None
    return (start_b - end_a) / 1e9


def readFictracClockSpan(path):
    """Seconds between the first and last FicTrac frame, from FicTrac's own ns clock.

    Independent of the DAQ, so a disagreement with the strobe span means one of the two
    clocks is not measuring the interval we think it is.
    """
    ts_col = 21  # 'timestamp', nanoseconds
    try:
        with open(path, 'r') as fh:
            first_line = fh.readline()
        with open(path, 'rb') as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - 65536))
            last_line = fh.read().splitlines()[-1].decode()
        return (float(last_line.split(',')[ts_col]) - float(first_line.split(',')[ts_col])) / 1e9
    except (IndexError, ValueError, OSError):
        return None


def validatePlan(plan, evidence, edges, strict=True, thresholds=None):
    """Run every hard and soft check. Nothing is written before this returns."""
    th = resolveThresholds(thresholds)
    fs = plan.sample_rate
    rise, fall = edges[plan.segmenter] if plan.segmenter in edges else (np.array([]), np.array([]))

    # V3 - block size floors
    for k, block in enumerate(plan.blocks):
        duration = block.n_samples / fs
        ok = duration >= th['MIN_BLOCK_S'] and block.n_strobes >= th['MIN_BLOCK_PULSES']
        plan.addCheck('V3_block{}_size'.format(k), 'pass' if ok else 'fail',
                      measured=[round(duration, 3), block.n_strobes],
                      tolerance=[th['MIN_BLOCK_S'], th['MIN_BLOCK_PULSES']],
                      message='block {} spans {:.3f} s with {} strobes'.format(
                          k, duration, block.n_strobes))

    # V4 - separation ratio
    if plan.n_blocks > 1 and np.isfinite(plan.separation_ratio):
        ok = plan.separation_ratio >= th['MIN_SEPARATION_RATIO']
        plan.addCheck('V4_separation_ratio', 'pass' if ok else 'fail',
                      measured=round(plan.separation_ratio, 1),
                      tolerance=th['MIN_SEPARATION_RATIO'],
                      message='smallest boundary gap vs p99 merged-edge interval')

    # V6 - strobe span vs epoch span
    for k, (block, sn) in enumerate(zip(plan.blocks, plan.series_for_block)):
        span, _, _ = _blockStrobeSpan(edges, plan.segmenter, block, fs)
        expected = evidence[sn].epoch_span_s
        if span is None or expected is None:
            plan.addCheck('V6_span_series_{}'.format(sn), 'skipped',
                          message='no epoch span recorded')
            continue
        tol = max(th['BLOCK_DURATION_TOL_S'], th['BLOCK_DURATION_TOL_FRAC'] * expected)
        delta = abs(span - expected)
        status = 'pass' if delta <= tol else 'fail'
        # A span check justifies a cut. With a single block there is no cut to justify, and the
        # DAQ legitimately runs longer than the series (lead-in, tail, or an un-segmented
        # ensemble under ensemble='off'), so it can only advise.
        if status == 'fail' and plan.n_blocks == 1:
            status = 'warn'
            plan.warn('Series {}: block spans {:.3f} s but the series covers {:.3f} s. Single '
                      'block, so not gated.'.format(sn, span, expected))
        elif evidence[sn].run_status not in (None, 'completed') and status == 'fail':
            status = 'warn'
            plan.warn('Series {} has run_status={!r}; span check downgraded to a warning.'.format(
                sn, evidence[sn].run_status))
        plan.addCheck('V6_span_series_{}'.format(sn), status,
                      measured=round(delta, 4), tolerance=round(tol, 4),
                      message='strobe span {:.4f} s vs epoch span {:.4f} s'.format(span, expected))

    # V7 - camera dead time vs the metadata dead time between consecutive series
    for k in range(plan.n_blocks - 1):
        sn_a, sn_b = plan.series_for_block[k], plan.series_for_block[k + 1]
        ev_a, ev_b = evidence[sn_a], evidence[sn_b]
        observed = plan.blocks[k].gap_after_s
        if observed is None or ev_a.last_epoch_end_unix is None or ev_b.first_epoch_unix is None:
            plan.addCheck('V7_gap_{}_{}'.format(sn_a, sn_b), 'skipped')
            continue
        expected = ev_b.first_epoch_unix - ev_a.last_epoch_end_unix
        delta = abs(observed - expected)
        plan.addCheck('V7_gap_{}_{}'.format(sn_a, sn_b),
                      'pass' if delta <= th['GAP_MATCH_TOL_S'] else 'fail',
                      measured=round(delta, 4), tolerance=th['GAP_MATCH_TOL_S'],
                      message='camera dark {:.4f} s vs metadata dead time {:.4f} s'.format(
                          observed, expected))

    # V8 - t0 anchor spread
    t0, spread, anchors = computeT0(evidence, plan, edges)
    plan.t0_unix, plan.t0_spread_s, plan.t0_anchors = t0, spread, anchors
    if spread is not None:
        status = 'pass' if spread <= th['T0_SPREAD_MAX_S'] else 'fail'
        # As with V6: a single block legitimately extends past its series at both ends, which
        # pushes the two anchors apart without saying anything about the segmentation.
        if status == 'fail' and plan.n_blocks == 1:
            status = 'warn'
            plan.warn('DAQ t=0 anchors span {:.3f} s; single block, so not gated.'.format(spread))
        plan.addCheck('V8_t0_spread', status,
                      measured=round(spread, 4), tolerance=th['T0_SPREAD_MAX_S'],
                      message='{} anchors for DAQ t=0'.format(len(anchors)))

    # V9 - FicTrac's own clock predicts the boundary independently of the DAQ
    for k in range(plan.n_blocks - 1):
        sn_a, sn_b = plan.series_for_block[k], plan.series_for_block[k + 1]
        predicted_s = readFictracBoundaryClock(evidence, sn_a, sn_b)
        if predicted_s is None:
            plan.addCheck('V9_fictrac_clock_{}_{}'.format(sn_a, sn_b), 'skipped',
                          message='FicTrac .dat timestamps unavailable')
            continue
        strobe_lo = plan.blocks[k].strobe_hi
        if strobe_lo < 1 or strobe_lo >= rise.size:
            plan.addCheck('V9_fictrac_clock_{}_{}'.format(sn_a, sn_b), 'skipped')
            continue
        observed_samples = float(rise[strobe_lo] - rise[strobe_lo - 2]) if strobe_lo >= 2 else None
        predicted_samples = predicted_s * fs
        if observed_samples is None:
            plan.addCheck('V9_fictrac_clock_{}_{}'.format(sn_a, sn_b), 'skipped')
            continue
        delta = abs(predicted_samples - observed_samples)
        plan.addCheck('V9_fictrac_clock_{}_{}'.format(sn_a, sn_b),
                      'pass' if delta <= th['FICTRAC_CLOCK_GAP_TOL_SAMP'] else 'fail',
                      measured=round(delta, 2), tolerance=th['FICTRAC_CLOCK_GAP_TOL_SAMP'],
                      message='FicTrac clock predicts {:.2f} samples, strobes give {:.0f}'.format(
                          predicted_samples, observed_samples))

    # V11 - the snap worked: every camera has matched edges inside every block.
    #
    # Exactness is only guaranteed at INTERIOR cuts, where snapCut placed the boundary on an
    # all-cameras-low sample. A block touching the start or end of the file may legitimately
    # hold one unmatched edge, because the DAQ can begin or end mid-exposure; sliceCamsTiming
    # drops it with the same +/-1 trim the pre-ensemble code used.
    for k, block in enumerate(plan.blocks):
        touches_file_edge = (block.sample_lo == 0 or block.sample_hi >= plan.n_samples)
        for nickname, (r, f) in edges.items():
            n_r = int(((r >= block.sample_lo) & (r < block.sample_hi)).sum())
            n_f = int(((f >= block.sample_lo) & (f < block.sample_hi)).sum())
            if n_r == 0 and n_f == 0:
                continue
            delta = n_r - n_f
            allowed = 1 if touches_file_edge else 0
            status = 'pass' if abs(delta) <= allowed else 'fail'
            if status == 'pass' and delta != 0:
                plan.warn('{} block {} starts or ends mid-exposure ({} rises vs {} falls); the '
                          'truncated pulse is trimmed.'.format(nickname, k, n_r, n_f))
            plan.addCheck('V11_edges_block{}_{}'.format(k, nickname), status,
                          measured=delta, tolerance=allowed,
                          message='{} rises vs {} falls{}'.format(
                              n_r, n_f, ' (file edge)' if touches_file_edge else ' (interior cut)'))

    # Soft checks
    total_span = 0.0
    for block, sn in zip(plan.blocks, plan.series_for_block):
        ev = evidence[sn]
        if ev.fictrac_dat_path:
            dat_span = readFictracClockSpan(ev.fictrac_dat_path)
            strobe_span, _, _ = _blockStrobeSpan(edges, plan.segmenter, block, fs)
            if dat_span is not None and strobe_span is not None:
                tol = max(th['FICTRAC_CLOCK_SPAN_TOL_S'],
                          th['FICTRAC_CLOCK_SPAN_TOL_PPM'] * strobe_span)
                delta = abs(dat_span - strobe_span)
                plan.addCheck('S_fictrac_clock_span_series_{}'.format(sn),
                              'pass' if delta <= tol else 'warn',
                              measured=round(delta, 4), tolerance=round(tol, 4),
                              message='FicTrac clock span {:.4f} s vs strobe span {:.4f} s '
                                      '({:.1f} ppm)'.format(dat_span, strobe_span,
                                                            1e6 * delta / max(strobe_span, 1e-9)))
                if delta > tol:
                    plan.warn('Series {}: FicTrac clock and camera strobes disagree on the block '
                              'duration by {:.4f} s.'.format(sn, delta))
        span = ev.epoch_span_s
        if span:
            total_span += span
    excess = plan.n_samples / fs - total_span
    if total_span and excess > th['DAQ_EXCESS_NOTE_S']:
        plan.addCheck('S_daq_excess', 'warn', measured=round(excess, 3),
                      tolerance=th['DAQ_EXCESS_NOTE_S'],
                      message='{:.1f} s of DAQ is not covered by any series in this group'.format(excess))
        plan.warn('{:.1f} s of {} is not accounted for by the series it was assigned to.'.format(
            excess, plan.daq_relpath))

    for sn in plan.series_for_block:
        ev = evidence[sn]
        if ev.n_log_lines is not None and ev.n_epoch_groups:
            if ev.n_log_lines != ev.n_epoch_groups:
                plan.warn('Series {}: log.txt has {} lines but the hdf5 has {} epochs.'.format(
                    sn, ev.n_log_lines, ev.n_epoch_groups))
        if ev.run_status not in (None, 'completed'):
            plan.warn('Series {} run_status={!r}.'.format(sn, ev.run_status))

    failures = plan.failed_checks
    if failures:
        if strict:
            raise EnsembleValidationError(
                'Validation failed for {}: {}'.format(
                    plan.daq_relpath,
                    '; '.join('{} (measured {}, tolerance {})'.format(
                        c['name'], c['measured'], c['tolerance']) for c in failures)),
                {'checks': plan.checks})
        plan.quality = 'forced'
        for check in failures:
            plan.warn('FORCED past failing check {}: measured {}, tolerance {}.'.format(
                check['name'], check['measured'], check['tolerance']))
    elif plan.warnings:
        plan.quality = 'validated_with_warnings'
    else:
        plan.quality = 'validated'
    return plan


###############################################################################
# Overrides
###############################################################################

def loadEnsembleOverride(spec, data_directory):
    """Load an ensemble override from a dict, an explicit path, or <date>/ensemble.json."""
    if spec is None:
        default = os.path.join(data_directory, 'ensemble.json')
        if not os.path.exists(default):
            return {}
        spec = default
    if isinstance(spec, dict):
        payload = spec
    else:
        with open(spec, 'r') as fh:
            payload = json.load(fh)

    out = {}
    for entry in payload.get('daq_assignments', []):
        if not entry.get('reason'):
            raise EnsembleError(
                "ensemble override for {!r} must carry a non-empty 'reason'.".format(
                    entry.get('daq_file')))
        key = os.path.normpath(os.path.join(data_directory, entry['daq_file']))
        out[key] = entry
    return out


###############################################################################
# Reporting
###############################################################################

def formatPlanReport(plan, evidence, segmenter_diag=None, pd_audit=None):
    """Human-readable plan summary, printed before anything is written."""
    lines = []
    lines.append('=== DAQ ENSEMBLE PLAN: {} ==='.format(plan.daq_relpath))
    lines.append('  {} samples @ {:.1f} Hz = {:.3f} s'.format(
        plan.n_samples, plan.sample_rate, plan.n_samples / plan.sample_rate))
    if segmenter_diag:
        for nickname, info in segmenter_diag.get('channels', {}).items():
            if not info.get('eligible'):
                lines.append('    {:22s} {:>7} pulses  -- {}'.format(
                    nickname, info.get('n_rise', 0), info.get('reason', 'not eligible')))
            else:
                tag = '[SEGMENTER]' if nickname == plan.segmenter else (
                    '[free-running]' if info.get('n_gaps') == 0 else '')
                lines.append('    {:22s} {:>7} pulses  {:8.3f} Hz  thr {:>6.0f}  gaps {}  {}'.format(
                    nickname, info.get('n_rise', 0), info.get('rate_hz', float('nan')),
                    info.get('threshold_samples', 0), info.get('n_gaps', 0), tag))
    lines.append('  METHOD  {}   blocks {}'.format(plan.method, plan.n_blocks))
    if plan.cut_samples:
        lines.append('  CUT     interior cuts at samples {} (t = {})'.format(
            plan.cut_samples, [round(c / plan.sample_rate, 4) for c in plan.cut_samples]))
        lines.append('          snap distance {} sample(s), separation ratio {:.1f}x'.format(
            plan.cut_snap_distance_samples, plan.separation_ratio))
    if plan.cut_uncertainty_samples:
        lines.append('          cut uncertainty +/-{} samples ({:.4f} s)'.format(
            plan.cut_uncertainty_samples, plan.cut_uncertainty_samples / plan.sample_rate))
    for k, (block, sn) in enumerate(zip(plan.blocks, plan.series_for_block)):
        ev = evidence.get(sn)
        lines.append('  BLOCK {} -> series {}  samples [{}, {})  {:.3f} s  {} strobes{}'.format(
            k, sn, block.sample_lo, block.sample_hi, block.n_samples / plan.sample_rate,
            block.n_strobes,
            '  (.dat {} lines, surplus {:+d})'.format(
                ev.n_fictrac_lines, block.n_strobes - ev.n_fictrac_lines)
            if ev is not None and ev.n_fictrac_lines is not None else ''))
    if plan.t0_unix is not None:
        lines.append('  T0      {:.6f} unix   spread {:.4f} s over {} anchors'.format(
            plan.t0_unix, plan.t0_spread_s, len(plan.t0_anchors)))
    for check in plan.checks:
        if check['status'] in ('fail', 'warn'):
            lines.append('  {:7s} {:28s} measured {} (tol {})  {}'.format(
                check['status'].upper(), check['name'], check['measured'],
                check['tolerance'], check['message']))
    n_pass = sum(1 for c in plan.checks if c['status'] == 'pass')
    lines.append('  CHECKS  {} passed, {} failed, {} skipped'.format(
        n_pass, len(plan.failed_checks),
        sum(1 for c in plan.checks if c['status'] == 'skipped')))
    if pd_audit:
        for sn, entry in pd_audit.items():
            lines.append('  PD AUDIT series {}: causal {} | zero-phase {}  (of {} epochs)'.format(
                sn, entry['counts_causal'], entry['counts_zero_phase'], entry['n_epochs']))
        recommended = [e for e in pd_audit.values() if e.get('recommended_channel') is not None]
        if recommended:
            entry = recommended[0]
            lines.append("          recommend cfg_dict={{'timing_channel_ind': {}, "
                         "'highpass_zero_phase': {}}}".format(
                             entry['recommended_channel'], entry['recommended_zero_phase']))
    for warning in plan.warnings:
        lines.append('  WARN    {}'.format(warning))
    lines.append('  QUALITY {}'.format(plan.quality))
    return '\n'.join(lines)


def planToDict(plan):
    """JSON-serializable form of a plan, for the hdf5 attr and the sidecar report."""
    return {
        'daq_file': plan.daq_relpath,
        'sample_rate': plan.sample_rate,
        'n_samples': plan.n_samples,
        'segmenter': plan.segmenter,
        'method': plan.method,
        'n_blocks': plan.n_blocks,
        'series_for_block': list(plan.series_for_block),
        'blocks': [{'sample_lo': b.sample_lo, 'sample_hi': b.sample_hi,
                    'strobe_lo': b.strobe_lo, 'strobe_hi': b.strobe_hi,
                    'gap_before_s': b.gap_before_s, 'gap_after_s': b.gap_after_s}
                   for b in plan.blocks],
        'cut_samples': list(plan.cut_samples),
        'cut_snap_distance_samples': list(plan.cut_snap_distance_samples),
        'cut_uncertainty_samples': plan.cut_uncertainty_samples,
        'separation_ratio': plan.separation_ratio,
        't0_unix': plan.t0_unix,
        't0_spread_s': plan.t0_spread_s,
        'checks': plan.checks,
        'warnings': plan.warnings,
        'quality': plan.quality,
    }
