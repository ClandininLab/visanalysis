"""
40 Hour Fitness rig plugin.

Attaches Jackfish DAQ data (stimulus photodiodes + camera strobes) and FicTrac data to a
stimpack hdf5 file.

Handles stimpack "ensemble" recordings, in which two or more protocol runs execute back to
back without user interruption.  Stimpack writes each run as its own series and FicTrac
writes each run as its own .dat, but Jackfish is not split up - one .jfdaqdata file covers
the whole ensemble.  The segmentation logic lives in visanalysis.util.daq_ensemble; this
module is the driver that parses each DAQ file once, slices it, and writes the pieces.

https://github.com/ClandininLab/visanalysis
"""
import os
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import h5py
import skimage.io as io
import functools
import nibabel as nib
import json
import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.signal as signal

from visanalysis.plugin import base as base_plugin
from visanalysis.util import h5io
from visanalysis.util import general_utils as gu
from visanalysis.util import daq_ensemble as de

from visanalysis.analysis.imaging_data import ImagingDataObject
from visanalysis.plugin.twentyfourhourfitness import TwentyFourHourDataObject

FICTRAC_DATA_HEADER = ['frame_count',
                       'rel_vec_cam_x', 'rel_vec_cam_y', 'rel_vec_cam_z', 'error',
                       'rel_vec_world_x', 'rel_vec_world_y', 'rel_vec_world_z',
                       'abs_vec_cam_x', 'abs_vec_cam_y', 'abs_vec_cam_z',
                       'abs_vec_world_x', 'abs_vec_world_y', 'abs_vec_world_z',
                       'integrated_xpos', 'integrated_ypos', 'integrated_heading',
                       'direction', 'speed', 'integrated_x_movement', 'integrated_y_movement',
                       'timestamp', 'sequence_number', 'delta_ts', 'timestamp_alt']


class FortyHourFitnessPlugin(base_plugin.BasePlugin):
    def __init__(self):
        super().__init__()
        self.current_series = None
        self.current_series_number = 0

    def attachData(self, experiment_file_name, file_path, data_directory,
                   ensemble='auto', ensemble_override=None, strict=True,
                   verify_photodiode=True, allow_counts_only=False,
                   report_path=None, thresholds=None):
        '''
        Attach Jackfish DAQ data and FicTrac data to each series in the hdf5 file.

        A single .jfdaqdata may cover several series (a stimpack "ensemble").  Such a file is
        segmented on camera strobe gaps, cross-checked against the FicTrac line counts and
        the stimpack unix timestamps, and each block is written to its owning series with the
        photodiode trace and every camera's exposure times re-zeroed by the same offset.

        args
            experiment_file_name: unused, kept for the BasePlugin signature
            file_path: full path to the hdf5 data file
            data_directory: date directory holding <series_number>/ subfolders
            ensemble: 'auto' segments when the evidence says to; 'off' never segments
                      (one block per DAQ file, i.e. the pre-ensemble behaviour);
                      'require' raises if any DAQ file yields a single block
            ensemble_override: dict, or path to an ensemble.json, pinning the assignment
            strict: False downgrades every failing check to a warning (quality='forced')
            verify_photodiode: run the photodiode audit before writing
            allow_counts_only: permit a cut derived from FicTrac line counts alone, when no
                               camera stopped between runs
            report_path: where to write the JSON sidecar report
            thresholds: dict overriding any threshold in visanalysis.util.daq_ensemble

        returns
            report dict (also written to report_path, if given)
        '''
        if ensemble not in ('auto', 'off', 'require'):
            raise ValueError("ensemble must be 'auto', 'off' or 'require'; got {!r}".format(ensemble))

        series_numbers = sorted(self.getSeriesNumbers(file_path))
        evidence = de.readSeriesEvidence(file_path, data_directory, series_numbers)
        candidates = de.findDaqCandidates(data_directory, series_numbers)
        overrides = de.loadEnsembleOverride(ensemble_override, data_directory)

        report = {'hdf5_file': os.path.basename(file_path),
                  'data_directory': data_directory,
                  'attach_utc': datetime.datetime.utcnow().isoformat(timespec='seconds') + 'Z',
                  'tool_version': ATTACH_TOOL_VERSION,
                  'ensemble_mode': ensemble,
                  'series_numbers': series_numbers,
                  'notes': [], 'plans': [], 'series': {}, 'warnings': []}

        if ensemble == 'off':
            ownership, series_without_daq, notes = {}, [], []
            for sn in series_numbers:
                ev = evidence[sn]
                if ev.series_dir is None:
                    notes.append('Series {} does not exist in directory strucutre. '
                                 'Skipping...'.format(sn))
                elif ev.daq_path is not None:
                    ownership[ev.daq_path] = [sn]
                else:
                    series_without_daq.append(sn)
        else:
            ownership, series_without_daq, notes = de.buildDaqOwnership(
                evidence, candidates, thresholds)

        # An explicit override replaces the inferred ownership for the files it names.
        for daq_path, entry in overrides.items():
            ownership = {k: v for k, v in ownership.items() if k != daq_path}
            ownership[daq_path] = list(entry['series'])
            for sn in entry['series']:
                if sn in series_without_daq:
                    series_without_daq.remove(sn)
            notes.append('Override: {} -> series {} ({}).'.format(
                os.path.basename(daq_path), entry['series'], entry['reason']))

        report['notes'] = list(notes)
        for note in notes:
            print(note)

        per_series_cams_timing = {}
        per_series_plan = {}

        for daq_path, group_series in ownership.items():
            override = overrides.get(daq_path, {})
            plan, cams_by_series = self._planAndWriteDaqFile(
                file_path=file_path,
                data_directory=data_directory,
                daq_path=daq_path,
                group_series=group_series,
                evidence=evidence,
                ensemble=ensemble,
                override=override,
                strict=strict,
                verify_photodiode=verify_photodiode,
                allow_counts_only=allow_counts_only,
                thresholds=thresholds,
                report=report)
            per_series_cams_timing.update(cams_by_series)
            for sn in plan.series_for_block:
                per_series_plan[sn] = plan

        for sn in series_without_daq:
            print('WARNING! Required DAQ data files not found for series {} in {}'.format(
                sn, data_directory))
            report['warnings'].append('No DAQ data for series {}.'.format(sn))

        # FicTrac, per series.  cams_timing is looked up explicitly per series rather than
        # carried over from the previous loop iteration; the old code let series N compare its
        # own .dat length against series N-1's strobe count.
        for series_number in series_numbers:
            cams_timing = per_series_cams_timing.get(series_number, {})
            self._attachFictrac(file_path, data_directory, series_number,
                                cams_timing, evidence.get(series_number), report)

        if report_path is None:
            stem = os.path.splitext(os.path.basename(file_path))[0]
            report_path = os.path.join(data_directory, 'attach_report_{}.json'.format(stem))
        try:
            with open(report_path, 'w') as fh:
                json.dump(report, fh, indent=2, default=_jsonSafe)
            print('Wrote attach report to {}'.format(report_path))
        except OSError as err:
            print('WARNING! Could not write attach report to {}: {}'.format(report_path, err))

        return report

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _planAndWriteDaqFile(self, file_path, data_directory, daq_path, group_series,
                             evidence, ensemble, override, strict, verify_photodiode,
                             allow_counts_only, thresholds, report):
        """Parse one DAQ file once, plan the split, validate it, then write every block."""
        relpath = os.path.relpath(daq_path, data_directory)
        print('Parsing {} ...'.format(relpath))
        voltage_recording, nicknames, _, sample_rate = getVoltageRecording(daq_path)
        n_samples = voltage_recording.shape[1]

        frame_monitor_idxs = [i for i, n in enumerate(nicknames) if n.startswith('frame_monitor')]
        strobe_idxs = [i for i, n in enumerate(nicknames) if n.startswith('cam_strobe')]
        strobe_rows = [voltage_recording[i, :] for i in strobe_idxs]

        edges = de.findStrobeEdges(voltage_recording, nicknames)
        for nickname, (rise, _) in edges.items():
            if rise.size == 0:
                print('Camera {} had no strobes / exposures.'.format(nickname))

        loco_series = [sn for sn in group_series if evidence[sn].do_loco]
        if not loco_series:
            loco_series = list(group_series)

        line_counts = [evidence[sn].n_fictrac_lines for sn in loco_series]
        expected_total = (sum(line_counts) if all(c is not None for c in line_counts) else None)

        n_blocks_wanted = 1 if ensemble == 'off' else len(loco_series)
        segmenter, seg_diag = de.chooseSegmenter(
            edges, sample_rate,
            expected_total if n_blocks_wanted > 1 else None,
            n_blocks_wanted, thresholds)

        forced_cuts = override.get('cut_sample_indices')
        if override.get('segmentation_camera'):
            segmenter = override['segmentation_camera']

        # Detection, run even for a single-series group: this is the "more trials than one
        # protocol run accounts for" test. planCuts short-circuits when the group holds one
        # series, so without this the most dangerous case - a DAQ file spanning runs the hdf5
        # does not list as contiguous - would attach silently.
        if ensemble != 'off' and not forced_cuts:
            n_viable = self._countViableBlocks(edges, segmenter, sample_rate, n_samples, thresholds)
            if n_viable > 1 and len(loco_series) == 1:
                raise de.EnsembleGroupingError(
                    '{} contains {} camera blocks on {} but the hdf5 offers only series {} '
                    '(no contiguous follow-on run). More trials were recorded than one protocol '
                    "run accounts for. Use ensemble='off' to attach it as a single block, or "
                    'supply an ensemble.json override naming the series it covers.'.format(
                        relpath, n_viable, segmenter, loco_series[0]),
                    {'daq_file': relpath, 'n_viable_blocks': n_viable, 'series': loco_series})

        plan = de.planCuts(
            edges=edges, segmenter=segmenter, sample_rate=sample_rate, n_samples=n_samples,
            line_counts=line_counts if n_blocks_wanted > 1 else [None],
            strobe_rows=strobe_rows, daq_path=daq_path, daq_relpath=relpath,
            channel_nicknames=nicknames, allow_counts_only=allow_counts_only,
            forced_cut_samples=forced_cuts, thresholds=thresholds)
        if override:
            plan.method = 'manual'
            plan.warn('Manual override: {}'.format(override.get('reason', '')))
        plan.diagnostics['segmenter'] = seg_diag

        if ensemble == 'require' and plan.n_blocks == 1:
            raise de.EnsembleGroupingError(
                "ensemble='require' but {} yielded a single block.".format(relpath),
                {'daq_file': relpath})

        # A DAQ file showing several camera blocks while the hdf5 offers only one series is a
        # contradiction we refuse to resolve silently.
        if plan.n_blocks > 1 and len(loco_series) == 1:
            raise de.EnsembleGroupingError(
                '{} contains {} camera blocks but the hdf5 has no contiguous follow-on series '
                "after series {}. Use ensemble='off' to attach it as one block, or supply an "
                'ensemble.json override.'.format(relpath, plan.n_blocks, loco_series[0]),
                {'daq_file': relpath, 'n_blocks': plan.n_blocks, 'series': loco_series})

        if plan.n_blocks == 1 and len(loco_series) > 1:
            plan.warn('The hdf5 says series {} form an ensemble but {} shows a single camera '
                      'block; attaching as one block to series {}.'.format(
                          loco_series, relpath, loco_series[0]))
            loco_series = loco_series[:1]

        plan.series_for_block, assoc_checks = de.associateBlocks(
            plan, evidence, loco_series, edges, thresholds)
        plan.checks.extend(assoc_checks)
        plan = de.validatePlan(plan, evidence, edges, strict=strict, thresholds=thresholds)

        pd_audit = {}
        if verify_photodiode and frame_monitor_idxs:
            pd_audit = self._auditPhotodiode(plan, evidence, voltage_recording,
                                             frame_monitor_idxs, sample_rate, strict, thresholds)

        print(de.formatPlanReport(plan, evidence, seg_diag, pd_audit))
        report['plans'].append(de.planToDict(plan))

        # ---- everything validated; only now touch the hdf5 ----
        cams_by_series = {}
        for block_index, (block, series_number) in enumerate(zip(plan.blocks, plan.series_for_block)):
            cams_timing = de.sliceCamsTiming(edges, block.sample_lo, block.sample_hi, sample_rate)
            cams_by_series[series_number] = cams_timing
            self._writeBlock(file_path, plan, block, block_index, series_number,
                             voltage_recording, frame_monitor_idxs, nicknames,
                             cams_timing, sample_rate, pd_audit.get(series_number), report)

        del voltage_recording, strobe_rows
        return plan, cams_by_series

    @staticmethod
    def _countViableBlocks(edges, segmenter, sample_rate, n_samples, thresholds):
        """How many camera blocks this DAQ file contains, counting only substantial ones.

        A brief camera hiccup splits the train without meaning a new protocol run, so a
        fragment below the block floors is folded back into its neighbour rather than counted.
        """
        th = de.resolveThresholds(thresholds)
        if segmenter not in edges:
            return 1
        rise, fall = edges[segmenter]
        if rise.size == 0:
            return 1
        gaps, _ = de.findEdgeGaps(rise, fall, sample_rate, thresholds)
        if not gaps:
            return 1
        boundaries = [0] + [g['cut_sample'] for g in gaps] + [int(n_samples)]
        n_viable = 0
        for lo, hi in zip(boundaries, boundaries[1:]):
            n_strobes = int(((rise >= lo) & (rise < hi)).sum())
            if (hi - lo) / sample_rate >= th['MIN_BLOCK_S'] and n_strobes >= th['MIN_BLOCK_PULSES']:
                n_viable += 1
        return max(1, n_viable)

    def _auditPhotodiode(self, plan, evidence, voltage_recording, frame_monitor_idxs,
                         sample_rate, strict, thresholds):
        """Run the existing photodiode detector on each block and compare to the metadata.

        Over-detection is the mis-slice signature and aborts.  Under-detection is a
        pre-existing property of the detector, orthogonal to segmentation, so it only warns.
        """
        th = de.resolveThresholds(thresholds)
        audit = {}
        for block, series_number in zip(plan.blocks, plan.series_for_block):
            ev = evidence[series_number]
            frame_monitor = voltage_recording[frame_monitor_idxs, block.sample_lo:block.sample_hi]
            mes = 0.9 * ((ev.pre_time or 0.0) + (ev.tail_time or 0.0)) * sample_rate
            entry = {'n_epochs': ev.n_epoch_groups}
            for label, zero_phase in (('causal', False), ('zero_phase', True)):
                try:
                    channel_timing = computeStimulusTiming(
                        frame_monitor, sample_rate, minimum_epoch_separation=mes,
                        quiet=True, highpass_zero_phase=zero_phase)
                except Exception as err:                       # detector failure is not fatal
                    entry['counts_{}'.format(label)] = None
                    plan.warn('Photodiode audit failed for series {} ({}): {}'.format(
                        series_number, label, err))
                    continue
                counts, missed, exact = [], [], []
                for ch, timing in enumerate(channel_timing):
                    starts = np.asarray(timing['stimulus_start_times'], dtype=float)
                    counts.append(int(starts.size))
                    missed_ch = self._missedEpochs(starts, ev, plan, block, th)
                    missed.append(missed_ch)
                    if missed_ch is not None and len(missed_ch) == 0 and starts.size == ev.n_epoch_groups:
                        exact.append(ch)
                entry['counts_{}'.format(label)] = counts
                entry['missed_{}'.format(label)] = missed
                entry['exact_{}'.format(label)] = exact
            audit[series_number] = entry

            overshoot_limit = ev.n_epoch_groups + th['PD_OVERSHOOT_MAX']
            for label in ('causal', 'zero_phase'):
                counts = entry.get('counts_{}'.format(label))
                if counts and max(counts) > overshoot_limit:
                    plan.addCheck('V12_pd_overshoot_series_{}'.format(series_number), 'fail',
                                  measured=max(counts), tolerance=overshoot_limit,
                                  message='photodiode found more stimuli than the series has '
                                          'epochs; the slice is wrong')
                    if strict:
                        raise de.EnsembleValidationError(
                            'Photodiode audit: series {} has {} epochs but the {} detector found '
                            '{} stimuli in its block. The DAQ slice does not match the series.'
                            .format(series_number, ev.n_epoch_groups, label, max(counts)),
                            {'series': series_number, 'counts': counts})
                    plan.quality = 'forced'

            if not (entry.get('exact_causal') or entry.get('exact_zero_phase')):
                plan.warn('Series {}: no photodiode channel gives an exact per-epoch match '
                          '(counts causal={}, zero-phase={} of {} epochs).'.format(
                              series_number, entry.get('counts_causal'),
                              entry.get('counts_zero_phase'), ev.n_epoch_groups))

        # One config has to serve every series in the group, so intersect across them rather
        # than reporting whatever the first series happened to like. Causal wins ties because
        # it is the current default and changes nothing for existing callers.
        recommended_channel, recommended_zero_phase = None, None
        for zero_phase, key in ((False, 'exact_causal'), (True, 'exact_zero_phase')):
            common = None
            for entry in audit.values():
                channels = set(entry.get(key) or [])
                common = channels if common is None else (common & channels)
            if common:
                recommended_channel = sorted(common)[0]
                recommended_zero_phase = zero_phase
                break
        for entry in audit.values():
            entry['recommended_channel'] = recommended_channel
            entry['recommended_zero_phase'] = recommended_zero_phase
        if recommended_channel is None and audit:
            plan.warn('No single photodiode channel is exact for every series in this group; '
                      'inspect pd_missed_epochs_* before trusting positional epoch alignment.')
        return audit

    @staticmethod
    def _missedEpochs(detected_starts, ev, plan, block, th):
        """Indices of epochs with no detected stimulus within tolerance. None if unknowable."""
        if plan.t0_unix is None or ev.epoch_onset_unix is None or ev.epoch_onset_unix.size == 0:
            return None
        predicted = ev.epoch_onset_unix - plan.t0_unix - block.sample_lo / plan.sample_rate
        missed, used = [], set()
        for idx, want in enumerate(predicted):
            if not np.isfinite(want):
                continue
            if detected_starts.size == 0:
                missed.append(idx)
                continue
            order = np.argsort(np.abs(detected_starts - want))
            hit = None
            for cand in order:
                if cand in used:
                    continue
                if abs(detected_starts[cand] - want) <= th['PD_MATCH_TOL_S']:
                    hit = int(cand)
                break
            if hit is None:
                missed.append(idx)
            else:
                used.add(hit)
        return missed

    def _writeBlock(self, file_path, plan, block, block_index, series_number,
                    voltage_recording, frame_monitor_idxs, nicknames, cams_timing,
                    sample_rate, pd_entry, report):
        """Write one block's photodiode trace and camera timing to its owning series."""
        lo, hi = block.sample_lo, block.sample_hi
        frame_monitor = voltage_recording[frame_monitor_idxs, lo:hi]
        time_vector = np.arange(hi - lo) / sample_rate

        with h5py.File(file_path, 'r+') as experiment_file:
            find_partial = functools.partial(h5io.find_series, sn=series_number)
            epoch_run_group = experiment_file.visititems(find_partial)
            if epoch_run_group is None:
                print('WARNING! Series {} not found in {}; skipping.'.format(
                    series_number, file_path))
                return

            stimulus_timing_group = epoch_run_group.require_group('stimulus_timing')
            h5io.overwriteDataSet(stimulus_timing_group, 'frame_monitor', frame_monitor)
            h5io.overwriteDataSet(stimulus_timing_group, 'time_vector', time_vector)
            attrs = stimulus_timing_group.attrs
            attrs['sample_rate'] = sample_rate

            # Provenance: the slice is fully described, so the operation is auditable and the
            # absolute DAQ time of any sample is recoverable.
            attrs['attach_tool_version'] = ATTACH_TOOL_VERSION
            attrs['attach_utc'] = report['attach_utc']
            attrs['daq_file'] = plan.daq_relpath
            attrs['daq_n_samples'] = int(plan.n_samples)
            attrs['daq_sample_offset'] = int(lo)
            attrs['daq_sample_count'] = int(hi - lo)
            attrs['daq_time_offset'] = float(lo / sample_rate)
            attrs['frame_monitor_channels'] = [nicknames[i] for i in frame_monitor_idxs]
            attrs['daq_t0_unix'] = float('nan') if plan.t0_unix is None else float(plan.t0_unix)
            attrs['daq_t0_unix_spread_s'] = (float('nan') if plan.t0_spread_s is None
                                             else float(plan.t0_spread_s))
            attrs['daq_t0_unix_source'] = 'strobe_block_anchor'

            attrs['ensemble_daq_group_id'] = os.path.splitext(os.path.basename(plan.daq_path))[0]
            attrs['ensemble_series'] = np.asarray(plan.series_for_block, dtype='int32')
            attrs['ensemble_block_index'] = int(block_index)
            attrs['ensemble_n_blocks'] = int(plan.n_blocks)
            attrs['ensemble_method'] = plan.method
            attrs['ensemble_segmenter_channel'] = plan.segmenter or ''
            attrs['ensemble_cut_samples'] = np.asarray(plan.cut_samples, dtype='int64')
            attrs['ensemble_cut_strobe_index'] = np.asarray(
                [block.strobe_lo, block.strobe_hi], dtype='int64')
            attrs['ensemble_cut_uncertainty_samples'] = int(plan.cut_uncertainty_samples)
            attrs['ensemble_cut_snap_distance_samples'] = np.asarray(
                plan.cut_snap_distance_samples, dtype='int64')
            attrs['ensemble_separation_ratio'] = float(plan.separation_ratio)
            attrs['ensemble_gap_before_s'] = (float('nan') if block.gap_before_s is None
                                              else float(block.gap_before_s))
            attrs['ensemble_gap_after_s'] = (float('nan') if block.gap_after_s is None
                                             else float(block.gap_after_s))

            attrs['attach_quality'] = plan.quality
            attrs['attach_warnings'] = np.asarray(plan.warnings, dtype=h5py.string_dtype())
            attrs['attach_report_json'] = json.dumps(de.planToDict(plan), default=_jsonSafe)[:200000]

            if pd_entry:
                for key, value in (('pd_stim_counts', pd_entry.get('counts_causal')),
                                   ('pd_stim_counts_zero_phase', pd_entry.get('counts_zero_phase')),
                                   ('pd_channels_exact', pd_entry.get('exact_causal')),
                                   ('pd_channels_exact_zero_phase', pd_entry.get('exact_zero_phase'))):
                    if value is not None:
                        attrs[key] = np.asarray(value, dtype='int32')
                for label in ('causal', 'zero_phase'):
                    for ch, missed in enumerate(pd_entry.get('missed_{}'.format(label)) or []):
                        if missed is not None:
                            attrs['pd_missed_epochs_{}_ch{}'.format(label, ch)] = np.asarray(
                                missed, dtype='int32')
                if pd_entry.get('recommended_channel') is not None:
                    attrs['pd_recommended_timing_channel_ind'] = int(pd_entry['recommended_channel'])
                    attrs['pd_recommended_highpass_zero_phase'] = bool(
                        pd_entry['recommended_zero_phase'])

            behavior_group = epoch_run_group.require_group('behavior')
            for nickname, timing in cams_timing.items():
                cam_group = behavior_group.require_group(nickname)
                h5io.overwriteDataSet(cam_group, 'exposure_onset', timing['exposure_onset'])
                h5io.overwriteDataSet(cam_group, 'exposure_offset', timing['exposure_offset'])
                cam_group.attrs['exposure_time'] = timing['exposure_time']
                cam_group.attrs['frame_rate'] = timing['frame_rate']
                cam_group.attrs['n_pulses'] = timing['n_pulses']
                cam_group.attrs['daq_sample_offset'] = timing['daq_sample_offset']
                cam_group.attrs['first_pulse_sample'] = timing['first_pulse_sample']
                cam_group.attrs['last_pulse_sample'] = timing['last_pulse_sample']
                cam_group.attrs['n_straddling_pulses'] = timing['n_straddling_pulses']
                cam_group.attrs['is_segmenter'] = bool(nickname == plan.segmenter)
                cam_group.attrs['free_running'] = bool(nickname != plan.segmenter)

                if timing['exposure_time'] < 0:
                    print('Camera {} strobes were all inverted for series {}. Proceed with '
                          'caution. Mean frame rate: {}'.format(
                              nickname, series_number, timing['frame_rate']))

                # The audit that proves the photodiode and camera re-zeroings agree.
                assert cam_group.attrs['daq_sample_offset'] == attrs['daq_sample_offset'], (
                    'daq_sample_offset mismatch between stimulus_timing and behavior/{}'.format(
                        nickname))

        report['series'].setdefault(str(series_number), {}).update({
            'daq_file': plan.daq_relpath,
            'block_index': block_index,
            'daq_sample_offset': int(lo),
            'daq_sample_count': int(hi - lo),
            'n_strobes': {k: v['n_pulses'] for k, v in cams_timing.items()},
            'attach_quality': plan.quality,
        })
        print('Attached timing data to series {} (block {}, samples [{}, {}))'.format(
            series_number, block_index, lo, hi))

    def _attachFictrac(self, file_path, data_directory, series_number, cams_timing,
                       ev, report):
        """Attach this series' own FicTrac .dat and log.txt."""
        series_directory = os.path.join(data_directory, str(series_number))
        fictrac_directory = os.path.join(series_directory, 'loco')
        if not os.path.exists(fictrac_directory):
            print('WARNING! Loco directory {} not found.'.format(fictrac_directory))
            return

        fictrac_data_paths = sorted(
            os.path.join(fictrac_directory, x) for x in os.listdir(fictrac_directory)
            if x.endswith('.dat'))
        if not fictrac_data_paths:
            print('WARNING! No Fictrac data found in {}.'.format(fictrac_directory))
            return
        fictrac_data_path = fictrac_data_paths[0]

        # Checked after the .dat, so a missing log.txt cannot abort the whole attach loop
        # partway through and leave the hdf5 half-written.
        log_path = os.path.join(fictrac_directory, 'log.txt')
        if not os.path.exists(log_path):
            print('WARNING! log.txt not found in {}; attaching Fictrac data without it.'.format(
                fictrac_directory))
            log_path = None

        fictrac_data = np.genfromtxt(fictrac_data_path, delimiter=",")
        if fictrac_data.ndim == 1:
            fictrac_data = fictrac_data[np.newaxis, :]

        fictrac_cams = [k for k in cams_timing if 'fictrac' in k.lower()]
        n_strobes = None
        if fictrac_cams:
            n_strobes = len(cams_timing[fictrac_cams[0]]['exposure_onset'])
            if n_strobes < len(fictrac_data):
                print('There are more Fictrac data lines than strobes for series {} '
                      '({} lines vs {} strobes).'.format(
                          series_number, len(fictrac_data), n_strobes))

        log_lines = []
        if log_path is not None:
            with open(log_path, 'r') as lf:
                log_lines = lf.readlines()

        with h5py.File(file_path, 'r+') as experiment_file:
            find_partial = functools.partial(h5io.find_series, sn=series_number)
            epoch_run_group = experiment_file.visititems(find_partial)
            if epoch_run_group is None:
                return
            behavior_group = epoch_run_group.require_group('behavior')
            h5io.overwriteDataSet(behavior_group, 'fictrac_data', fictrac_data)
            behavior_group['fictrac_data'].attrs['fictrac_data_header'] = FICTRAC_DATA_HEADER
            behavior_group['fictrac_data'].attrs['fictrac_dat_file'] = os.path.relpath(
                fictrac_data_path, data_directory)
            behavior_group['fictrac_data'].attrs['n_lines'] = int(len(fictrac_data))
            if n_strobes is not None:
                behavior_group['fictrac_data'].attrs['n_strobes'] = int(n_strobes)
                behavior_group['fictrac_data'].attrs['strobe_surplus'] = int(
                    n_strobes - len(fictrac_data))
                behavior_group['fictrac_data'].attrs['strobe_surplus_convention'] = 'trailing'
            frame_counts = fictrac_data[:, 0]
            behavior_group['fictrac_data'].attrs['frame_count_contiguous'] = bool(
                frame_counts.size > 0 and frame_counts[0] == 0
                and np.all(np.diff(frame_counts) == 1))

            if log_lines:
                log_group = behavior_group.require_group('log_lines')
                log_group.attrs['n_lines'] = len(log_lines)
                for i, log_line in enumerate(log_lines):
                    line_json = json.loads(log_line)
                    line_group = log_group.require_group('line_{:03d}'.format(i))
                    line_group.attrs['ts'] = line_json.pop('ts')
                    for log_k, log_v in line_json.items():
                        item_group = line_group.require_group(log_k)
                        for k, v in log_v.items():
                            item_group.attrs[k] = v

        entry = report['series'].setdefault(str(series_number), {})
        entry['fictrac_lines'] = int(len(fictrac_data))
        entry['fictrac_strobes'] = n_strobes
        print('Attached Fictrac data to series {}'.format(series_number))


ATTACH_TOOL_VERSION = 'daq_ensemble/1.0'


def _jsonSafe(obj):
    """Fallback encoder for numpy scalars/arrays in the report."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


# %%
###########################################################################
# Functions for timing and metadata
#   Accessible outside of the plugin object
###########################################################################


def getVoltageRecording(filepath, use_pandas=True):
    """
    Get frame monitor voltage traces, associated timestamps, and frame rate
    params:
        :filepath: path to voltage recording file, with no suffix
        :use_pandas: parse the body with pandas instead of np.genfromtxt.  Roughly 7x faster
            on a 1 GB file and, with float_precision='round_trip', bit-identical.  Falls back
            to np.genfromtxt if pandas raises.
    """

    with open(filepath, 'r') as jf:
        header = jf.readline()
        if header == '\n': # Old format with a blank header; assume header
            header = None
            input_channels = {"AIN0": "frame_monitor_R",
                              "AIN2": "frame_monitor_C",
                              "AIN4": "frame_monitor_L",
                              "FIO0": "cam_strobe_Fictrac",
                              "FIO2": "cam_strobe_Top",
                              "FIO4": "cam_strobe_Left"}
            daq_framerate = 5000
            n_input_chs = len(input_channels)

            daq_data = []
            for line in jf:
                if line.endswith('\n'):
                    line = line[:-1]
                if line.startswith('['):
                    line = line[1:]
                if line.endswith(']'):
                    line = line[:-1]

                line = [float(entry) for entry in line.split(',') if entry!='']
                assert len(line)%n_input_chs == 0, line
                daq_data.extend(line)
            daq_data_np = np.asarray(daq_data).reshape((-1, n_input_chs))

        else: # New format with a header
            header = json.loads(header)
            daq_framerate = header['scan_rate']
            input_channels = header['input_channels']

            if use_pandas:
                try:
                    # float_precision='round_trip' is required: pandas' default fast parser
                    # differs from np.genfromtxt by 1 ULP on ~16% of values.
                    daq_data_np = pd.read_csv(jf, sep=r'\s+', header=None,
                                              dtype=np.float64,
                                              float_precision='round_trip').to_numpy()
                except Exception as err:
                    print('pandas parse failed ({}); falling back to np.genfromtxt.'.format(err))
                    jf.seek(0)
                    jf.readline()
                    daq_data_np = np.genfromtxt(jf)
            else:
                daq_data_np = np.genfromtxt(jf)

    input_ch_addresses = list(input_channels.keys())
    input_ch_nicknames = list(input_channels.values())

    time_vector = np.arange(daq_data_np.shape[0]) / daq_framerate # in seconds; approximate

    return daq_data_np.T, input_ch_nicknames, time_vector, daq_framerate


def computeStimulusTiming(frame_monitor_channels, sample_rate, minimum_epoch_separation,
                          command_frame_rate=120, threshold=(0.6,), frame_slop=(20,),
                          time_vector=None, run_parameters=None, n_epochs_parameterized=None,
                          plot_trace_flag=False, quiet=True, highpass_zero_phase=False):
    """
    Stimulus timing from square-wave photodiode traces.

    This is the per-channel body of FortyHourDataObject.getStimulusTiming, moved out verbatim
    so the attach path can reuse it instead of reimplementing stimulus detection.

    `threshold` and `frame_slop` default to the 1-tuples that ImagingDataObject.__init__ sets;
    find_peaks reads a 1-tuple `height` as "minimum only" and the frame_slop comparison
    broadcasts, so normalizing them to scalars would silently change existing results.

    `highpass_zero_phase=False` keeps today's causal signal.sosfilt.  True uses sosfiltfilt,
    which removes the photodiode's slow baseline drift symmetrically instead of leaving a
    ~16 s transient; that stops spurious baseline peaks from bridging minimum_epoch_separation
    and merging adjacent epochs.  It is off by default because it is not a strict improvement
    on every channel.

    returns
        list of dicts, one per channel, in channel order
    """
    if len(frame_monitor_channels.shape) == 1:
        frame_monitor_channels = frame_monitor_channels[np.newaxis, :]

    # A multi-valued stim_time (which is exactly what ensemble protocols produce) cannot be
    # formatted with {:.3f}; fall back to the measured durations for the printout.
    _st = run_parameters.get('stim_time') if run_parameters else None
    _st_scalar = float(np.ravel(_st)[0]) if (_st is not None and np.size(_st) == 1) else None

    num_channels = frame_monitor_channels.shape[0]
    channel_timing = []
    for ch in range(num_channels):
        frame_monitor = frame_monitor_channels[ch, :]

        # Low-pass filter frame_monitor trace
        b, a = signal.butter(4, min(10*command_frame_rate, sample_rate/2-1), btype='low', fs=sample_rate)
        frame_monitor = signal.filtfilt(b, a, frame_monitor)

        # High-pass filter frame_monitor trace
        sos = signal.butter(2, 0.01, 'highpass', fs=sample_rate, output='sos')
        if highpass_zero_phase:
            frame_monitor = signal.sosfiltfilt(sos, frame_monitor)
        else:
            frame_monitor = signal.sosfilt(sos, frame_monitor)

        # shift & normalize so frame monitor trace lives on [0 1]
        frame_monitor = frame_monitor - np.nanmin(frame_monitor)
        frame_monitor = frame_monitor / np.nanmax(frame_monitor)

        # find frame flip times
        ideal_frame_len = 1 / command_frame_rate * sample_rate  # datapoints
        ideal_frame_len_samples = int(np.round(1 / command_frame_rate * sample_rate))  # datapoints
        min_peak_distance = int(np.floor(ideal_frame_len * 1.8))  # datapoints
        ups, peak_params = signal.find_peaks(frame_monitor, height=threshold, threshold=None, distance=min_peak_distance, prominence=0.04, width=None, wlen=None, rel_height=0.5, plateau_size=None)

        downs = []
        for i in range(len(ups)):
            up_0 = ups[i]
            down = up_0 + ideal_frame_len_samples
            downs.append(down)
        downs = np.asarray(downs)

        if plot_trace_flag:
            plt.figure()
            plt.plot(frame_monitor)
            plt.plot(ups, np.ones(ups.shape), 'go')
            plt.plot(downs, np.ones(downs.shape), 'rx')
            plt.show()

        frame_times = np.sort(np.append(ups, downs)) # datapoints

        # Use frame flip times to find stimulus start times
        stimulus_start_frames = np.append(0, np.where(np.diff(frame_times) > minimum_epoch_separation)[0] + 1)
        stimulus_end_frames = np.append(np.where(np.diff(frame_times) > minimum_epoch_separation)[0], len(frame_times)-1)
        stimulus_start_times = frame_times[stimulus_start_frames] / sample_rate  # datapoints -> sec
        stimulus_end_times = frame_times[stimulus_end_frames] / sample_rate  # datapoints -> sec

        stim_durations = stimulus_end_times - stimulus_start_times  # sec

        ideal_frame_len = 1 / command_frame_rate * sample_rate  # datapoints
        frame_durations = []
        dropped_frame_times = []
        good_frame_times = []
        for s_ind, ss in enumerate(stimulus_start_frames):
            frame_len = np.diff(frame_times[stimulus_start_frames[s_ind]:stimulus_end_frames[s_ind]+1])
            dropped_frame_inds = np.where(np.abs(frame_len - ideal_frame_len)>frame_slop)[0]  # +1 b/c diff
            if len(dropped_frame_inds) > 0:
                stim_dropped_frame_times = frame_times[ss+dropped_frame_inds]  # time when dropped frames should have flipped
                dropped_frame_times.append(stim_dropped_frame_times)
            good_frame_inds = np.where(np.abs(frame_len - ideal_frame_len) <= frame_slop)[0]
            if len(good_frame_inds) > 0:
                stim_good_frame_times = frame_times[ss+good_frame_inds]
                good_frame_times.append(stim_good_frame_times)
                frame_durations.append(np.diff(stim_good_frame_times))  # only include non-dropped frames in frame rate calc

        if len(dropped_frame_times) > 0:
            dropped_frame_times = np.hstack(dropped_frame_times)  # datapoints
        else:
            dropped_frame_times = np.array(dropped_frame_times)
        if len(good_frame_times) > 0:
            good_frame_times = np.hstack(good_frame_times)  # datapoints
        else:
            good_frame_times = np.array(good_frame_times)

        frame_durations = np.hstack(frame_durations)  # datapoints
        measured_frame_len = np.mean(frame_durations)  # datapoints
        frame_rate = 1 / (measured_frame_len / sample_rate)  # Hz

        if plot_trace_flag:
            frame_monitor_figure = plt.figure(figsize=(12, 8))
            gs1 = gridspec.GridSpec(2, 2)
            ax = frame_monitor_figure.add_subplot(gs1[1, :])
            if time_vector is not None:
                ax.plot(time_vector, frame_monitor)
            ax.plot(stimulus_start_times, threshold * np.ones(stimulus_start_times.shape), 'go', label='Stim start')
            ax.plot(stimulus_end_times, threshold * np.ones(stimulus_end_times.shape)-0.05, 'ro', label='Stim end')
            ax.plot(good_frame_times / sample_rate, 1 * np.ones(good_frame_times.shape), 'go', markerfacecolor='none', label='Good frame')
            ax.plot(dropped_frame_times / sample_rate, 1 * np.ones(dropped_frame_times.shape), 'rx', label='Dropped frame')
            ax.legend()
            ax.set_xlabel('Time [s]')
            ax.set_title('Ch. {}: Frame rate = {:.2f} Hz'.format(ch, frame_rate), fontsize=12)

            ax = frame_monitor_figure.add_subplot(gs1[0, 0])
            ax.hist(frame_durations)
            ax.axvline(ideal_frame_len, color='k')
            ax.set_xlabel('Frame duration (datapoints)')

            ax = frame_monitor_figure.add_subplot(gs1[0, 1])
            ax.plot(stim_durations, 'b.')
            n_epochs_for_axis = (run_parameters or {}).get('num_epochs', len(stim_durations))
            if _st_scalar is not None:
                ax.axhline(y=_st_scalar, xmin=0, xmax=n_epochs_for_axis, color='k', linestyle='-', marker='None', alpha=0.50)
            else:
                ax.axhline(y=np.mean(stim_durations), xmin=0, xmax=n_epochs_for_axis, color='k', linestyle='-', marker='None', alpha=0.50)
            ymin = 0.9 * np.min(stim_durations)
            ymax = 1.1 * np.max(stim_durations)
            ax.set_ylim([ymin, ymax])
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Stim duration (sec)')

            frame_monitor_figure.tight_layout()
            plt.show()

        if quiet:
            pass
        else:
            # Print timing summary
            print('===================TIMING: Channel {}======================'.format(ch))
            if n_epochs_parameterized is None:
                print('{} Stims presented'.format(len(stim_durations)))
            else:
                print('{} Stims presented (of {} parameterized)'.format(len(stim_durations), n_epochs_parameterized))
            inter_stim_starts = np.diff(stimulus_start_times)
            if len(inter_stim_starts) >= 1:
                if _st_scalar is not None:
                    print('Stim start to start: [min={:.3f}, median={:.3f}, max={:.3f}] / parameterized = {:.3f} sec'.format(inter_stim_starts.min(),
                                                                                                                            np.median(inter_stim_starts),
                                                                                                                            inter_stim_starts.max(),
                                                                                                                            _st_scalar + run_parameters['pre_time'] + run_parameters['tail_time']))
                else:
                    print('Stim start to start: [min={:.3f}, median={:.3f}, max={:.3f}] sec'.format(inter_stim_starts.min(),
                                                                                                                            np.median(inter_stim_starts),
                                                                                                                            inter_stim_starts.max()))
            if _st_scalar is not None:
                print('Stim duration: [min={:.3f}, median={:.3f}, max={:.3f}] / parameterized = {:.3f} sec'.format(stim_durations.min(), np.median(stim_durations), stim_durations.max(), _st_scalar))
            else:
                print('Stim duration: [min={:.3f}, median={:.3f}, max={:.3f}] sec'.format(stim_durations.min(), np.median(stim_durations), stim_durations.max()))
            total_frames = len(frame_times)
            dropped_frames = len(dropped_frame_times)
            print('Dropped {} / {} frames ({:.2f}%)'.format(dropped_frames, total_frames, 100*dropped_frames/total_frames))
            print('==========================================================')

        new_dict = {'stimulus_end_times': stimulus_end_times,
                    'stimulus_start_times': stimulus_start_times,
                    'dropped_frame_times': dropped_frame_times,
                    'frame_rate': frame_rate}
        channel_timing.append(new_dict)

    return channel_timing




# %%
###########################################################################
# DataObject specific to FortyHourFitness data. Inherits ImagingDataObject
###########################################################################

class FortyHourDataObject(ImagingDataObject):
    """
    FortyHourDataObject inherits ImagingDataObject and alters the getStimulusTiming method.

    cfg_dict accepts two extra keys beyond ImagingDataObject's:
        timing_channel_ind:   which photodiode channel defines stimulus timing (default 0)
        highpass_zero_phase:  use a zero-phase high-pass in the detector (default False)
    """

    def getStimulusTiming(self,
                          plot_trace_flag=False,
                          use_square_photodiodes=True):
        """
        Returns stimulus timing information based on photodiode voltage trace from frame tracker signal.
        """
        if not use_square_photodiodes:
            return TwentyFourHourDataObject.getStimulusTiming(self, plot_trace_flag=plot_trace_flag)

        frame_monitor_channels, time_vector, sample_rate = self.getVoltageData()
        run_parameters = self.getRunParameters()
        epoch_parameters = self.getEpochParameters()

        # If more than two voltage channels, just take the LAST two in the list as photodiodes
        if len(frame_monitor_channels) > 3:
            frame_monitor_channels = frame_monitor_channels[-3:]

        if len(frame_monitor_channels.shape) == 1:
            frame_monitor_channels = frame_monitor_channels[np.newaxis, :]

        minimum_epoch_separation = 0.9 * (run_parameters['pre_time'] + run_parameters['tail_time']) * sample_rate

        channel_timing = computeStimulusTiming(
            frame_monitor_channels, sample_rate,
            minimum_epoch_separation=minimum_epoch_separation,
            command_frame_rate=self.command_frame_rate,
            threshold=self.threshold,
            frame_slop=self.frame_slop,
            time_vector=time_vector,
            run_parameters=run_parameters,
            n_epochs_parameterized=len(epoch_parameters),
            plot_trace_flag=plot_trace_flag,
            quiet=self.quiet,
            highpass_zero_phase=getattr(self, 'highpass_zero_phase', False))

        return channel_timing[self.timing_channel_ind]
