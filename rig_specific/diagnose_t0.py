#!/usr/bin/env python3
"""Print the DAQ t=0 anchors for one date, so a large V8 spread can be attributed.

Each anchor is (trial time) - (camera edge time). They agree only if the camera starts and
stops the same distance from the trials in every block, so an outlier names the block and the
end (start vs stop) responsible.

    python diagnose_t0.py <date_directory>
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from visanalysis.plugin import base as base_plugin
from visanalysis.plugin.fortyhourfitness import getVoltageRecording
from visanalysis.util import daq_ensemble as de


def main(date_dir):
    hdf5 = sorted(x for x in os.listdir(date_dir) if x.endswith('.hdf5'))
    assert len(hdf5) == 1, 'expected one .hdf5 in {}, found {}'.format(date_dir, hdf5)
    file_path = os.path.join(date_dir, hdf5[0])

    series_numbers = sorted(base_plugin.BasePlugin().getSeriesNumbers(file_path))
    evidence = de.readSeriesEvidence(file_path, date_dir, series_numbers)
    candidates = de.findDaqCandidates(date_dir, series_numbers)
    ownership, _, notes = de.buildDaqOwnership(evidence, candidates)
    for note in notes:
        print(note)

    for daq_path, group in ownership.items():
        print('\n=== {} -> series {} ==='.format(os.path.relpath(daq_path, date_dir), group))
        voltage, nicknames, _, fs = getVoltageRecording(daq_path)
        edges = de.findStrobeEdges(voltage, nicknames)
        rows = [voltage[i, :] for i, n in enumerate(nicknames) if n.startswith('cam_strobe')]
        loco = [sn for sn in group if evidence[sn].do_loco]
        counts = [evidence[sn].n_fictrac_lines for sn in loco]
        segmenter, _ = de.chooseSegmenter(
            edges, fs, sum(counts) if all(c is not None for c in counts) else None, len(loco))
        plan = de.planCuts(edges, segmenter, fs, voltage.shape[1], counts, rows,
                           daq_path=daq_path, daq_relpath=os.path.basename(daq_path),
                           channel_nicknames=nicknames)
        plan.series_for_block, _ = de.associateBlocks(plan, evidence, loco, edges)
        t0, spread, anchors = de.computeT0(evidence, plan, edges)

        rise, fall = edges[segmenter]
        print('  segmenter: {}   blocks: {}'.format(
            segmenter, [(b.sample_lo, b.sample_hi) for b in plan.blocks]))
        labels = []
        for block, sn in zip(plan.blocks, plan.series_for_block):
            rb = rise[(rise >= block.sample_lo) & (rise < block.sample_hi)]
            fb = fall[(fall >= block.sample_lo) & (fall < block.sample_hi)]
            ev = evidence[sn]
            first_off = (t0 + rb[0] / fs) - ev.first_epoch_unix
            last_off = (t0 + fb[-1] / fs) - ev.last_epoch_end_unix
            print('  series {}: first strobe {:+9.4f} s vs first trial start   '
                  '({:.0f} frames)'.format(sn, first_off, abs(first_off) * 300))
            print('            last  strobe {:+9.4f} s vs last  trial end     '
                  '({:.0f} frames)'.format(last_off, abs(last_off) * 300))
            labels += ['series {} start'.format(sn), 'series {} end'.format(sn)]

        base = min(anchors)
        worst = int(np.argmax([abs(x - np.median(anchors)) for x in anchors]))
        print('\n  anchors (unix estimates of DAQ t=0):')
        for i, (lab, x) in enumerate(zip(labels, anchors)):
            print('    {:16s} {:.6f}   {:+.4f} s from lowest{}'.format(
                lab, x, x - base, '   <-- OUTLIER' if i == worst else ''))
        print('  spread = {:.4f} s   (median-absolute-deviation {:.4f} s)'.format(
            spread, float(np.median(np.abs(np.asarray(anchors) - np.median(anchors))))))
        del voltage


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
