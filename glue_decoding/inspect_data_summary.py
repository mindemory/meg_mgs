#!/usr/bin/env python3
"""
inspect_data_summary.py

Quick per-subject data-shape/availability summary, read from the small
precomputed per-ROI caches (visual/parietal/frontal) for ONE band -- default
lowgamma, since it's cheap to read for every subject (a few MB each via
precompute_roi_splits.py's caches, not the 8-10GB whole-grid file) -- rather
than anything requiring a whole-grid load. Stim-locked only.

Prints (and saves to a log file):
  1. Per-subject source counts for visual/parietal/frontal ROIs.
  2. Per-subject trial counts per raw target location (1-10).
  3. Per-subject timepoint counts within three stim-locked windows:
       fixation [-1.0, 0.0)   -- pre-stimulus baseline
       stimulus [0.0, 0.2)    -- stimulus epoch
       delay    [0.2, 1.7]    -- delay epoch
     (boundaries match manifold_capacity.py's EPOCHS / plot_timeseries.py's
     BASELINE_WINDOWS convention elsewhere in this repo).

Requires precompute_roi_splits.py to have already built this band's ROI
caches (same requirement as everything else that reads via roi=...) --
missing/failed subjects are reported inline rather than aborting the run.

Usage:
    python inspect_data_summary.py [--subjects 1 2 ...] [--voxRes 8mm]
                                    [--band lowgamma] [--rois visual parietal frontal]
                                    [--logfile <path>]
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from constants import SUBJECT_LIST, ROI_NAMES, ANGLE_MAPPING, get_bids_root
from io_g04 import load_g04_band

FIXATION_WINDOW = (-1.0, 0.0)   # [lo, hi)
STIMULUS_WINDOW = (0.0, 0.2)    # [lo, hi)
DELAY_WINDOW    = (0.2, 1.7)    # [lo, hi]


def _count_window(tv, lo, hi, inclusive_hi):
    if inclusive_hi:
        return int(np.sum((tv >= lo) & (tv <= hi)))
    return int(np.sum((tv >= lo) & (tv < hi)))


def summarize_subject(subjID, band, voxRes, bids_root, rois):
    row = {'subjID': subjID, 'warnings': [], 'errors': []}
    n_trials_seen = None
    target_labels = None
    tv = None
    fsample = None

    for roi in rois:
        key = f'n_{roi}'
        try:
            g04 = load_g04_band(subjID, 'stim', band, voxRes, bids_root, want_phase=False, roi=roi)
        except (FileNotFoundError, ValueError, OSError) as e:
            row[key] = None
            row['errors'].append(f'{roi}: {e}')
            continue

        row[key] = g04['amp'].shape[2]
        if n_trials_seen is None:
            n_trials_seen = g04['amp'].shape[0]
            target_labels = g04['target_labels'].astype(int)
            tv = g04['time_vector']
            fsample = float(g04['actualRate'])
        elif g04['amp'].shape[0] != n_trials_seen:
            row['warnings'].append(
                f"{roi}: trial count {g04['amp'].shape[0]} != {n_trials_seen} (first ROI seen)")

    row['n_trials_total'] = n_trials_seen
    row['fsample'] = fsample

    if target_labels is not None:
        row['loc_counts'] = {loc: int(np.sum(target_labels == loc)) for loc in sorted(ANGLE_MAPPING)}
    else:
        row['loc_counts'] = None

    if tv is not None:
        row['n_fixation']    = _count_window(tv, *FIXATION_WINDOW, inclusive_hi=False)
        row['n_stimulus']    = _count_window(tv, *STIMULUS_WINDOW, inclusive_hi=False)
        row['n_delay']       = _count_window(tv, *DELAY_WINDOW, inclusive_hi=True)
        row['n_times_total'] = int(tv.shape[0])
        row['t_min'], row['t_max'] = float(tv[0]), float(tv[-1])
    else:
        row['n_fixation'] = row['n_stimulus'] = row['n_delay'] = row['n_times_total'] = None
        row['t_min'] = row['t_max'] = None

    return row


def _cell(val, width, fmt=None):
    if val is None:
        s = '-'
    elif fmt is not None:
        s = fmt.format(val)
    else:
        s = str(val)
    return s.rjust(width)


def format_main_table(rows, rois):
    cols = [('subj', 4)] + [(f'n_{roi}', 10) for roi in rois] + \
           [('n_trials', 8), ('fsample', 7), ('n_fix', 5), ('n_stim', 6),
            ('n_delay', 7), ('n_times', 7)]
    header = ' | '.join(name.rjust(w) for name, w in cols)
    lines = [header, '-' * len(header)]
    for r in rows:
        cells = [_cell(r['subjID'], 4)]
        for roi in rois:
            cells.append(_cell(r.get(f'n_{roi}'), 10))
        cells.append(_cell(r.get('n_trials_total'), 8))
        cells.append(_cell(r.get('fsample'), 7, fmt='{:.0f}'))
        cells.append(_cell(r.get('n_fixation'), 5))
        cells.append(_cell(r.get('n_stimulus'), 6))
        cells.append(_cell(r.get('n_delay'), 7))
        cells.append(_cell(r.get('n_times_total'), 7))
        lines.append(' | '.join(cells))
    return '\n'.join(lines)


def format_location_table(rows):
    locs = sorted(ANGLE_MAPPING)
    header = 'subj | ' + ' | '.join(f'loc{loc}'.rjust(5) for loc in locs)
    lines = [header, '-' * len(header)]
    for r in rows:
        counts = r.get('loc_counts')
        if counts is None:
            lines.append(f"{str(r['subjID']).rjust(4)} | (no data)")
            continue
        cells = [str(r['subjID']).rjust(4)] + [str(counts[loc]).rjust(5) for loc in locs]
        lines.append(' | '.join(cells))
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Quick per-subject data-shape summary read from small per-ROI caches.')
    parser.add_argument('--subjects', nargs='+', type=int, default=SUBJECT_LIST)
    parser.add_argument('--voxRes', default='8mm')
    parser.add_argument('--band', default='lowgamma')
    parser.add_argument('--rois', nargs='+', default=list(ROI_NAMES))
    parser.add_argument('--logfile', default=None,
                         help='Path to save the printed tables (default: '
                              '<bids_root>/derivatives/glueDecoding/'
                              'dataSummary_<band>_<voxRes>.log)')
    args = parser.parse_args()

    bids_root = get_bids_root()
    rows = [summarize_subject(s, args.band, args.voxRes, bids_root, args.rois)
            for s in args.subjects]

    out = []
    out.append(f'Data summary | band={args.band} | voxRes={args.voxRes} | '
               f'rois={args.rois} | subjects={args.subjects}')
    out.append(f'Windows (s): fixation={FIXATION_WINDOW} [lo,hi) | '
               f'stimulus={STIMULUS_WINDOW} [lo,hi) | delay={DELAY_WINDOW} [lo,hi]')
    out.append('')
    out.append(format_main_table(rows, args.rois))
    out.append('')
    out.append('Trials per raw target location (1-10):')
    out.append(format_location_table(rows))
    out.append('')

    any_notes = False
    for r in rows:
        if r['warnings']:
            any_notes = True
            out.append(f"WARNING sub-{r['subjID']:02d}: " + '; '.join(r['warnings']))
        if r['errors']:
            any_notes = True
            out.append(f"MISSING sub-{r['subjID']:02d}: " + '; '.join(r['errors']))
    if not any_notes:
        out.append('No warnings or missing-data errors.')

    text = '\n'.join(out)
    print(text)

    logfile = args.logfile or os.path.join(
        bids_root, 'derivatives', 'glueDecoding', f'dataSummary_{args.band}_{args.voxRes}.log')
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    with open(logfile, 'w') as fh:
        fh.write(text + '\n')
    print(f'\nSaved: {logfile}')


if __name__ == '__main__':
    main()
