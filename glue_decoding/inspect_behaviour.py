#!/usr/bin/env python3
"""
inspect_behaviour.py

Per-subject audit of the eyetracking/behavioural file that the performance
binning depends on, so a subject that silently contributes nothing (or
contributes something wrong) is visible rather than inferred from a missing
line in a log.

For every subject it reports:
  * whether the file exists, and the RAW shape of each stored field;
  * i_sacc_err validity: how many trials are NaN, how many are at or below the
    ~0 threshold that marks a missing/unparsed saccade, and how many therefore
    survive into the bins;
  * the S02B alignment checksum against G03's trialinfo -- if this fails the
    binning is skipped for that subject entirely, so it is worth seeing;
  * the G04 trial count and whether it matches, since the behavioural join is
    row-order-sensitive (G04 groups trials by target location);
  * the smallest per-location count each bin would end up with, which is what
    decides whether crossnobis can use that bin at all.

The i_sacc_raw shape line is included because it is the field behind the
"i_sacc_raw shape mismatch ... i_sacc_angle will be NaN" warning. That warning
concerns i_sacc_ANGLE only: load_behav reads i_sacc_err separately and
straight from the file, so the binning (which uses i_sacc_err) is unaffected.
The shape is printed here so it is possible to confirm that per subject rather
than take it on trust.

Usage:
    python inspect_behaviour.py [--subjects 1 2 ...] [--voxRes 8mm]
                                 [--roi visual] [--n_bins 3]
                                 [--logfile <path>]
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from constants import SUBJECT_LIST, ANGLE_MAPPING, get_bids_root, open_h5
from align import load_behav, verify_alignment, g04_orig_row_index, attach_behav
from io_g03 import load_g03_unfiltered
from io_g04 import load_g04_band
from visual_geometry_cell import MIN_TRIALS_PER_LOC
from visual_geometry_epochs_cell import (I_SACC_ERR_THRESH, performance_bins,
                                          bin_labels)

LOCK_TYPE = 'stim'


def behav_path(bids_root, subjID, behav_dir='eyetracking', raw=False):
    """
    Path to the behavioural file.

    raw=False -> sub-XX_task-mgs-iisess_forSource.mat, S02B's OUTPUT and what
                 the analyses actually read (align.load_behav).
    raw=True  -> sub-XX_task-mgs-iisess.mat, S02B's INPUT. Comparing the two
                 answers whether near-zero i_sacc_err originates upstream in
                 iEye or is introduced by S02B's trial removal/reordering.

    behav_dir lets the same audit be pointed at an alternative directory (e.g.
    an 'eyetracking_old'), which is the only way to tell whether the directory
    currently being read is the intended one -- NOTHING in this codebase
    references any directory but 'eyetracking', so if a stale copy is in place
    the code cannot tell.
    """
    subName = f'sub-{subjID:02d}'
    fn = (f'{subName}_task-mgs-iisess.mat' if raw
          else f'{subName}_task-mgs-iisess_forSource.mat')
    return os.path.join(bids_root, 'derivatives', subName, behav_dir, fn)


def raw_field_shapes(subjID, bids_root, behav_dir='eyetracking', raw=False):
    """Shapes of the stored fields, straight from the .mat, before any reshaping."""
    fp = behav_path(bids_root, subjID, behav_dir, raw)
    if not os.path.exists(fp):
        return None, f'file not found: {fp}'
    try:
        f = open_h5(fp, os.path.basename(fp))
    except Exception as e:
        return None, f'could not open: {e}'
    try:
        key = 'ii_sess' if raw else 'ii_sess_forSource'
        grp = f[key] if key in f else f[list(f.keys())[0]]
        shapes = {k: tuple(np.array(grp[k]).shape) for k in grp.keys()}
    except Exception as e:
        return None, f'could not read ii_sess_forSource: {e}'
    finally:
        f.close()
    return shapes, None


def audit(subjID, bids_root, voxRes, roi, n_bins):
    r = dict(subjID=subjID, ok=False, note='')

    shapes, err = raw_field_shapes(subjID, bids_root)
    r['raw_shapes'] = shapes
    if shapes is None:
        r['note'] = err
        return r
    r['i_sacc_raw_shape'] = shapes.get('i_sacc_raw')

    behav = load_behav(subjID, bids_root)
    if behav is None:
        r['note'] = 'load_behav returned None'
        return r

    e = np.asarray(behav['i_sacc_err'], float)
    r['n_behav'] = e.size
    r['n_nan'] = int(np.isnan(e).sum())
    fin = np.isfinite(e)
    # EXACTLY zero vs merely small is the decisive distinction: an exact 0 is a
    # missing/unparsed saccade flag and is rightly excluded, whereas a small but
    # nonzero value is a real measurement that the threshold would be discarding.
    r['n_zero'] = int((fin & (e == 0)).sum())
    r['n_tiny'] = int((fin & (e > 0) & (e <= I_SACC_ERR_THRESH)).sum())
    r['n_at_thresh'] = r['n_zero'] + r['n_tiny']
    valid = fin & (e > I_SACC_ERR_THRESH)
    r['n_valid'] = int(valid.sum())
    if r['n_valid']:
        v = e[valid]
        r['err_min'] = float(v.min()); r['err_p5'] = float(np.percentile(v, 5))
        r['err_med'] = float(np.median(v)); r['err_max'] = float(v.max())
    tiny = e[fin & (e > 0) & (e <= I_SACC_ERR_THRESH)]
    if tiny.size:
        r['tiny_med'] = float(np.median(tiny)); r['tiny_max'] = float(tiny.max())
    r['angle_all_nan'] = bool(np.all(~np.isfinite(np.asarray(behav['i_sacc_angle'], float))))

    # G03 metadata + alignment checksum
    try:
        g03 = load_g03_unfiltered(subjID, LOCK_TYPE, voxRes, bids_root, roi=roi)
    except Exception as ex:
        r['note'] = f'G03 unavailable ({ex})'
        return r
    r['aligned'] = bool(verify_alignment(g03['trialinfo_col2'], behav['tarlocCode']))

    # G04 trial count + row map
    try:
        g04 = load_g04_band(subjID, LOCK_TYPE, 'alpha', voxRes, bids_root,
                             want_phase=False, roi=roi)
        n_g04 = int(g04['amp'].shape[0])
        y = np.asarray(g04['target_labels']).astype(int)
        del g04
    except Exception as ex:
        r['note'] = f'G04 unavailable ({ex})'
        return r
    r['n_g04'] = n_g04

    idx = g04_orig_row_index(g03['trialinfo_col2'])
    r['n_rowmap'] = int(idx.shape[0])
    r['rowmap_ok'] = bool(idx.shape[0] == n_g04)
    if not r['aligned'] or not r['rowmap_ok']:
        r['note'] = ('alignment checksum FAILED' if not r['aligned']
                     else 'row map length != G04 trials')
        return r

    err_g04 = attach_behav(idx, behav)['i_sacc_err']
    b = performance_bins(err_g04, y, n_bins, scope='location')
    names = bin_labels(n_bins)
    r['bin_counts'] = [int((b == k).sum()) for k in range(n_bins)]
    r['bin_min_per_loc'] = [int(min(int(((y == l) & (b == k)).sum())
                                     for l in sorted(ANGLE_MAPPING)))
                            for k in range(n_bins)]
    r['n_excluded'] = int((b < 0).sum())
    r['bin_names'] = names
    # Would this subject survive at 2 / 3 / 4 bins? Computed from its OWN data
    # rather than projected, so the choice of n_bins can be made from fact.
    r['survives'] = {}
    for nb in (2, 3, 4):
        bb = performance_bins(err_g04, y, nb, scope='location')
        mins = [int(min(int(((y == l) & (bb == k)).sum())
                        for l in sorted(ANGLE_MAPPING))) for k in range(nb)]
        r['survives'][nb] = (min(mins) >= MIN_TRIALS_PER_LOC, mins)
    r['ok'] = True
    return r


def main():
    ap = argparse.ArgumentParser(description='Audit the behavioural file used for binning.')
    ap.add_argument('--subjects', nargs='+', type=int, default=SUBJECT_LIST)
    ap.add_argument('--voxRes', default='8mm')
    ap.add_argument('--roi', default='visual')
    ap.add_argument('--n_bins', type=int, default=3)
    ap.add_argument('--behav_dir', default='eyetracking',
                     help="Directory under derivatives/sub-XX/ to read (default "
                          "'eyetracking'). Nothing in this codebase reads any other "
                          "directory, so point this elsewhere to check whether the one "
                          "in place is the intended one.")
    ap.add_argument('--compare_raw', action='store_true',
                     help="With --dist, also read S02B's INPUT (..._iisess.mat) so it is "
                          "visible whether near-zero i_sacc_err comes from upstream iEye "
                          "or is introduced by S02B.")
    ap.add_argument('--compare_dir', default=None,
                     help="With --dist, also read this directory (e.g. eyetracking_old) "
                          "for a side-by-side comparison.")
    ap.add_argument('--dist', action='store_true',
                     help='Dump the distribution of i_sacc_err near zero, pooled across '
                          'subjects. This is what decides whether the threshold is '
                          'removing a separate no-saccade population (a GAP below the '
                          'cut) or trimming the low tail of real measurements (CONTINUITY '
                          'through the cut).')
    ap.add_argument('--logfile', default=None)
    args = ap.parse_args()

    bids_root = get_bids_root()

    if args.dist:
        def pull(behav_dir, raw):
            vals, missing = [], 0
            for sid in args.subjects:
                fp = behav_path(bids_root, sid, behav_dir, raw)
                if not os.path.exists(fp):
                    missing += 1
                    continue
                try:
                    f = open_h5(fp, os.path.basename(fp))
                except Exception:
                    missing += 1
                    continue
                try:
                    key = 'ii_sess' if raw else 'ii_sess_forSource'
                    grp = f[key] if key in f else f[list(f.keys())[0]]
                    v = np.array(grp['i_sacc_err']).flatten()
                    vals.append(np.asarray(v, float))
                except Exception:
                    missing += 1
                finally:
                    f.close()
            return (np.concatenate(vals) if vals else np.array([])), missing

        srcs = [(args.behav_dir, False, f'{args.behav_dir}/ ..._forSource.mat  (what the '
                                        f'analyses read)')]
        if args.compare_raw:
            srcs.append((args.behav_dir, True,
                         f'{args.behav_dir}/ ..._iisess.mat        (S02B INPUT)'))
        if args.compare_dir:
            srcs.append((args.compare_dir, False,
                         f'{args.compare_dir}/ ..._forSource.mat'))
            srcs.append((args.compare_dir, True,
                         f'{args.compare_dir}/ ..._iisess.mat'))

        for bd, raw, label in srcs:
            e_all, miss = pull(bd, raw)
            if e_all.size == 0:
                print(f'\n{label}\n  not found / unreadable for all subjects '
                      f'({miss} missing)')
                continue
            e = e_all[np.isfinite(e_all)]
            print(f'\n{label}')
            print(f'  {e_all.size} trials ({e_all.size - e.size} NaN, {miss} subjects '
                  f'missing this file)')
            print(f'  exactly 0: {int((e==0).sum())}   '
                  f'<=0.001: {int((e<=I_SACC_ERR_THRESH).sum())} '
                  f'({100*(e<=I_SACC_ERR_THRESH).mean():.1f}%)   '
                  f'median of >0.001: {np.median(e[e>I_SACC_ERR_THRESH]):.3g}')
        e = pull(args.behav_dir, False)[0]
        e = e[np.isfinite(e)]
        print()
        edges = [0, 1e-12, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2,
                 1e-1, 3e-1, 1.0, 3.0, 10.0, np.inf]
        print(f'Distribution of i_sacc_err across {len(pooled)} subjects '
              f'({e.size} finite trials). Threshold = {I_SACC_ERR_THRESH}.')
        print(f'\n{"range":>24s} {"count":>7s} {"%":>6s}')
        print('-' * 40)
        print(f'{"exactly 0":>24s} {int((e == 0).sum()):7d} {100*(e==0).mean():5.1f}%')
        for lo, hi in zip(edges[1:-1], edges[2:]):
            n = int(((e > lo) & (e <= hi)).sum())
            mark = '   <== THRESHOLD' if lo == 1e-3 else ''
            print(f'{f"({lo:g}, {hi:g}]":>24s} {n:7d} {100*n/e.size:5.1f}%{mark}')
        print()
        print('READ IT LIKE THIS:')
        print('  A GAP (bins just below the threshold populated, bins just above EMPTY,')
        print('  then the bulk far above) => a separate no-saccade population, correctly')
        print('  excluded.')
        print('  CONTINUITY (counts rising smoothly through the threshold into the bulk)')
        print('  => the threshold is cutting into real measurements, and because those')
        print('  are the SMALLEST errors it is removing the BEST-performing trials --')
        print('  which biases exactly the "best" bin this analysis is built around.')
        return

    rows = [audit(s, bids_root, args.voxRes, args.roi, args.n_bins) for s in args.subjects]

    out = []
    out.append(f'Behavioural audit | voxRes={args.voxRes} | roi={args.roi} | '
               f'n_bins={args.n_bins} | i_sacc_err valid if > {I_SACC_ERR_THRESH}')
    out.append('')
    hdr = (f"{'subj':>4s} {'behav':>6s} {'align':>5s} {'NaN':>5s} {'exact0':>7s} "
           f"{'0<x<=t':>7s} {'valid':>6s} {'%val':>5s} "
           f"{'err p5':>8s} {'err med':>8s} {'min/loc per bin':>16s}")
    out.append(hdr); out.append('-' * len(hdr))
    for r in rows:
        if not r['ok']:
            out.append(f"{r['subjID']:4d} " + ' ' * 52 + f"  {r.get('note','')}"
                       + (f"  [i_sacc_raw={r.get('i_sacc_raw_shape')}]"
                          if r.get('i_sacc_raw_shape') else ''))
            continue
        pct = 100.0 * r['n_valid'] / max(r['n_behav'], 1)
        out.append(
            f"{r['subjID']:4d} {r['n_behav']:6d} "
            f"{'yes' if r['aligned'] else 'NO':>5s} {r['n_nan']:5d} {r['n_zero']:7d} "
            f"{r['n_tiny']:7d} {r['n_valid']:6d} {pct:5.0f} "
            f"{r.get('err_p5', float('nan')):8.4g} {r.get('err_med', float('nan')):8.4g} "
            f"{str(r['bin_min_per_loc']):>16s}")

    ok = [r for r in rows if r['ok']]
    out.append('')
    out.append(f'Usable subjects: {len(ok)}/{len(rows)}')
    if ok:
        pv = [100.0 * r['n_valid'] / max(r['n_behav'], 1) for r in ok]
        out.append(f'  i_sacc_err valid: {min(pv):.0f}%-{max(pv):.0f}% of trials '
                   f'(median {np.median(pv):.0f}%)')
        mins = [m for r in ok for m in r['bin_min_per_loc']]
        out.append(f'  smallest per-location count in any bin: {min(mins)} '
                   f'(crossnobis needs >= 4; bins below that lose that location)')
        thin = [(r['subjID'], r['bin_min_per_loc']) for r in ok
                if min(r['bin_min_per_loc']) < 4]
        if thin:
            out.append(f'  subjects with a bin below 4 trials for some location '
                       f'({len(thin)}): ' + ', '.join(f'sub-{s:02d}{c}' for s, c in thin))
        else:
            out.append('  every subject/bin clears the 4-trial minimum for all locations.')
    if ok:
        out.append('')
        out.append('EXCLUSION BREAKDOWN -- is the threshold discarding real data?')
        tz = sum(r['n_zero'] for r in ok); tt = sum(r['n_tiny'] for r in ok)
        out.append(f'  exactly 0 (missing/unparsed saccade, correctly excluded): {tz}')
        out.append(f'  0 < err <= {I_SACC_ERR_THRESH} (real measurements the threshold '
                   f'discards): {tt}')
        tm = [r['tiny_med'] for r in ok if 'tiny_med' in r]
        p5s = [r['err_p5'] for r in ok if 'err_p5' in r]
        if tm and p5s:
            out.append(f'  excluded-value median across subjects: {np.median(tm):.3g}; '
                       f'retained 5th-percentile range: {min(p5s):.3g}-{max(p5s):.3g} '
                       f'({min(p5s)/max(np.median(tm), 1e-12):.0f}x-'
                       f'{max(p5s)/max(np.median(tm), 1e-12):.0f}x above it)')
            out.append('  A continuous distribution would place its 5th percentile just '
                       'above the cut. A large gap instead means the excluded values are '
                       'a SEPARATE population (no-saccade / unparsed), i.e. the threshold '
                       'is removing missing data rather than trimming the real low tail.')
        if tt == 0:
            out.append('  -> every excluded trial is an exact 0, so the threshold is only '
                       'removing missing data and is doing its job. The low valid% in some '
                       'subjects is genuinely missing eyetracking, not over-aggressive '
                       'thresholding.')
        else:
            out.append(f'  -> {tt} trials sit strictly between 0 and the threshold. Compare '
                       f'against the err p5 column: if p5 is orders of magnitude larger, '
                       f'these are still effectively zero; if comparable, the threshold is '
                       f'cutting into real data and should be lowered.')

        out.append('')
        out.append('BIN SURVIVAL -- subjects clearing the '
                   f'{MIN_TRIALS_PER_LOC}-trial floor for EVERY location in EVERY bin:')
        for nb in (2, 3, 4):
            good = [r['subjID'] for r in ok if r['survives'][nb][0]]
            lost = [r['subjID'] for r in ok if not r['survives'][nb][0]]
            out.append(f'  {nb} bins: {len(good):2d}/{len(ok)} survive' +
                       (f'   lost: ' + ', '.join(f'sub-{s:02d}' for s in lost) if lost else ''))

    bad = [r for r in rows if not r['ok']]
    if bad:
        out.append('')
        out.append('Subjects that would be SKIPPED by the binning:')
        for r in bad:
            out.append(f"  sub-{r['subjID']:02d}: {r.get('note','')}")
    out.append('')
    out.append('NOTE the "i_sacc_raw shape mismatch / i_sacc_angle will be NaN" warning '
               'seen during runs concerns i_sacc_ANGLE only. load_behav reads i_sacc_err '
               'separately and directly from the file, and the binning uses i_sacc_err, '
               'so that warning does not affect the bins. The raw shape is printed above '
               'for the affected subjects so this can be confirmed rather than assumed.')

    text = '\n'.join(out)
    print(text)
    logfile = args.logfile or os.path.join(
        bids_root, 'derivatives', 'glueDecoding', f'behaviour_audit_{args.voxRes}.log')
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    with open(logfile, 'w') as fh:
        fh.write(text + '\n')
    print(f'\nSaved: {logfile}')


if __name__ == '__main__':
    main()
