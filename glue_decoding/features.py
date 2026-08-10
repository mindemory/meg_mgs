"""
features.py

Feature construction for the three glue_decoding conditions. See the plan's
"decision 4" for why the phase condition is [amp*cos(phase), amp*sin(phase)]
(real/imag of the analytic signal, 2x source count) and NOT
[amplitude, amp*cos(phase), amp*sin(phase)] (3x): amplitude is fully
recoverable from (real, imag) as sqrt(real**2 + imag**2), so including it
explicitly alongside its own decomposition double-counts it and inflates
feature count 3x vs. the 1x amplitude-only condition -- confounding any
"phase helps" result with a "more features helps" result under LOO.
"""

import numpy as np

VALID_CONDITIONS = ('unfiltered', 'ampOnly', 'ampPhase')


def build_features(condition, amp, phase=None):
    """
    amp, phase: (n_trials, n_times, n_sources) arrays (already ROI-sliced).

    'unfiltered' / 'ampOnly' -> amp unchanged, (n_trials, n_times, n_sources).
    'ampPhase'   -> concat([amp*cos(phase), amp*sin(phase)], axis=-1),
                    (n_trials, n_times, 2*n_sources).
    """
    if condition not in VALID_CONDITIONS:
        raise ValueError(f'Unknown condition {condition!r}, expected one of {VALID_CONDITIONS}')

    if condition in ('unfiltered', 'ampOnly'):
        return amp

    if phase is None:
        raise ValueError("condition='ampPhase' requires phase data, got None "
                         "(this band likely has no saved phase -- lowgamma/highgamma)")
    if phase.shape != amp.shape:
        raise ValueError(f'amp shape {amp.shape} != phase shape {phase.shape}')

    real = amp * np.cos(phase)
    imag = amp * np.sin(phase)
    return np.concatenate([real, imag], axis=-1)
