"""
svr_tgm.py

Temporal Generalization Matrix (TGM) decoding, adapted from
megScripts/temporalGeneralizationDecoding.py's _process_train_t/run_tgm.
Same math (2x SVR for sin/cos angle regression, Leave-One-Out over trials,
z-scored per train-timepoint, recombined via arctan2), but n_jobs is an
explicit parameter that glue_decoding ALWAYS calls with n_jobs=1.

Why: glue_decoding's parallelism lives at the outer grid level (one process
per subject x lockType, fanned out by run_glue_decoding.sh across vader's
cores). If this module's joblib.Parallel also spun up multiple workers per
process, the two parallel levels would oversubscribe cores and run slower
than serial -- see run_glue_decoding.sh's OMP/MKL/OPENBLAS_NUM_THREADS=1
exports, which address the same oversubscription risk one level down (BLAS
calls inside a single sklearn fit).
"""

import numpy as np
from joblib import Parallel, delayed
from sklearn.svm import SVR

from constants import ANGLE_MAPPING


def _process_train_t(train_t, data_matrix, sin_targets, cos_targets):
    """
    Train SVR (LOO) at train_t, test on ALL test time points.
    Returns (train_t, pred_angles_deg) where pred_angles_deg is (n_trials, n_test_t).
    """
    n_trials, n_test_t, n_sources = data_matrix.shape

    X_train_all = data_matrix[:, train_t, :]  # (trials, sources)
    mu = X_train_all.mean(axis=0)
    sd = X_train_all.std(axis=0) + 1e-10
    X_z = (X_train_all - mu) / sd

    pred_angles = np.zeros((n_trials, n_test_t))

    svr_sin = SVR(kernel='rbf')
    svr_cos = SVR(kernel='rbf')

    for left_out in range(n_trials):
        train_mask = np.ones(n_trials, dtype=bool)
        train_mask[left_out] = False

        svr_sin.fit(X_z[train_mask], sin_targets[train_mask])
        svr_cos.fit(X_z[train_mask], cos_targets[train_mask])

        for test_t in range(n_test_t):
            x_test = (data_matrix[left_out, test_t, :] - mu) / sd
            pred_sin = svr_sin.predict(x_test.reshape(1, -1))[0]
            pred_cos = svr_cos.predict(x_test.reshape(1, -1))[0]
            pred_angles[left_out, test_t] = np.degrees(
                np.mod(np.arctan2(pred_sin, pred_cos), 2 * np.pi))

    return train_t, pred_angles  # (n_trials, n_test_t)


def run_tgm(data_matrix, target_labels, n_jobs=1, control=False):
    """
    Full Temporal Generalization Matrix. data_matrix: (n_trials, n_train_t, n_sources).
    Returns pred_angles of shape (n_trials, n_train_t, n_test_t). No error
    computation here -- that's aggregate_glue_decoding.py's job.

    n_jobs is explicit and glue_decoding always passes n_jobs=1 -- see module
    docstring. It is NOT derived from any host/core-count heuristic here.
    """
    if control:
        target_labels = np.random.permutation(target_labels)

    angles_rad = np.array([np.radians(ANGLE_MAPPING[int(t)]) for t in target_labels])
    sin_targets = np.sin(angles_rad)
    cos_targets = np.cos(angles_rad)

    n_trials, n_train_t, _ = data_matrix.shape

    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_process_train_t)(tr_t, data_matrix, sin_targets, cos_targets)
        for tr_t in range(n_train_t)
    )

    pred_angles = np.zeros((n_trials, n_train_t, n_train_t))
    for tr_t, preds in results:
        pred_angles[:, tr_t, :] = preds

    return pred_angles
