"""Load ODDS-style MATLAB datasets used in classic outlier benchmarks."""

from __future__ import print_function

import os

import numpy as np
from scipy.io import loadmat

DATASETS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets'
)

# Common ODDS benchmarks shipped / documented with this repo
DEFAULT_DATASETS = ('cardio', 'ionosphere', 'arrhythmia', 'pima')


def list_local_datasets(datasets_dir=None):
    """Return sorted basenames (without .mat) available under datasets/."""
    root = datasets_dir or DATASETS_DIR
    if not os.path.isdir(root):
        return []
    names = []
    for fname in os.listdir(root):
        if fname.endswith('.mat'):
            names.append(fname[:-4])
    return sorted(names)


def load_odds_mat(name, datasets_dir=None):
    """Load an ODDS .mat file.

    Parameters
    ----------
    name : str
        Dataset name with or without ``.mat`` suffix (e.g. ``'cardio'``).
    datasets_dir : str, optional
        Directory containing ``*.mat`` files.

    Returns
    -------
    X : ndarray, shape (n_samples, n_features)
    y : ndarray, shape (n_samples,)
        Binary labels (0 = inlier, 1 = outlier).
    meta : dict
        name, n_samples, n_features, contamination
    """
    root = datasets_dir or DATASETS_DIR
    base = name if name.endswith('.mat') else name + '.mat'
    path = os.path.join(root, base)
    if not os.path.isfile(path):
        raise IOError(
            'Dataset not found: {}. Place ODDS .mat files under {} '
            '(see datasets/README.md).'.format(path, root)
        )

    mat = loadmat(path)
    if 'X' not in mat or 'y' not in mat:
        raise ValueError('Expected keys X and y in {}'.format(path))

    X = np.asarray(mat['X'], dtype=np.float64)
    y = np.asarray(mat['y']).ravel().astype(int)
    contamination = float(np.count_nonzero(y)) / float(len(y))
    meta = {
        'name': base[:-4],
        'path': path,
        'n_samples': int(X.shape[0]),
        'n_features': int(X.shape[1]),
        'n_outliers': int(np.count_nonzero(y)),
        'contamination': contamination,
    }
    return X, y, meta


def describe_dataset(name, datasets_dir=None):
    """Print a one-line summary for a local ODDS dataset."""
    _, _, meta = load_odds_mat(name, datasets_dir=datasets_dir)
    print(
        '{name}: n={n_samples}, d={n_features}, '
        'outliers={n_outliers} ({contam:.2f}%)'.format(
            name=meta['name'],
            n_samples=meta['n_samples'],
            n_features=meta['n_features'],
            n_outliers=meta['n_outliers'],
            contam=100.0 * meta['contamination'],
        )
    )
    return meta
