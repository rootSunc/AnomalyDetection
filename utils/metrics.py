"""Evaluation helpers for unsupervised outlier detectors (PyOD-style)."""

from __future__ import print_function

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_detector(y_true, y_pred, y_scores, digits=4):
    """Compute common ranking / classification metrics for anomaly detection.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        Ground-truth labels (0 = inlier, 1 = outlier).
    y_pred : array-like, shape (n_samples,)
        Binary predictions from the detector.
    y_scores : array-like, shape (n_samples,)
        Continuous anomaly scores (higher = more abnormal).
    digits : int
        Rounding precision for the returned dict.

    Returns
    -------
    dict
        roc_auc, average_precision, precision, recall, f1
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    y_scores = np.asarray(y_scores).ravel()

    # sklearn<=0.21 has no zero_division kwarg; guard empty positive class
    if y_pred.sum() == 0:
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    else:
        precision = float(precision_score(y_true, y_pred))
        recall = float(recall_score(y_true, y_pred))
        f1 = float(f1_score(y_true, y_pred))

    metrics = {
        'roc_auc': float(roc_auc_score(y_true, y_scores)),
        'average_precision': float(average_precision_score(y_true, y_scores)),
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }
    return {k: round(v, digits) for k, v in metrics.items()}


def summarize_results(results, sort_by='roc_auc'):
    """Pretty-print a mapping of detector_name -> metrics dict."""
    if not results:
        print('No results to summarize.')
        return

    headers = ['model', 'roc_auc', 'avg_prec', 'precision', 'recall', 'f1']
    rows = []
    for name, m in results.items():
        rows.append([
            name,
            m['roc_auc'],
            m['average_precision'],
            m['precision'],
            m['recall'],
            m['f1'],
        ])

    rows.sort(key=lambda r: r[headers.index(sort_by)], reverse=True)

    col_widths = [
        max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))
    ]

    def fmt(row):
        return '  '.join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))

    print(fmt(headers))
    print('  '.join('-' * w for w in col_widths))
    for row in rows:
        print(fmt(row))
