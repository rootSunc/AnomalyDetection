"""Benchmark PyOD detectors on ODDS real-world datasets.

Protocol (aligned with common 2019 PyOD benchmark practice):
  - train/test split 60/40
  - standardize features with training statistics
  - contamination = outlier ratio estimated on the full label vector
  - report ROC-AUC and Average Precision on the held-out test set
"""

from __future__ import print_function

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.utils.utility import standardizer
from sklearn.model_selection import train_test_split

from utils.data_loading import DEFAULT_DATASETS, load_odds_mat
from utils.metrics import evaluate_detector

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
warnings.filterwarnings('ignore')


def build_classifiers(contamination, random_state):
    return {
        'PCA': PCA(contamination=contamination, random_state=random_state),
        'MCD': MCD(contamination=contamination, random_state=random_state),
        'OCSVM': OCSVM(contamination=contamination),
        'KNN': KNN(contamination=contamination),
        'HBOS': HBOS(contamination=contamination),
        'LOF': LOF(n_neighbors=20, contamination=contamination),
        'CBLOF': CBLOF(
            contamination=contamination,
            check_estimator=False,
            random_state=random_state,
        ),
        'ABOD': ABOD(contamination=contamination),
        'IForest': IForest(contamination=contamination, random_state=random_state),
        'FeatureBagging': FeatureBagging(
            LOF(n_neighbors=20),
            contamination=contamination,
            random_state=random_state,
        ),
    }


def run_one_dataset(name, random_state=42, test_size=0.4):
    X, y, meta = load_odds_mat(name)
    contamination = max(meta['contamination'], 0.01)
    contamination = min(contamination, 0.5)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_test = standardizer(X_train, X_test)

    rows = []
    classifiers = build_classifiers(contamination, random_state)
    print(
        '\n=== {} | n={}, d={}, contamination={:.2%} ==='.format(
            meta['name'], meta['n_samples'], meta['n_features'], contamination
        )
    )

    for clf_name, clf in classifiers.items():
        t0 = time.time()
        try:
            clf.fit(X_train)
            y_pred = clf.predict(X_test)
            y_scores = clf.decision_function(X_test)
            metrics = evaluate_detector(y_test, y_pred, y_scores)
            elapsed = time.time() - t0
            print(
                '  {:>14s}  ROC-AUC={:.4f}  AP={:.4f}  F1={:.4f}  ({:.2f}s)'.format(
                    clf_name,
                    metrics['roc_auc'],
                    metrics['average_precision'],
                    metrics['f1'],
                    elapsed,
                )
            )
            row = {
                'dataset': meta['name'],
                'n_samples': meta['n_samples'],
                'n_features': meta['n_features'],
                'contamination': round(contamination, 4),
                'model': clf_name,
                'seconds': round(elapsed, 3),
            }
            row.update(metrics)
            rows.append(row)
        except Exception as exc:
            print('  {:>14s}  FAILED: {}'.format(clf_name, exc))

    return rows


def main(datasets, random_state=42):
    if not os.path.isdir(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    all_rows = []
    for name in datasets:
        all_rows.extend(run_one_dataset(name, random_state=random_state))

    if not all_rows:
        print('No results produced.')
        return

    df = pd.DataFrame(all_rows)
    out_path = os.path.join(RESULTS_DIR, 'odds_benchmark.csv')
    df.to_csv(out_path, index=False)
    print('\nSaved {}'.format(out_path))

    # pivot-style summary: mean ROC-AUC per model across datasets
    summary = (
        df.groupby('model')[['roc_auc', 'average_precision', 'f1']]
        .mean()
        .sort_values('roc_auc', ascending=False)
        .round(4)
    )
    print('\n=== Mean metrics across datasets ===')
    print(summary.to_string())
    summary_path = os.path.join(RESULTS_DIR, 'odds_benchmark_summary.csv')
    summary.to_csv(summary_path)
    print('Saved {}'.format(summary_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ODDS outlier detection benchmark')
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=list(DEFAULT_DATASETS),
        help='Dataset names under datasets/ (default: all bundled)',
    )
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args.datasets, random_state=args.seed)
