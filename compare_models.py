"""Compare multiple PyOD detectors on the same synthetic dataset.

Runs classic linear / proximity / density / ensemble methods, saves score
scatter plots, and prints a ranking table (ROC-AUC, AP, Precision, Recall, F1).
"""

from __future__ import print_function

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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
from pyod.utils.data import generate_data

from utils.metrics import evaluate_detector, summarize_results

IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'img')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def build_classifiers(contamination, random_state):
    return {
        'PCA': PCA(contamination=contamination, random_state=random_state),
        'MCD': MCD(contamination=contamination, random_state=random_state),
        'OCSVM': OCSVM(contamination=contamination),
        'KNN': KNN(contamination=contamination),
        'Average_KNN': KNN(method='mean', contamination=contamination),
        'HBOS': HBOS(contamination=contamination),
        'LOF': LOF(n_neighbors=35, contamination=contamination),
        'CBLOF': CBLOF(
            contamination=contamination,
            check_estimator=False,
            random_state=random_state,
        ),
        'ABOD': ABOD(contamination=contamination),
        'Isolation_Forest': IForest(
            contamination=contamination, random_state=random_state
        ),
        'Feature_Bagging': FeatureBagging(
            LOF(n_neighbors=35),
            contamination=contamination,
            random_state=random_state,
        ),
    }


def main(save_plots=True):
    for path in (IMG_DIR, RESULTS_DIR):
        if not os.path.isdir(path):
            os.makedirs(path)

    contamination = 0.15
    random_state = 11

    X_train, y_train = generate_data(
        behaviour='new',
        n_features=5,
        train_only=True,
        contamination=contamination,
        random_state=13,
    )
    df_train = pd.DataFrame(X_train)
    df_train['y'] = y_train

    if save_plots:
        sns.scatterplot(x=0, y=1, hue='y', data=df_train)
        plt.title('Ground Truth')
        plt.savefig(os.path.join(IMG_DIR, 'GroundTruth.png'))
        plt.close()

    classifiers = build_classifiers(contamination, random_state)
    results = {}

    for i, (clf_name, clf) in enumerate(classifiers.items()):
        print('[{}/{}] fitting {}'.format(i + 1, len(classifiers), clf_name))
        clf.fit(X_train)

        y_pred = clf.labels_
        y_scores = clf.decision_scores_
        results[clf_name] = evaluate_detector(y_train, y_pred, y_scores)

        if save_plots:
            sns.scatterplot(
                x=0, y=1, hue=y_scores, data=df_train, palette='RdBu_r'
            )
            plt.title('Anomaly Scores by {}'.format(clf_name))
            out_path = os.path.join(IMG_DIR, '{}_AnomalyScores.png'.format(clf_name))
            plt.savefig(out_path)
            plt.close()

    print('\n=== Ranking (sorted by ROC-AUC) ===')
    summarize_results(results, sort_by='roc_auc')

    rows = []
    for name, m in results.items():
        row = {'model': name}
        row.update(m)
        rows.append(row)
    df = pd.DataFrame(rows).sort_values('roc_auc', ascending=False)
    csv_path = os.path.join(RESULTS_DIR, 'synthetic_comparison.csv')
    df.to_csv(csv_path, index=False)
    print('\nSaved metrics to {}'.format(csv_path))
    return results


if __name__ == '__main__':
    # optional: python compare_models.py --no-plots
    save_plots = '--no-plots' not in sys.argv
    main(save_plots=save_plots)
