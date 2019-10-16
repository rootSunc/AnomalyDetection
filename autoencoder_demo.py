"""AutoEncoder-based outlier detection with PyOD (Keras backend).

Contrasts a non-linear deep reconstruction model against linear PCA:
points with large reconstruction error receive high anomaly scores.
"""

from __future__ import print_function

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.pca import PCA
from pyod.utils.data import generate_data
from pyod.utils.utility import standardizer

from utils.metrics import evaluate_detector, summarize_results

IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'img')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

# Quiet noisy TF1 / Keras logs for a cleaner CLI demo
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
warnings.filterwarnings('ignore')


def main():
    for path in (IMG_DIR, RESULTS_DIR):
        if not os.path.isdir(path):
            os.makedirs(path)

    contamination = 0.15
    X_train, y_train = generate_data(
        behaviour='new',
        n_train=800,
        n_features=10,
        train_only=True,
        contamination=contamination,
        random_state=42,
    )
    X_train = standardizer(X_train)

    # Linear baseline
    pca = PCA(contamination=contamination, random_state=42)
    pca.fit(X_train)
    pca_metrics = evaluate_detector(y_train, pca.labels_, pca.decision_scores_)

    # Non-linear AutoEncoder: encoder dims compress then reconstruct
    # hidden_neurons layout follows PyOD 0.7.x AutoEncoder API
    ae = AutoEncoder(
        hidden_neurons=[64, 32, 32, 64],
        epochs=30,
        batch_size=32,
        contamination=contamination,
        verbose=0,
        random_state=42,
    )
    ae.fit(X_train)
    ae_metrics = evaluate_detector(y_train, ae.labels_, ae.decision_scores_)

    results = {'PCA': pca_metrics, 'AutoEncoder': ae_metrics}
    print('=== PCA vs AutoEncoder ===')
    summarize_results(results, sort_by='roc_auc')

    df = pd.DataFrame(X_train[:, :2], columns=['f0', 'f1'])
    df['ae_score'] = ae.decision_scores_
    sns.scatterplot(x='f0', y='f1', hue='ae_score', data=df, palette='RdBu_r')
    plt.title('Anomaly Scores by AutoEncoder')
    plt.savefig(os.path.join(IMG_DIR, 'AutoEncoder_AnomalyScores.png'))
    plt.close()

    # Save a small CSV for the README / reports
    out = pd.DataFrame(
        [
            dict(model='PCA', **pca_metrics),
            dict(model='AutoEncoder', **ae_metrics),
        ]
    )
    csv_path = os.path.join(RESULTS_DIR, 'pca_vs_autoencoder.csv')
    out.to_csv(csv_path, index=False)
    print('Saved {}'.format(csv_path))
    print(
        'Mean AE reconstruction-based score: {:.4f}'.format(
            float(np.mean(ae.decision_scores_))
        )
    )


if __name__ == '__main__':
    main()
