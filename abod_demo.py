"""Angle-Based Outlier Detection (ABOD) demo with PyOD."""

from __future__ import print_function

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pyod.models.abod import ABOD
from pyod.utils.data import generate_data

IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'img')


def main():
    if not os.path.isdir(IMG_DIR):
        os.makedirs(IMG_DIR)

    X_train, y_train = generate_data(
        behaviour='new',
        n_features=5,
        train_only=True,
        contamination=0.15,
        random_state=13,
    )
    df_train = pd.DataFrame(X_train)
    df_train['y'] = y_train

    clf = ABOD(contamination=0.15)
    clf.fit(X_train)

    y_train_scores = clf.decision_scores_
    sns.scatterplot(x=0, y=1, hue=y_train_scores, data=df_train, palette='RdBu_r')
    plt.title('Anomaly Scores by ABOD')
    plt.savefig(os.path.join(IMG_DIR, 'ABOD_AnomalyScores.png'))
    plt.close()

    n_outliers = int(clf.labels_.sum())
    print('ABOD flagged {} / {} points as outliers'.format(n_outliers, len(y_train)))


if __name__ == '__main__':
    main()
