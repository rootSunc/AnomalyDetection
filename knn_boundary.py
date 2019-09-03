"""kNN decision-boundary visualization for 2D synthetic outliers."""

from __future__ import print_function

import os

import matplotlib.font_manager
import matplotlib.pyplot as plt
import numpy as np
from pyod.models.knn import KNN
from pyod.utils.data import generate_data, get_outliers_inliers
from scipy import stats

IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'img')


def main():
    if not os.path.isdir(IMG_DIR):
        os.makedirs(IMG_DIR)

    X_train, y_train = generate_data(
        n_train=300,
        n_features=2,
        contamination=0.2,
        train_only=True,
        random_state=20,
    )
    outlier_fraction = 0.2
    X_outliers, X_inliers = get_outliers_inliers(X_train, y_train)
    n_outliers = len(X_outliers)

    clf_name = 'KNN'
    clf = KNN(contamination=outlier_fraction)
    clf.fit(X_train)

    score_pred = clf.decision_function(X_train) * -1
    y_pred = clf.predict(X_train)
    n_errors = (y_pred != y_train).sum()
    print('Number of errors ({}): {}'.format(clf_name, n_errors))

    xx, yy = np.meshgrid(np.linspace(-10, 10, 300), np.linspace(-10, 10, 300))
    threshold = stats.scoreatpercentile(score_pred, 100 * outlier_fraction)
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 10), cmap=plt.cm.Blues_r)
    a = plt.contour(xx, yy, Z, levels=[threshold], linewidths=2, colors='red')
    plt.contourf(xx, yy, Z, levels=[threshold, Z.max()], colors='orange')
    b = plt.scatter(
        X_train[:-n_outliers, 0],
        X_train[:-n_outliers, 1],
        c='white',
        s=20,
        edgecolor='k',
    )
    c = plt.scatter(
        X_train[-n_outliers:, 0],
        X_train[-n_outliers:, 1],
        c='black',
        s=20,
        edgecolor='k',
    )
    plt.axis('tight')
    plt.legend(
        [a.collections[0], b, c],
        ['learned decision function', 'true inliers', 'true outliers'],
        prop=matplotlib.font_manager.FontProperties(size=10),
        loc='lower right',
    )
    plt.title(clf_name)
    plt.xlim((-10, 10))
    plt.ylim((-10, 10))
    plt.savefig(os.path.join(IMG_DIR, 'KNN_DecisionBoundary.png'), dpi=120)
    plt.close()
    print('Saved decision boundary figure.')


if __name__ == '__main__':
    main()
