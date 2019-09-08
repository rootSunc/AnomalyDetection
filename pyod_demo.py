import matplotlib.font_manager
import matplotlib.pyplot as plt
import numpy as np
from pyod.models.knn import KNN
from pyod.utils.data import generate_data, get_outliers_inliers
from scipy import stats

if __name__ == '__main__':

    # generate estimated training data
    # X_train -> training data
    # y_train -> training ground truth
    X_train, y_train = generate_data(n_train=300, n_features=2, contamination=0.2, train_only=True, random_state=20)
    outlier_fraction = 0.2

    X_outliers, X_inliers = get_outliers_inliers(X_train, y_train)
    n_outliers = len(X_outliers)
    n_inliers = len(X_inliers)


    F1 = X_train[:,0].reshape(-1,1)
    F2 = X_train[:,1].reshape(-1,1)
    plt.scatter(F1, F2)
    plt.xlabel('F1')
    plt.ylabel('F2')
    plt.show()

    '''
        KNN -> K-Nearest Neighbors Detector
        For an observation, its distance to its kth nearest neighbors could be viewed as the outlying scores
        Method: -Largest -Average -Median
    '''
    clf_name = 'KNN'
    clf = KNN()
    clf.fit(X_train)

    X_train_pred = clf.labels_
    X_train_score = clf.decision_scores_

    score_pred = clf.decision_function(X_train)*-1
    y_pred = clf.predict(X_train)
    n_errors = (y_pred != y_train).sum()
    print('No of Errors:', clf_name, n_errors)

    # visualization
    xx, yy = np.meshgrid(np.linspace(-10, 10, 300), np.linspace(-10, 10, 300))
    threshold = stats.scoreatpercentile(score_pred, 100*outlier_fraction)
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
    Z = Z.reshape(xx.shape)
    # fill blue colormap from minimum anomaly score to threshold value
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 10), cmap=plt.cm.Blues_r)
    a = plt.contour(xx, yy, Z, levels=[threshold],linewidths=2, colors='red')
    plt.contourf(xx, yy, Z, levels=[threshold, Z.max()],colors='orange')
    b = plt.scatter(X_train[:-n_outliers, 0], X_train[:-n_outliers, 1], c='white',s=20, edgecolor='k')
    c = plt.scatter(X_train[-n_outliers:, 0], X_train[-n_outliers:, 1], c='black',s=20, edgecolor='k')
    plt.axis('tight')

    plt.legend(
        [a.collections[0], b, c],
        ['learned decision function', 'true inliers', 'true outliers'],
        prop=matplotlib.font_manager.FontProperties(size=10),
        loc='lower right')

    plt.title(clf_name)
    plt.xlim((-10, 10))
    plt.ylim((-10, 10))
    plt.show()