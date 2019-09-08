import pandas as pd
import matplotlib.pyplot as plt
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


if __name__ == "__main__":

    # generate training data
    X_train, y_train = generate_data(behaviour='new',
                                     n_features=5,
                                     train_only=True,
                                     contamination=0.15,
                                     random_state=13)
    df_train = pd.DataFrame(X_train)
    df_train['y'] = y_train
    sns.scatterplot(x=0, y=1, hue='y', data=df_train);
    plt.title('Ground Truth')
    plt.show()

    # Define nine outlier detection tools to be compared
    random_state = 11
    outliers_fraction = 0.15
    classifiers = {
        'Cluster-based Local Outlier Factor (CBLOF)':
            CBLOF(contamination=outliers_fraction,
                  check_estimator=False, random_state=random_state),
        'Feature Bagging':
            FeatureBagging(LOF(n_neighbors=35),
                           contamination=outliers_fraction,
                           random_state=random_state),
        'Histogram-base Outlier Detection (HBOS)': HBOS(
            contamination=outliers_fraction),
        'Isolation Forest': IForest(contamination=outliers_fraction,
                                    random_state=random_state),
        'K Nearest Neighbors (KNN)': KNN(
            contamination=outliers_fraction),
        'Average KNN': KNN(method='mean',
                           contamination=outliers_fraction),
        'Local Outlier Factor (LOF)':
            LOF(n_neighbors=35, contamination=outliers_fraction),
        'Minimum Covariance Determinant (MCD)': MCD(
            contamination=outliers_fraction, random_state=random_state),
        'One-class SVM (OCSVM)': OCSVM(contamination=outliers_fraction),

    }

    for i, clf in enumerate(classifiers.keys()):
        print('Model', i + 1, clf)

    # train an unsupervised model methods
    for i, (clf_name, clf) in enumerate(classifiers.items()):
        print(i + 1, 'fitting', clf_name)
        # fit the data and tag outliers
        clf.fit(X_train)

        # training score
        y_train_pred = clf.labels_  # inliers -> 0; outliers -> 1
        y_train_scores = clf.decision_scores_
        sns.scatterplot(x=0, y=1, hue=y_train_scores, data=df_train, palette='RdBu_r');
        plt.title('Anomaly Scores by %s' %clf_name);
        plt.savefig("./img/%s_AnomalyScores.png" %clf_name)
        plt.show()

