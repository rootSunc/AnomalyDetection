import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    plt.savefig("./img/PCA_GroundTruth.png")
    plt.show()

    # train an unsupervised PCA
    clf = PCA()
    clf.fit(X_train)

    # training score
    y_train_pred = clf.labels_  # inliers -> 0; outliers -> 1
    y_train_scores = clf.decision_scores_
    sns.scatterplot(x=0, y=1, hue=y_train_scores, data=df_train, palette='RdBu_r');
    plt.title('Anomaly Scores by PCA');
    plt.savefig("./img/PCA_AnomalyScores.png")
    plt.show()

