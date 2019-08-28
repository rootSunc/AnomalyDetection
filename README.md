# Anomaly Detection with PyOD

Unsupervised outlier detection experiments in Python ([PyOD](https://github.com/yzhao062/pyod)).

## Setup

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
python pca_demo.py
python abod_demo.py
```

## Model Taxonomy

1. **Linear**: PCA, MCD, OCSVM
2. **Proximity**: kNN, HBOS
3. **Density**: LOF, CBLOF, ABOD
4. **Ensemble**: Isolation Forest, Feature Bagging, LSCP

## Model Notes

### PCA
Linear encode/decode via principal components; weighted reconstruction distance as score.

![PCA](resources/PCA_arch.jpg)

### AutoEncoder
Non-linear multi-layer reconstruction; larger error ⇒ more anomalous.

![AutoEncoder](resources/AutoEncoder_arch.jpg)

### MCD
Robust covariance via minimum determinant subset; Mahalanobis distance scoring.

### OCSVM
Learn a boundary around normal points in kernel space.

### kNN
Distance to the k-th neighbor (or mean/median of k distances).

### LOF
Ratio of local reachability densities versus neighbors.
