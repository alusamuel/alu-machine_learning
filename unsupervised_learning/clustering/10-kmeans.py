#!/usr/bin/env python3
"""K-means using scikit-learn."""
import sklearn.cluster


def kmeans(X, k):
    """Fit K-means and return its centers and labels."""
    model = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    return model.cluster_centers_, model.labels_
