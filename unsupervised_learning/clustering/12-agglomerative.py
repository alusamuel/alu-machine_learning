#!/usr/bin/env python3
"""Ward agglomerative clustering and dendrogram visualization."""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """Display the Ward dendrogram and return distance-threshold labels."""
    linkage = scipy.cluster.hierarchy.linkage(X, method='ward')
    clss = scipy.cluster.hierarchy.fcluster(
        linkage, dist, criterion='distance')
    scipy.cluster.hierarchy.dendrogram(linkage, color_threshold=dist)
    plt.show()
    return clss
