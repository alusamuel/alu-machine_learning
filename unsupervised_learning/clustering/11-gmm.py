#!/usr/bin/env python3
"""Gaussian mixture clustering using scikit-learn."""
import sklearn.mixture


def gmm(X, k):
    """Return fitted priors, means, covariances, labels, and model BIC."""
    model = sklearn.mixture.GaussianMixture(n_components=k).fit(X)
    return (model.weights_, model.means_, model.covariances_,
            model.predict(X), model.bic(X))
