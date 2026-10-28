# Clustering

NumPy implementations of K-means, variance-based cluster selection, Gaussian
mixture expectation maximization, and Bayesian Information Criterion selection.
Tasks 10–12 use scikit-learn and SciPy for clustering and Matplotlib to display
a Ward-linkage dendrogram.

Dependencies: NumPy, SciPy, scikit-learn, and Matplotlib.

Import numbered modules from this directory with `__import__`, for example:

```python
kmeans = __import__('1-kmeans').kmeans
centroids, labels = kmeans(X, 3)
```

`11-gmm.py` returns the scalar BIC for the fitted model, as in the task example.
