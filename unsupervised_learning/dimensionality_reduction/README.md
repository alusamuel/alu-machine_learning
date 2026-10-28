# Dimensionality reduction

PCA implementations using NumPy singular value decomposition:

- `0-pca.py` returns projection weights for an already centered dataset. The
  component count uses cumulative singular values to match the task example.
- `1-pca.py` centers the dataset and returns its projection onto a specified
  number of principal components.

Both functions expect valid inputs as described in the assignment.
