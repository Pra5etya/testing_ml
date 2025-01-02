# **1. Principal Component Analysis (PCA)**
PCA is a linear dimensionality reduction technique that aims to maximize variance while reducing the number of dimensions.

```python
param_grid_pca = {
    'n_components': [2, 5, 10, 0.95],        # Number of principal components or explained variance ratio
    'svd_solver': ['auto', 'full', 'arpack', 'randomized'], # Solver for PCA decomposition
}
```

### Additional Details
- **`n_components`**: Can be an integer (number of components) or a float (variance ratio to retain). Use `0.95` to retain 95% of variance. Evaluate the choice of `n_components` using explained variance ratio plots.
- **`svd_solver='randomized'`**: Faster for large datasets but may be less accurate compared to `full` or `arpack`.
- PCA assumes linear relationships in data. Check for multicollinearity before applying PCA.

---

# **2. Linear Discriminant Analysis (LDA)**
LDA is a supervised dimensionality reduction technique that maximizes class separability.

```python
param_grid_lda = {
    'n_components': [1, 2, 5],             # Number of components (less than number of classes - 1)
    'solver': ['svd', 'lsqr', 'eigen'],    # Solver for computation
}
```

### Additional Details
- **`n_components`**: Must be less than the number of classes minus one.
- **`solver='svd'`**: Recommended for large datasets.
- **`solver='eigen'`**: Suitable for small datasets or when the covariance matrix is non-singular.
- Ensure the target variable has more than one class to apply LDA.

---

# **3. Truncated SVD**
Truncated SVD is used for dimensionality reduction of sparse datasets.

```python
param_grid_svd = {
    'n_components': [2, 5, 10],          # Number of components
}
```

### Additional Details
- **`n_components`**: Determines the dimensionality of the reduced data. Use grid search to optimize this parameter.
- Unlike PCA, Truncated SVD does not center the data, making it suitable for sparse matrices like TF-IDF representations.

---

# **4. t-SNE**
t-SNE is a non-linear dimensionality reduction technique that preserves local structure.

```python
param_grid_tsne = {
    'n_components': [2, 3],               # Output dimensions
    'perplexity': [5, 30, 50],           # Balances local and global structure
    'learning_rate': [10, 200, 1000],    # Step size for optimization
    'n_iter': [1000, 2000, 5000],        # Number of optimization iterations
}
```

### Additional Details
- **`perplexity`**: Small values focus on local details; large values consider global structure. Rule of thumb: should be less than the number of data points.
- **`learning_rate`**: Too high or low values can lead to convergence issues. Experiment with values in the range 50–200 for most datasets.
- **`n_iter`**: Higher values lead to better convergence but increase computation time. Check convergence with `n_iter_without_progress` parameter.

---

# **5. UMAP**
UMAP is a non-linear dimensionality reduction technique suitable for both global and local structure preservation.

```python
param_grid_umap = {
    'n_neighbors': [5, 15, 50],          # Size of local neighborhood
    'min_dist': [0.1, 0.5, 1.0],         # Minimum distance between points
    'metric': ['euclidean', 'manhattan', 'cosine'], # Distance metric
}
```

### Additional Details
- **`n_neighbors`**: Smaller values emphasize local structure, while larger values focus on global relationships. Values between 10–50 often work well.
- **`min_dist`**: Lower values create tighter clusters; higher values spread points apart. Use smaller values for visualization.
- **`metric`**: Choose based on the nature of your data. `euclidean` for continuous data, `cosine` for sparse or high-dimensional data.

---

# **6. Kernel PCA**
Kernel PCA extends PCA to handle non-linear relationships using kernel functions.

```python
param_grid_kernel_pca = {
    'n_components': [2, 5, 10],          # Number of components
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], # Kernel type
    'gamma': [None, 0.1, 1],             # Kernel coefficient (for rbf, poly, sigmoid)
}
```

### Additional Details
- **`kernel`**: Choose `rbf` for Gaussian-like structures, `poly` for polynomial relationships, or `linear` for simple linear separation.
- **`gamma`**: Lower values expand the influence of data points. Use `auto` for small datasets.
- Ensure data is normalized or standardized before applying Kernel PCA.

---

# **7. Isomap**
Isomap preserves geodesic distances and is suitable for non-linear manifolds.

```python
param_grid_isomap = {
    'n_components': [2, 5, 10],          # Number of components
    'n_neighbors': [5, 10, 20],         # Number of neighbors to construct manifold
}
```

### Additional Details
- **`n_neighbors`**: Determines the balance between local and global structure. Experiment with values to preserve geodesic distance accurately.
- Isomap works well on smooth manifolds but may struggle with noisy data.

---

# **8. Locally Linear Embedding (LLE)**
LLE preserves local relationships by reducing dimensions.

```python
param_grid_lle = {
    'n_components': [2, 5, 10],          # Number of components
    'n_neighbors': [5, 10, 20],         # Number of neighbors
    'method': ['standard', 'modified', 'hessian', 'ltsa'], # LLE method
}
```

### Additional Details
- **`method`**: Use `modified` for better handling of noisy data or `ltsa` for global structure.
- **`n_neighbors`**: Too low values may fail to capture local structure; too high may oversmooth data.
- Standardize data before applying LLE.

---

# **9. Multi-Dimensional Scaling (MDS)**
MDS reduces dimensions by preserving pairwise distances.

```python
param_grid_mds = {
    'n_components': [2, 5, 10],          # Number of components
    'metric': [True, False],            # Metric MDS (True) or non-metric MDS (False)
    'n_init': [4, 10],                  # Number of initializations
}
```

### Additional Details
- **`metric`**: Use `True` for preserving distances or `False` for rank order.
- **`n_init`**: Higher values ensure better solutions but increase computation time. Recommended: 10.
- Works best on datasets with well-defined distances.

---

# **10. Variance Threshold**
Variance Threshold removes features with low variance.

```python
param_grid_variance_threshold = {
    'threshold': [0.0, 0.01, 0.1],       # Variance threshold
}
```

### Additional Details
- **`threshold`**: Higher values remove more features. Set based on the expected variance of important features. Use with caution for datasets with imbalanced scales.

---

# **11. SelectKBest**
SelectKBest selects features with the highest scores.

```python
param_grid_selectkbest = {
    'k': [5, 10, 20],                    # Number of top features to select
    'score_func': ['f_classif', 'mutual_info_classif'], # Scoring function
}
```

### Additional Details
- **`score_func`**: Choose `f_classif` for ANOVA F-value or `mutual_info_classif` for mutual information. For regression, use `f_regression`.
- Use cross-validation to evaluate feature importance and avoid overfitting.

---

# **12. Recursive Feature Elimination (RFE)**
RFE selects features recursively by eliminating the least important ones.

```python
param_grid_rfe = {
    'n_features_to_select': [5, 10, None], # Number of features to select
    'step': [1, 2, 5],                   # Number of features to eliminate per iteration
}
```

### Additional Details
- **`step`**: Higher values speed up computation but may skip important features.
- Pair RFE with strong base models like SVMs or Random Forests for better feature ranking.
- Evaluate feature importance using metrics like cross-validation accuracy after selection.




# Penerapan parameter yang baik
```python
param_grid = [
    {
        'param_1'
    }, 
    {
        'param_2'
    }
]
```