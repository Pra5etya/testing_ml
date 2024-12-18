# Hyperparameter Grid for Clustering Models in scikit-learn

This document provides a comprehensive hyperparameter grid for clustering models available in scikit-learn, along with detailed explanations of each hyperparameter.

---

## **1. K-Means**
K-Means is a centroid-based clustering algorithm that aims to minimize intra-cluster distances.

```python
param_grid_kmeans = {
    'n_clusters': [2, 3, 5, 8, 10],       # Number of clusters
    'init': ['k-means++', 'random'],      # Initialization method for centroids
    'n_init': [10, 20, 50],               # Number of different initializations
    'max_iter': [300, 500, 1000],         # Maximum number of iterations
    'tol': [1e-4, 1e-3, 1e-2],            # Convergence tolerance
    'algorithm': ['auto', 'full', 'elkan'], # Optimization algorithm
}
```

### Additional Details
- **`n_clusters`**: Higher values may lead to overfitting; evaluate with metrics like Silhouette Score or Elbow Method.
- **`algorithm='elkan'`**: Recommended for dense datasets with low-to-medium dimensions.

---

## **2. Hierarchical Clustering (AgglomerativeClustering)**  
Hierarchical clustering builds a dendrogram, and each level represents different clustering granularities.

```python
param_grid_agglomerative = {
    'n_clusters': [2, 3, 5, 8, 10],           # Number of clusters
    'affinity': ['euclidean', 'manhattan', 'cosine'], # Distance metric
    'linkage': ['ward', 'complete', 'average', 'single'], # Linkage method
    'distance_threshold': [None, 5, 10],      # Maximum distance for clusters
    'compute_full_tree': [True, False],       # Whether to compute the full dendrogram
}
```

### Additional Details
- **`distance_threshold`**: Useful when the number of clusters is undefined (`n_clusters=None`).
- **`linkage='ward'`**: Only supports **Euclidean distance** and minimizes within-cluster variance.

---

## **3. DBSCAN**  
DBSCAN is a density-based clustering algorithm, ideal for non-uniform data shapes.

```python
param_grid_dbscan = {
    'eps': [0.1, 0.5, 1.0, 2.0],          # Maximum distance for neighbors
    'min_samples': [5, 10, 20],           # Minimum points to form a cluster
    'metric': ['euclidean', 'manhattan', 'cosine', 'minkowski'], # Distance metric
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'], # Nearest neighbor algorithm
    'leaf_size': [10, 30, 50],            # Leaf size for tree-based methods
    'p': [1, 2],                          # Power parameter for Minkowski distance
}
```

### Additional Details
- **`eps`**: Crucial for controlling density. Small values lead to noise, while large values merge clusters.
- **`min_samples`**: Minimum number of points required within the `eps` radius.

---

## **4. Mean-Shift**  
Mean-Shift is a centroid-based algorithm that does not require pre-defining the number of clusters.

```python
param_grid_meanshift = {
    'bandwidth': [0.5, 1.0, 2.0, 3.0],    # Bandwidth for the Gaussian kernel
    'bin_seeding': [True, False],         # Use binning for acceleration
    'cluster_all': [True, False],         # Assign all points to clusters
    'max_iter': [300, 500, 1000],         # Maximum iterations
}
```

### Additional Details
- **`bandwidth`**: Determines the radius of the kernel; can be estimated using `estimate_bandwidth`.

---

## **5. Spectral Clustering**  
Spectral clustering uses eigen-decomposition to detect clusters based on affinity matrices.

```python
param_grid_spectral = {
    'n_clusters': [2, 3, 5, 8, 10],        # Number of clusters
    'eigen_solver': [None, 'arpack', 'lobpcg', 'amg'], # Eigen decomposition solver
    'affinity': ['nearest_neighbors', 'rbf', 'precomputed'], # Affinity matrix construction
    'n_neighbors': [5, 10, 20],            # Neighbors for nearest_neighbors affinity
    'gamma': [0.1, 1.0, 10.0],             # Kernel coefficient for rbf affinity
    'assign_labels': ['kmeans', 'discretize'], # Label assignment strategy
}
```

### Additional Details
- **`affinity='precomputed'`**: Used if the similarity matrix is precomputed.
- **`assign_labels='kmeans'`**: Suitable for larger datasets.

---

## **6. Gaussian Mixture Model (GMM)**  
GMM is a probabilistic approach to clustering, assuming clusters are Gaussian-distributed.

```python
param_grid_gmm = {
    'n_components': [2, 3, 5, 8, 10],        # Number of Gaussian components
    'covariance_type': ['full', 'tied', 'diag', 'spherical'], # Covariance structure
    'tol': [1e-4, 1e-3, 1e-2],              # Convergence tolerance
    'reg_covar': [1e-6, 1e-5, 1e-4],        # Regularization for covariance
    'max_iter': [100, 300, 500],            # Maximum iterations
    'n_init': [1, 5, 10],                   # Number of initializations
    'init_params': ['kmeans', 'random'],    # Initialization strategy
}
```

### Additional Details
- **`covariance_type='diag'`**: Recommended for high-dimensional datasets.

---

## **7. Birch**  
Birch (Balanced Iterative Reducing and Clustering using Hierarchies) clusters data by building a tree structure.

```python
param_grid_birch = {
    'n_clusters': [None, 2, 3, 5, 8, 10], # Final number of clusters
    'threshold': [0.1, 0.5, 1.0],         # Maximum sub-cluster diameter
    'branching_factor': [25, 50, 100],    # Maximum branching factor
    'compute_labels': [True, False],      # Whether to compute labels
}
```

### Additional Details
- **`threshold`**: Smaller values result in more granular clustering.
- **`branching_factor`**: Affects tree structure and computational performance.

---

## **8. Affinity Propagation**  
Affinity Propagation exchanges messages to determine cluster centers.

```python
param_grid_affinity_propagation = {
    'damping': [0.5, 0.7, 0.9],           # Damping factor
    'preference': [-50, -10, 0, 10, 50],  # Preferences for points
    'max_iter': [200, 500, 1000],         # Maximum iterations
    'convergence_iter': [10, 20, 50],     # Iterations for convergence
}
```

---

## **9. OPTICS**  
OPTICS (Ordering Points To Identify Clustering Structure) extends DBSCAN with hierarchical analysis.

```python
param_grid_optics = {
    'min_samples': [5, 10, 20],           # Minimum points for clustering
    'max_eps': [0.5, 1.0, 2.0],           # Maximum distance between points
    'metric': ['euclidean', 'manhattan', 'cosine'], # Distance metric
    'cluster_method': ['xi', 'dbscan'],   # Clustering method
    'xi': [0.05, 0.1, 0.2],              # Minimum steepness on reachability plot
    'p': [1, 2],                         # Minkowski distance parameter
}
```

### Additional Details
- **`xi`**: Controls sensitivity of hierarchical clusters.

---

## **Evaluation and Automation**
- Use metrics such as **Silhouette Score**, **Davies-Bouldin Score**, or **Calinski-Harabasz Index** to evaluate clustering performance.
- Automate hyperparameter testing using `ParameterGrid` and custom loops:

```python
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import silhouette_score

param_grid = param_grid_kmeans  # Example: Choose your parameter grid
best_score = -1
best_params = None

for params in ParameterGrid(param_grid):
    model = KMeans(**params).fit(X)  # Replace with your clustering model
    labels = model.labels_
    score = silhouette_score(X, labels)
    if score > best_score:
        best_score = score
        best_params = params

print("Best Params:", best_params)
print("Best Silhouette Score:", best_score)
```

Feel free to expand or customize for additional clustering models or datasets. 😊


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