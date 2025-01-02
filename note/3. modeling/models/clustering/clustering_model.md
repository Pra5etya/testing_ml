 # **Pipeline clustering Berdasarkan Level**

 ---

 ## **1. Level Pemula: Model Dasar**

### **Pipeline**  
1. **Preprocessing**:  
   - Menghilangkan noise (jika ada).  
   - Standardisasi atau normalisasi fitur (opsional untuk beberapa metode).  

2. **Models**:  
   - **K-Means**  
   - **Hierarchical Clustering (Agglomerative)**  
   - **DBSCAN**  

### **Example Pipeline Implementation**  
```python
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Agglomerative Clustering
agg_clustering = AgglomerativeClustering(n_clusters=3)
agg_clustering.fit(X_scaled)

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(X_scaled)
```

---

## **2. Level Menengah: Model Lanjutan**

### **Pipeline**  
1. **Preprocessing**:  
   - Mengatasi missing values (imputasi jika perlu).  
   - Normalisasi atau transformasi data untuk menjaga struktur cluster.  

2. **Models**:  
   - **Gaussian Mixture Models (GMM)**  
   - **Spectral Clustering**  
   - **OPTICS**  

### **Example Pipeline Implementation**  
```python
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering, OPTICS

# Gaussian Mixture Model
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X_scaled)

# Spectral Clustering
spectral = SpectralClustering(n_clusters=3, affinity='nearest_neighbors', random_state=42)
spectral.fit(X_scaled)

# OPTICS
optics = OPTICS(min_samples=5, xi=0.05, min_cluster_size=0.1)
optics.fit(X_scaled)
```

---

## **3. Level Mahir: Model Kompleks**

### **Pipeline**  
1. **Preprocessing**:  
   - Feature extraction (misalnya PCA untuk dimensi tinggi).  
   - Transformasi data menggunakan deep learning (autoencoder).  

2. **Models**:  
   - **HDBSCAN**  
   - **Deep Embedded Clustering (DEC)**  
   - **Autoencoder + K-Means**  

### **Example Pipeline Implementation**  
```python
from hdbscan import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# HDBSCAN
hdbscan = HDBSCAN(min_cluster_size=10, cluster_selection_epsilon=0.5)
hdbscan.fit(X_scaled)

# PCA + K-Means
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_pca)
```

# Note
For Deep Embedded Clustering (DEC), implement using TensorFlow or PyTorch. 
This typically involves training an autoencoder and clustering the latent space.





# Hyperparameter Tuning in Unsupervised Learning

Hyperparameter tuning for unsupervised learning, such as clustering or dimensionality reduction, can be challenging since there are no labels for direct evaluation. The performance of the model is often evaluated using metrics such as **silhouette score**, **distortion (inertia)**, or **DB index**, depending on the model type. This guide demonstrates how to apply hyperparameter tuning using `ParameterGrid` and `ParameterSampler` for unsupervised learning.

---

## 1. Using `ParameterGrid` for Clustering (K-Means & DBSCAN)

### Example: Hyperparameter Tuning for K-Means and DBSCAN

```python
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.model_selection import ParameterGrid
from sklearn.datasets import make_blobs
import numpy as np

# Example dataset
X, _ = make_blobs(n_samples=500, centers=4, cluster_std=1.0, random_state=42)

# Hyperparameter grid
param_grid = [
    # K-Means
    {
        'model': [KMeans()],
        'model__n_clusters': [2, 3, 4, 5, 6],
        'model__init': ['k-means++', 'random'],
        'model__max_iter': [300, 500, 1000],
        'model__n_init': [10, 20],
        'model__random_state': [42]
    },
    # DBSCAN
    {
        'model': [DBSCAN()],
        'model__eps': [0.3, 0.5, 0.7],
        'model__min_samples': [3, 5, 10],
        'model__metric': ['euclidean', 'manhattan']
    }
]

# Hyperparameter evaluation
best_score = -1
best_params = None

# Iterate over all combinations
for params in ParameterGrid(param_grid):
    model = params.pop('model')  # Extract model
    model.set_params(**params)  # Set model parameters
    labels = model.fit_predict(X)  # Fit and predict
    if len(np.unique(labels)) > 1:  # Avoid single cluster
        score = silhouette_score(X, labels)
        if score > best_score:
            best_score = score
            best_params = {**params, 'model': model.__class__.__name__}

print("Best Score:", best_score)
print("Best Params:", best_params)
```

---

## 2. Using `ParameterSampler` for Random Sampling

When the hyperparameter space is large, `ParameterSampler` can be used to sample a random subset of combinations. This is particularly useful for algorithms like DBSCAN or Gaussian Mixture Models.

### Example: Hyperparameter Tuning with DBSCAN

```python
from sklearn.model_selection import ParameterSampler
from scipy.stats import uniform, randint

# Hyperparameter distribution
param_dist = {
    'eps': uniform(0.1, 1.0),            # Uniform distribution for eps
    'min_samples': randint(3, 15),       # Integer sampling for min_samples
    'metric': ['euclidean', 'manhattan'] # Discrete choices
}

# Generate 10 random combinations
random_params = list(ParameterSampler(param_dist, n_iter=10, random_state=42))

# Evaluate with silhouette score
best_score = -1
best_params = None

for params in random_params:
    model = DBSCAN(**params)
    labels = model.fit_predict(X)
    if len(np.unique(labels)) > 1:  # Avoid single cluster
        score = silhouette_score(X, labels)
        if score > best_score:
            best_score = score
            best_params = {**params}

print("Best Score:", best_score)
print("Best Params:", best_params)
```

---

## 3. Hyperparameter Tuning for Dimensionality Reduction

When using methods like PCA, hyperparameter tuning focuses on selecting the number of principal components (`n_components`).

### Example: Tuning PCA + K-Means

```python
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# PCA + K-Means pipeline
pipe = Pipeline([
    ('pca', PCA()),
    ('kmeans', KMeans(random_state=42))
])

# Hyperparameter grid
param_grid = {
    'pca__n_components': [2, 3, 4],
    'kmeans__n_clusters': [2, 3, 4, 5]
}

# Evaluate hyperparameter combinations
best_score = -1
best_params = None

for params in ParameterGrid(param_grid):
    pipe.set_params(**params)
    labels = pipe.fit_predict(X)
    if len(np.unique(labels)) > 1:  # Avoid single cluster
        score = silhouette_score(X, labels)
        if score > best_score:
            best_score = score
            best_params = params

print("Best Score:", best_score)
print("Best Params:", best_params)
```

---

## 4. When to Use `ParameterGrid` vs `ParameterSampler`

| **Condition**                      | **Use ParameterGrid**            | **Use ParameterSampler**          |
|------------------------------------|----------------------------------|-----------------------------------|
| Small parameter space              | ✅                                | ❌                                 |
| Large parameter space              | ❌                                | ✅                                 |
| Need to test all combinations      | ✅                                | ❌                                 |
| Limited time and resources         | ❌                                | ✅                                 |
| Sampling from probability distribution | ❌                            | ✅                                 |

---

## Tips for Evaluating Models in Unsupervised Learning

1. **Use Appropriate Metrics**:
   - **Silhouette Score**: Commonly used for clustering.
   - **Inertia (distortion)**: Specific to K-Means.
   - **DB Index**: Measures cluster quality based on distances between and within clusters.

2. **Avoid Single Clusters**:
   - Ensure the model does not produce only one cluster.

3. **Distribution Sampling**:
   - Use `loguniform` for parameters with large ranges (e.g., `eps` in DBSCAN).
   - Use `uniform` for parameters with smaller ranges.

4. **Combine Steps in a Pipeline**:
   - Use pipelines for preprocessing and clustering or dimensionality reduction.

By following these steps, you can efficiently perform hyperparameter tuning for unsupervised learning models.
