# Dimensionality Reduction with Pipelines

Berikut adalah implementasi kode **dimensionality reduction** menggunakan **Pipeline** dari scikit-learn, berdasarkan kategori level:  

---

## **1. Linear Dimensionality Reduction (Level Dasar)**

Menggunakan **PCA** sebagai contohnya:  
```python
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Load dataset
X, y = load_iris(return_X_y=True)

# Pipeline for Linear Dimensionality Reduction
pipeline_linear = Pipeline([
    ('scaler', StandardScaler()),  # Normalize features
    ('pca', PCA(n_components=2)),  # Reduce to 2 dimensions
    ('classifier', RandomForestClassifier(random_state=42))
])

# Fit the pipeline
pipeline_linear.fit(X, y)

# Predict
predictions = pipeline_linear.predict(X)
print("Linear Dimensionality Reduction Pipeline Completed.")
```

---

## **2. Nonlinear Dimensionality Reduction (Level Menengah)**

Menggunakan **t-SNE** untuk visualisasi dan klasifikasi:  
```python
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits

# Load dataset
X, y = load_digits(return_X_y=True)

# t-SNE is usually used for visualization (dimensionality <= 3)
pipeline_nonlinear = Pipeline([
    ('scaler', StandardScaler()),  # Normalize features
    ('tsne', TSNE(n_components=2, random_state=42, perplexity=30)),  # Reduce to 2D
])

# Transform data with t-SNE
X_transformed = pipeline_nonlinear.fit_transform(X)

# Use transformed data for classification
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_transformed, y)
print("Nonlinear Dimensionality Reduction Pipeline Completed.")
```

---

## **3. Deep Learning-Based Dimensionality Reduction (Level Lanjut)**

Menggunakan **Autoencoder** dari Keras:  
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import numpy as np

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(500, 50)  # High-dimensional data
y = np.random.randint(0, 2, 500)

# Autoencoder model
input_dim = X.shape[1]
encoding_dim = 10  # Reduced dimensions

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
encoder = Model(inputs=input_layer, outputs=encoded)

autoencoder.compile(optimizer=Adam(learning_rate=0.01), loss='mse')

# Train the autoencoder
autoencoder.fit(X, X, epochs=20, batch_size=32, shuffle=True, verbose=0)

# Encode the data
X_encoded = encoder.predict(X)

# Use encoded data for classification
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_encoded, y)
print("Deep Learning-Based Dimensionality Reduction Pipeline Completed.")
```

---

## Penjelasan dan Catatan

### **Level Dasar (Linear):**  
- Memanfaatkan teknik linier seperti **PCA**.  
- Cepat dan mudah diterapkan.  

### **Level Menengah (Nonlinear):**  
- **t-SNE** dan **UMAP** lebih cocok untuk visualisasi.  
- Perlu perhatian lebih pada parameter seperti **perplexity** dan **n_neighbors**.  

### **Level Lanjut (Deep Learning):**  
- Membutuhkan lebih banyak data dan waktu untuk melatih model.  
- Menggunakan **Autoencoder** memungkinkan fleksibilitas untuk data yang kompleks.  

Jika Anda ingin mencoba teknik lain seperti **UMAP** atau menambahkan evaluasi, beri tahu saya! 😊


