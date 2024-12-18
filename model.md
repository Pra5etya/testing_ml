# Panduan Lengkap Pemilihan dan Penggunaan Model dalam Machine Learning

## Machine Learning

### 1. Supervised Learning
Supervised learning membutuhkan data yang memiliki label untuk merepresentasikan target (baik berupa nilai numerik maupun kategori).

#### A. Menentukan Label Data
1. **Korelasi Antar Kolom**
   - Korelasi digunakan untuk menentukan hubungan antara fitur dan target. Gunakan `.corr()` untuk data numerik. 
   - Metode pada `.corr`:
     - **Pearson**: Untuk data numerik.
     - **Spearman**: Untuk data monotonic (dengan fluktuasi).
     - **Kendall**: Untuk data ordinal atau campuran.
   - Penting dilakukan visualisasi data untuk memilih metode yang paling sesuai.

2. **Uji Perbedaan Antar Kolom**
   - Gunakan ANOVA untuk memeriksa apakah terdapat perbedaan signifikan antar grup.
   - Parameter ANOVA:
     - **F-statistic**:
       - > 1: Perbedaan antar grup signifikan.
       - <= 1: Tidak ada perbedaan signifikan.
     - **P-value**:
       - < 0.05: Perbedaan antar grup signifikan.
       - >= 0.05: Tidak ada perbedaan signifikan.

#### B. Pemilihan Model

##### 1. Regression: Model untuk memprediksi nilai kontinu
- **Linear Models**: Cocok untuk hubungan linear antara fitur dan target.
  - Contoh model: `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`.
  - Gunakan Ridge atau Lasso untuk mengatasi multikolinearitas atau jika ada banyak fitur tidak relevan.
  - **Contoh Aplikasi**: Memprediksi harga rumah berdasarkan luas, lokasi, dan fasilitas.

- **Tree-Based Models**: Cocok untuk data non-linear dan kompleks.
  - Contoh model: `DecisionTreeRegressor`, `RandomForestRegressor`, `GradientBoostingRegressor`, `XGBRegressor` (via xgboost).
  - Gradient Boosting biasanya memberikan akurasi lebih tinggi tetapi lebih lambat dibanding Random Forest.
  - **Contoh Aplikasi**: Memprediksi konsumsi energi berdasarkan cuaca dan waktu.

- **Support Vector Machine (SVM)**: Cocok untuk dataset kecil hingga sedang dengan noise tinggi.
  - Contoh model: `SVR` (Support Vector Regression).
  - **Contoh Aplikasi**: Memprediksi harga saham berdasarkan indikator keuangan.

- **Lainnya**:
  - `KNeighborsRegressor`: Efektif untuk dataset kecil dengan pola sederhana.
  - `GaussianProcessRegressor`: Cocok untuk data kecil dengan pola kompleks.
  - **Contoh Aplikasi**: Memprediksi perubahan suhu dalam eksperimen kecil.

##### Contoh Implementasi dalam Pipeline
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', Ridge(alpha=1.0))
])

# Train model
pipeline.fit(X_train, y_train)

# Evaluate
predictions = pipeline.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)
```

##### 2. Classification: Model untuk memprediksi kategori atau kelas
- **Linear Models**: Cocok untuk klasifikasi sederhana.
  - Contoh model: `LogisticRegression`, `RidgeClassifier`, `Perceptron`.
  - Perceptron hanya cocok untuk klasifikasi dengan hyperplane linier.
  - **Contoh Aplikasi**: Menentukan apakah pelanggan akan membeli produk (ya/tidak).

- **Tree-Based Models**: Cocok untuk dataset non-linear.
  - Contoh model: `DecisionTreeClassifier`, `RandomForestClassifier`, `GradientBoostingClassifier`, `XGBClassifier`.
  - **Contoh Aplikasi**: Memprediksi apakah seseorang akan terkena penyakit berdasarkan riwayat kesehatan.

- **Support Vector Machine (SVM)**: Cocok untuk dataset dengan dimensi tinggi dan sedikit data.
  - Contoh model: `SVC` (Support Vector Classification).
  - **Contoh Aplikasi**: Klasifikasi gambar (misalnya, kucing vs. anjing).

- **Lainnya**:
  - `KNeighborsClassifier`: Cocok untuk dataset kecil tanpa banyak fitur.
  - `GaussianNB` dan `MultinomialNB`: Efektif untuk data probabilistik.
  - **Contoh Aplikasi**: Klasifikasi pelanggan berdasarkan lokasi geografis.

##### Contoh Implementasi dalam Pipeline
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train model
pipeline.fit(X_train, y_train)

# Evaluate
predictions = pipeline.predict(X_test)
print(classification_report(y_test, predictions))
```

### 2. Unsupervised Learning
Unsupervised learning tidak membutuhkan label data. Model akan secara otomatis mengidentifikasi pola dari data yang ada.

#### A. Pemilihan Model

##### 1. Clustering: Mengelompokkan data berdasarkan kesamaan
- **KMeans**: Cocok jika jumlah klaster diketahui sebelumnya.
  - **Contoh**: Segmentasi pelanggan berdasarkan pola pembelian.
- **DBSCAN**: Cocok untuk data dengan distribusi non-sferis atau adanya noise.
  - **Contoh**: Mendeteksi pola perjalanan abnormal dari GPS.
- **AgglomerativeClustering**: Cocok untuk analisis hierarki.
  - **Contoh**: Mengelompokkan dokumen berdasarkan topik.
- **MeanShift**: Mencari klaster secara adaptif.
  - **Contoh**: Menganalisis pola rute perjalanan kendaraan.
- **Birch**: Efektif untuk dataset besar.
  - **Contoh**: Segmentasi pelanggan dari database e-commerce.
- **SpectralClustering**: Cocok untuk struktur data non-linear.
  - **Contoh**: Mengelompokkan gambar berdasarkan warna dominan.

##### Contoh Implementasi dalam Pipeline
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clustering', KMeans(n_clusters=3, random_state=42))
])

# Fit pipeline
pipeline.fit(X)

# Predict cluster labels
cluster_labels = pipeline.named_steps['clustering'].labels_
print("Cluster Labels:", cluster_labels)
```

### 3. Semi-Supervised Learning
Menggabungkan data berlabel dan tidak berlabel dalam pelatihan.

#### A. Pemilihan Model
- **LabelPropagation**: Cocok jika data tanpa label memiliki pola konsisten.
  - **Contoh**: Melabeli gambar medis berdasarkan diagnosis sebagian data.
- **LabelSpreading**: Lebih fleksibel dalam menangani noise dibanding LabelPropagation.
  - **Contoh**: Melabeli data sensor lingkungan di lokasi sulit dijangkau.

##### Contoh Implementasi dalam Pipeline
```python
from sklearn.semi_supervised import LabelPropagation

# Model
model = LabelPropagation()

# Fit
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
print("Predictions:", predictions)
```

### 4. Reinforcement Learning
Tidak didukung oleh Scikit-learn. Gunakan library seperti Stable-Baselines atau OpenAI Gym.

### 5. Lainnya

#### A. Anomaly Detection: Mendeteksi outlier atau anomali
- **IsolationForest**: Efektif untuk dataset dengan banyak fitur dan sedikit anomali.
  - **Contoh**: Mendeteksi transaksi keuangan mencurigakan.
- **OneClassSVM**: Cocok untuk data distribusi kompleks.
  - **Contoh**: Mendeteksi kegagalan mesin.
- **EllipticEnvelope**: Efektif untuk data dengan distribusi Gaussian.
  - **Contoh**: Mendeteksi transaksi tidak biasa.
- **LocalOutlierFactor**: Efektif untuk klaster lokal kecil.
  - **Contoh**: Mendeteksi perilaku mencurigakan di log server.

##### Contoh Implementasi dalam Pipeline
```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('anomaly_detector', IsolationForest(random_state=42))
])

# Fit pipeline
pipeline.fit(X)

# Predict anomalies (-1: anomaly, 1: normal)
anomaly_predictions = pipeline.predict(X)
print("Anomaly Predictions:", anomaly_predictions)
```

