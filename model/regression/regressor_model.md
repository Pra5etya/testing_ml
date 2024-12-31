# Regression Models in Scikit-Learn: Pipeline and Hyperparameters by Level

---

## **1. Level Pemula: Model Dasar**

### **Pipeline**
1. **Preprocessing:**
   - Impute missing values (`SimpleImputer`).
   - Scale numerical features (`StandardScaler` or `MinMaxScaler`).

2. **Models:**
   - `LinearRegression`
   - `Ridge`
   - `Lasso`
   - `KNeighborsRegressor`
   - `DecisionTreeRegressor`

### **Example Pipeline Implementation**
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

# Example pipeline for Ridge
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('model', Ridge(alpha=1.0))
])
```

---

## **2. Level Menengah: Regularisasi dan Ensemble**

### **Pipeline**
1. **Preprocessing:**
   - Similar to Level Pemula.
   - Add polynomial features (`PolynomialFeatures`) if needed.

2. **Models:**
   - `ElasticNet`
   - `RandomForestRegressor`
   - `GradientBoostingRegressor`
   - `SVR`

### **Example Pipeline Implementation**
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR

# Example pipeline for RandomForest
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(n_estimators=100))
])
```

---

## **3. Level Lanjutan: Data Kompleks atau Skala Besar**

### **Pipeline**
1. **Preprocessing:**
   - Handle categorical data (`OneHotEncoder` or `OrdinalEncoder`).
   - Scale features as needed.
   - Handle target transformation (`TransformedTargetRegressor`).

2. **Models:**
   - `HistGradientBoostingRegressor`
   - `MLPRegressor`
   - `GaussianProcessRegressor`

### **Example Pipeline Implementation**
```python
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

# Example pipeline for MLPRegressor
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', MLPRegressor(hidden_layer_sizes=(100,), activation='relu'))
])
```

---

## **4. Eksperimen atau Khusus**

### **Pipeline**
1. **Preprocessing:**
   - Transform features or targets (`PowerTransformer`, `QuantileTransformer`).
   - Handle missing values robustly.

2. **Models:**
   - `TweedieRegressor`
   - `QuantileRegressor`
   - `BayesianRidge`

### **Example Pipeline Implementation**
```python
from sklearn.linear_model import TweedieRegressor, BayesianRidge

# Example pipeline for TweedieRegressor
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('model', TweedieRegressor(power=1.5))
])
```

---

### **Tips:**
- Gunakan **`GridSearchCV`** atau **`RandomizedSearchCV`** untuk hyperparameter tuning.
- Sesuaikan preprocessing sesuai dataset Anda.



# Robust Regression

## 1. Huber Regressor

### Apa itu Huber Regressor?
Huber Regressor adalah model regresi yang robust terhadap outlier. Metode ini menggunakan **Huber Loss**, yang menggabungkan dua jenis fungsi loss:
- **Squared loss** untuk error kecil.
- **Absolute loss** untuk error besar.

Fungsi loss ini memungkinkan Huber Regressor untuk memberikan penalti yang lebih kecil pada outlier dibandingkan dengan Least Squares Regression, yang bisa sangat dipengaruhi oleh outlier ekstrem.

### Mengapa Huber Regressor digunakan?
**Kelebihan:**
- Efektif ketika ada outlier dalam dataset yang dapat merusak model regresi standar.
- Memberikan penalti yang lebih stabil pada error besar dibandingkan dengan Least Squares Regression.

**Penggunaan:**
- Dataset dengan beberapa outlier moderat, seperti data keuangan, prediksi cuaca, atau data pengukuran eksperimen.

### Contoh Implementasi
```python
from sklearn.linear_model import HuberRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import numpy as np

# Membuat dataset dengan outlier
X, y = make_regression(n_samples=200, n_features=2, noise=10, random_state=42)
y[::10] += 100  # Menambahkan outlier

# Membagi dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', ColumnTransformer([
        ('scaler', StandardScaler(), [0, 1])
    ])),
    ('regressor', HuberRegressor())
])

# Melatih model
pipeline.fit(X_train, y_train)

# Evaluasi model
print("Score Huber Regressor:", pipeline.score(X_test, y_test))
```

**Penjelasan Pipeline:**
- `ColumnTransformer` digunakan untuk scaling fitur sebelum diteruskan ke model.
- `StandardScaler` menstandarisasi fitur agar memiliki mean = 0 dan variansi = 1, sehingga algoritma bekerja lebih baik.

---

## 2. Theil-Sen Estimator

### Apa itu Theil-Sen Estimator?
Theil-Sen Estimator adalah metode regresi robust yang mengestimasi garis terbaik dengan menggunakan **median** dari kemiringan garis antara semua pasangan titik data. Karena menggunakan median, metode ini sangat tahan terhadap outlier ekstrem.

### Mengapa Theil-Sen Estimator digunakan?
**Kelebihan:**
- Sangat robust terhadap outlier.
- Ideal untuk data dengan noise tinggi atau nilai yang tidak biasa.

**Penggunaan:**
- Analisis data ilmiah atau teknik, seperti pengukuran fisika atau data ekologi.

### Contoh Implementasi
```python
from sklearn.linear_model import TheilSenRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Membuat dataset dengan outlier
X, y = make_regression(n_samples=200, n_features=2, noise=10, random_state=42)
y[::10] += 100  # Menambahkan outlier

# Membagi dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', ColumnTransformer([
        ('scaler', StandardScaler(), [0, 1])
    ])),
    ('regressor', TheilSenRegressor())
])

# Melatih model
pipeline.fit(X_train, y_train)

# Evaluasi model
print("Score Theil-Sen Estimator:", pipeline.score(X_test, y_test))
```

**Penjelasan Pipeline:**
- `ColumnTransformer` dan `StandardScaler` menstandarisasi data sebelum diteruskan ke model.

---

## 3. RANSAC

### Apa itu RANSAC?
RANSAC (RANdom SAmple Consensus) adalah algoritma yang sangat robust terhadap outlier. Algoritma ini memilih subset acak dari data, mengestimasi model, lalu memeriksa seberapa banyak data yang sesuai dengan model tersebut (inliers). Proses ini diulang untuk menemukan model terbaik.

### Mengapa RANSAC digunakan?
**Kelebihan:**
- Sangat efektif untuk dataset dengan banyak outlier ekstrem.
- Dapat mengidentifikasi subset data yang relevan untuk membangun model representatif.

**Penggunaan:**
- Data dengan kesalahan sensor atau noise tinggi, seperti pengolahan citra atau data geospasial.

### Contoh Implementasi
```python
from sklearn.linear_model import RANSACRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Membuat dataset dengan outlier
X, y = make_regression(n_samples=200, n_features=2, noise=10, random_state=42)
y[::10] += 100  # Menambahkan outlier

# Membagi dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', ColumnTransformer([
        ('scaler', StandardScaler(), [0, 1])
    ])),
    ('regressor', RANSACRegressor())
])

# Melatih model
pipeline.fit(X_train, y_train)

# Evaluasi model
print("Score RANSAC Regressor:", pipeline.score(X_test, y_test))
```

**Penjelasan Pipeline:**
- `ColumnTransformer` memastikan data sudah distandarisasi sebelum diteruskan ke model.

---

## Kesimpulan

| **Metode Regresi**      | **Kelebihan**                                                                 | **Kekurangan**                                                | **Jenis Dataset yang Cocok**                                                  |
|-------------------------|-------------------------------------------------------------------------------|----------------------------------------------------------------|--------------------------------------------------------------------------------|
| **Huber Regressor**     | Robust terhadap outlier moderat.                                             | Sensitif terhadap outlier yang sangat ekstrem.                 | Dataset dengan beberapa outlier moderat.                                      |
|                         | Lebih efisien secara komputasi dibandingkan RANSAC.                          | Tidak seefektif Theil-Sen untuk banyak outlier.                | Data keuangan atau eksperimen dengan noise sedang.                            |
| **Theil-Sen Estimator** | Sangat robust terhadap outlier ekstrem.                                      | Kurang efisien untuk dataset besar.                            | Dataset dengan banyak outlier ekstrem, seperti data ilmiah.                   |
|                         | Menggunakan median kemiringan, tidak terpengaruh outlier.                   | Tidak cocok untuk data yang sangat bersih.                     | Data dengan noise tinggi atau kesalahan pengukuran.                           |
| **RANSAC**              | Efektif untuk dataset dengan banyak outlier ekstrem.                        | Kinerja buruk jika proporsi inliers kecil.                     | Dataset dengan banyak outlier dan kesalahan besar, seperti data sensor.       |

**Rekomendasi Penggunaan:**
- **Huber Regressor:** Untuk dataset yang mengandung outlier moderat dan ukuran data cukup besar.
- **Theil-Sen Estimator:** Untuk data dengan noise tinggi atau banyak outlier ekstrem.
- **RANSAC:** Untuk data yang sangat bising dengan outlier ekstrem dan banyak data tidak relevan.



# Kombinasi Model Lengkap pada VotingRegressor dan StackingRegressor

Berikut adalah implementasi kombinasi model lengkap untuk **VotingRegressor** dan **StackingRegressor**. Kombinasi ini mencakup berbagai jenis model regresi, dari yang sederhana hingga kompleks.

---

## **1. Kombinasi Model Lengkap untuk VotingRegressor**
Dalam **VotingRegressor**, kita menggabungkan beberapa model regresi dan menghitung rata-rata prediksi mereka untuk mendapatkan hasil akhir.

### **Kombinasi Model**
- **Linear Models**:
  - `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`
- **Tree-Based Models**:
  - `DecisionTreeRegressor`, `RandomForestRegressor`, `GradientBoostingRegressor`
- **K-Nearest Neighbors**:
  - `KNeighborsRegressor`
- **Support Vector Regressor**:
  - `SVR`
- **Bagging Models**:
  - `BaggingRegressor`

### **Implementasi**
```python
from sklearn.ensemble import VotingRegressor, RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV

# Load dataset
data = fetch_california_housing()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = [
    ('lr', LinearRegression()),
    ('ridge', Ridge()),
    ('lasso', Lasso()),
    ('elastic', ElasticNet()),
    ('tree', DecisionTreeRegressor()),
    ('rf', RandomForestRegressor()),
    ('gb', GradientBoostingRegressor()),
    ('knn', KNeighborsRegressor()),
    ('svr', SVR()),
    ('bagging', BaggingRegressor())
]

# Combine models into VotingRegressor
voting_model = VotingRegressor(estimators=models)

# Define pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('model', voting_model)
])

# Hyperparameter grid for GridSearchCV
param_grid = [
    # Linear Regression
    {
        'pca__n_components': [0.90, 0.95, 0.99],
        'voting__lr': [LinearRegression()],
    },
    # Ridge Regression
    {
        'pca__n_components': [0.90, 0.95, 0.99],
        'voting__ridge': [Ridge()],
        'voting__ridge__alpha': [0.1, 1.0, 100.0],
        'voting__ridge__max_iter': [50000, 100000],
    },
    # Lasso Regression
    {
        'pca__n_components': [0.90, 0.95, 0.99],
        'voting__lasso': [Lasso()],
        'voting__lasso__alpha': [0.1, 1.0, 100.0],
    },
    # ElasticNet
    {
        'pca__n_components': [0.90, 0.95, 0.99],
        'voting__elasticnet': [ElasticNet()],
        'voting__elasticnet__alpha': [0.1, 1.0, 100.0],
        'voting__elasticnet__l1_ratio': [0.2, 0.5, 0.8],
    }
]

# Perform GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Display results
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)
```

---

## **2. Kombinasi Model Lengkap untuk StackingRegressor**
Dalam **StackingRegressor**, kita menggabungkan prediksi dari beberapa model dasar menggunakan model meta untuk mendapatkan hasil akhir.

### **Kombinasi Model**
- **Base Models (Model Dasar)**:
  - **Linear Models**: `LinearRegression`, `Ridge`, `ElasticNet`
  - **Tree-Based Models**: `DecisionTreeRegressor`, `RandomForestRegressor`, `GradientBoostingRegressor`
  - **K-Nearest Neighbors**: `KNeighborsRegressor`
  - **Support Vector Regressor**: `SVR`
  - **Bagging Models**: `BaggingRegressor`
- **Meta Model (Model Meta)**:
  - **Linear Models**: `LinearRegression`, `Ridge`
  - **Ensemble Models**: `RandomForestRegressor`, `GradientBoostingRegressor`

### **Implementasi**
```python
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV

# Load dataset
data = fetch_california_housing()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base models
base_models = [
    ('ridge', Ridge()),
    ('elastic', ElasticNet()),
    ('tree', DecisionTreeRegressor()),
    ('rf', RandomForestRegressor()),
    ('gb', GradientBoostingRegressor()),
    ('knn', KNeighborsRegressor()),
    ('svr', SVR())
]

# Define meta model
meta_model = LinearRegression()

# Combine base models and meta model into StackingRegressor
stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)

# Define pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('model', stacking_model)
])

# Hyperparameter grid for GridSearchCV
param_grid = [
    # Linear Regression
    {
        'pca__n_components': [0.90, 0.95, 0.99],
        'voting__lr': [LinearRegression()],
    },
    # Ridge Regression
    {
        'pca__n_components': [0.90, 0.95, 0.99],
        'voting__ridge': [Ridge()],
        'voting__ridge__alpha': [0.1, 1.0, 100.0],
        'voting__ridge__max_iter': [50000, 100000],
    },
    # Lasso Regression
    {
        'pca__n_components': [0.90, 0.95, 0.99],
        'voting__lasso': [Lasso()],
        'voting__lasso__alpha': [0.1, 1.0, 100.0],
    },
    # ElasticNet
    {
        'pca__n_components': [0.90, 0.95, 0.99],
        'voting__elasticnet': [ElasticNet()],
        'voting__elasticnet__alpha': [0.1, 1.0, 100.0],
        'voting__elasticnet__l1_ratio': [0.2, 0.5, 0.8],
    }
]

# Perform GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Display results
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)
```

---

## **Perbedaan Utama dalam Kombinasi Model**

| **Aspek**              | **VotingRegressor**                                  | **StackingRegressor**                                   |
|------------------------|----------------------------------------------------|--------------------------------------------------------|
| **Jumlah Model Dasar**  | Semua model digunakan secara independen            | Semua model digunakan sebagai input untuk meta model   |
| **Model Meta**          | Tidak ada, menggunakan rata-rata prediksi          | Ada, digunakan untuk mempelajari kombinasi optimal prediksi dasar |
| **Fleksibilitas**       | Mudah digunakan, cocok untuk eksperimen sederhana  | Lebih fleksibel untuk kombinasi model yang kompleks    |
| **Performansi**         | Stabil dengan model sederhana                      | Potensi lebih tinggi, terutama dengan model meta yang baik |
| **Hyperparameter**      | Setiap model dasar dioptimalkan secara independen  | Model dasar dan meta dioptimalkan bersama              |

---

## **Kapan Menggunakan Kombinasi Ini?**

1. **VotingRegressor**:
   - Ketika Anda ingin pendekatan sederhana untuk menggabungkan beberapa model tanpa kompleksitas tambahan.
   - Cocok jika model-model dasar memiliki performa yang cukup stabil.

2. **StackingRegressor**:
   - Ketika Anda memiliki model dasar yang beragam dan ingin memanfaatkan kekuatan model meta untuk menggabungkan prediksi.
   - Cocok untuk data yang kompleks di mana kombinasi model dapat memberikan keuntungan signifikan.
