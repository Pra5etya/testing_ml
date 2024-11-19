# Ringkasan Modul Scikit-Learn

Berikut adalah ringkasan lengkap dari berbagai modul dan submodul yang tersedia di `scikit-learn`. Ini mencakup komponen utama dan fungsionalitas yang sering digunakan dalam proyek pembelajaran mesin:

## 1. Dataset (`sklearn.datasets`)
- **Dataset Bawaan**: Iris, Boston Housing, Digit, Wine, dll.
- **Fungsi Pemuatan**: `load_iris()`, `load_boston()`, `load_digits()`, dll.
- **Pembuatan Dataset Tiruan**: `make_classification()`, `make_regression()`, `make_blobs()`.

## 2. Prapemrosesan Data (`sklearn.preprocessing`)
- **Standarisasi & Normalisasi**: `StandardScaler`, `MinMaxScaler`, `Normalizer`.
- **Encoding Kategori**: `LabelEncoder`, `OneHotEncoder`, `OrdinalEncoder`.
- **Transformasi Fitur**: `PolynomialFeatures`, `Binarizer`, `PowerTransformer`.

## 3. Seleksi & Validasi Model (`sklearn.model_selection`)
- **Pemecahan Data**: `train_test_split`, `KFold`, `StratifiedKFold`.
- **Hyperparameter Tuning**: `GridSearchCV`, `RandomizedSearchCV`.
- **Validasi Silang**: `cross_val_score`, `cross_validate`, `LeaveOneOut`.

## 4. Pipeline & Alur Kerja (`sklearn.pipeline`)
- **Pipeline**: Menyusun prapemrosesan dan model ke dalam satu alur kerja yang lebih mudah digunakan.
- **FeatureUnion**: Menggabungkan beberapa transformator untuk diterapkan pada data.

## 5. Ekstraksi Fitur (`sklearn.feature_extraction`)
- **Teks**: `CountVectorizer`, `TfidfVectorizer`, `HashingVectorizer`.
- **Data Kategorikal**: `DictVectorizer`.
- **Gambar**: `extract_patches_2d`.

## 6. Seleksi Fitur (`sklearn.feature_selection`)
- **Pemilihan Berbasis Statistik**: `SelectKBest`, `chi2`, `f_classif`.
- **Pemilihan Berdasarkan Model**: `SelectFromModel`, `RFE`, `RFECV`.

## 7. Algoritma Pembelajaran Mesin
- **Klasifikasi (`sklearn.linear_model`, `sklearn.ensemble`, `sklearn.svm`, dll.)**
  - **Regresi Logistik**: `LogisticRegression`.
  - **Support Vector Machine**: `SVC`.
  - **K-Nearest Neighbors**: `KNeighborsClassifier`.
  - **Decision Trees**: `DecisionTreeClassifier`.
  - **Ensemble**: `RandomForestClassifier`, `GradientBoostingClassifier`, `AdaBoostClassifier`.
- **Regresi (`sklearn.linear_model`, `sklearn.ensemble`, dll.)**
  - **Regresi Linear**: `LinearRegression`, `Ridge`, `Lasso`.
  - **Support Vector Regressor**: `SVR`.
  - **Decision Trees**: `DecisionTreeRegressor`.
  - **Ensemble**: `RandomForestRegressor`, `GradientBoostingRegressor`.
- **Clustering (`sklearn.cluster`)**
  - **K-Means**: `KMeans`.
  - **Hierarki**: `AgglomerativeClustering`.
  - **Density-Based**: `DBSCAN`.
  - **Clustering Spektral**: `SpectralClustering`.
- **Reduksi Dimensi (`sklearn.decomposition`, `sklearn.manifold`)**
  - **PCA (Principal Component Analysis)**: `PCA`.
  - **LDA (Linear Discriminant Analysis)**: `LinearDiscriminantAnalysis`.
  - **t-SNE (t-distributed Stochastic Neighbor Embedding)**: `TSNE`.
  - **Isomap**: `Isomap`.

## 8. Penskalaan & Pengelolaan Data (`sklearn.compose`, `sklearn.impute`)
- **ColumnTransformer**: Memungkinkan prapemrosesan berbeda pada kolom yang berbeda.
- **SimpleImputer**: Mengisi nilai yang hilang dengan rata-rata, median, atau strategi lain.
- **KNNImputer**: Mengisi nilai yang hilang menggunakan algoritma KNN.

## 9. Evaluasi & Metrik (`sklearn.metrics`)
- **Metrik Klasifikasi**: `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`, `roc_auc_score`.
- **Metrik Regresi**: `mean_squared_error`, `mean_absolute_error`, `r2_score`.
- **Metrik Clustering**: `silhouette_score`, `adjusted_rand_score`.

## 10. Kalibrasi Model (`sklearn.calibration`)
- **CalibratedClassifierCV**: Digunakan untuk mengkalibrasi probabilitas model, seperti model yang memberikan output probabilitas yang dapat diinterpretasikan.

## 11. Metode Ansambel (`sklearn.ensemble`)
- **Voting Classifier**: Menggabungkan beberapa model untuk memprediksi kelas mayoritas.
- **Bagging & Boosting**: `BaggingClassifier`, `RandomForest`, `AdaBoost`, `GradientBoosting`, `HistGradientBoosting`.

## 12. Statistik & Perubahan Data (`sklearn.kernel_approximation`, `sklearn.neighbors`)
- **Approximation**: `Nystroem`, `RBFSampler` untuk mempercepat algoritma kernel.
- **Analisis Tetangga Terdekat**: `KNeighborsClassifier`, `KNeighborsRegressor`.

## 13. Preprocessing Lanjutan (`sklearn.preprocessing`)
- **Penskalaan Ciri**: `RobustScaler` untuk data dengan outlier.
- **Diskritisasi**: `KBinsDiscretizer` untuk mengubah data numerik menjadi data kategorikal.

## 14. Dukungan untuk Data Imbalanced (`imblearn`, bagian dari `scikit-learn-contrib`)
- **Resampling**: `SMOTE`, `RandomUnderSampler`, `RandomOverSampler`.

## 15. Transformasi untuk Pemetaan Non-Linear (`sklearn.kernel_approximation`)
- **Kernel Approximations**: `Nystroem`, `RBFSampler`, `AdditiveChi2Sampler`.

## 16. Pemodelan Statistika (`sklearn.covariance`, `sklearn.gaussian_process`)
- **Covariance Estimation**: `EllipticEnvelope`, `EmpiricalCovariance`.
- **Gaussian Processes**: `GaussianProcessClassifier`, `GaussianProcessRegressor`.

## 17. Pemodelan Data Waktu (`sklearn.metrics`, `sklearn.preprocessing`)
- **Forecasting Tools**: Fitur tambahan seperti `TimeSeriesSplit` untuk validasi data yang diurutkan waktu.