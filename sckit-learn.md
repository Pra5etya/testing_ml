1. sklearn.datasets: 
    * Menyediakan dataset yang siap pakai seperti iris, wine, digits, serta alat untuk memuat dataset eksternal (CSV, JSON, dll.).
    * Contoh: load_iris(), fetch_openml().

2. sklearn.model_selection
    * Modul ini berfungsi untuk pembagian data, validasi model, dan pemilihan model terbaik melalui cross-validation.
    * Komponen penting:
        * train_test_split: Memisahkan dataset menjadi set pelatihan dan pengujian.
        * cross_val_score dan cross_val_predict: Evaluasi model dengan cross-validation.
        * GridSearchCV dan RandomizedSearchCV: Tuning hyperparameter untuk menemukan model terbaik.

3. sklearn.preprocessing
    * Menyediakan berbagai alat untuk praproses data sebelum diterapkan ke model.
    * Komponen penting:
        * StandardScaler, MinMaxScaler: Normalisasi fitur.
        * OneHotEncoder, LabelEncoder: Encoding variabel kategorikal.
        * PolynomialFeatures: Membuat fitur polinomial dari fitur yang ada.
        * FunctionTransformer: Menggunakan fungsi kustom untuk transformasi data.

4. sklearn.impute
    * Modul untuk menangani data yang hilang (missing values).
    * Komponen penting:
        * SimpleImputer: Mengisi nilai yang hilang dengan rata-rata, median, atau nilai khusus.
        * KNNImputer: Menggunakan K-Nearest Neighbors untuk mengisi nilai yang hilang.

5. sklearn.pipeline
    * Memungkinkan Anda menggabungkan beberapa langkah praproses dan model dalam satu pipeline terstruktur.
    * Komponen penting:
        * Pipeline: Menggabungkan serangkaian transformasi dan model menjadi satu alur.
        * FeatureUnion: Menggabungkan beberapa transformasi secara paralel.
        * Caching pipeline untuk meningkatkan efisiensi komputasi.

6. sklearn.compose
    * Digunakan untuk menggabungkan berbagai transformasi pada fitur yang berbeda (misalnya numerik dan kategorikal).
    * Komponen penting:
        * ColumnTransformer: Menerapkan transformasi yang berbeda pada kolom-kolom yang berbeda.
        * TransformedTargetRegressor: Mengaplikasikan transformasi pada target (label).

7. sklearn.decomposition
    * Alat untuk pengurangan dimensi (dimensionality reduction).
    * Komponen penting:
        * PCA: Principal Component Analysis untuk pengurangan dimensi.
        * TruncatedSVD: Pengurangan dimensi tanpa perlu data dipusatkan (biasa untuk data sparse).

8. sklearn.ensemble
    * Modul yang berisi metode ensemble, yang menggabungkan beberapa model untuk meningkatkan performa.
    * Komponen penting:
        * RandomForestClassifier dan RandomForestRegressor: Algoritma ensemble berbasis pohon keputusan.
        * GradientBoostingClassifier dan GradientBoostingRegressor: Model boosting yang bertahap.
        * VotingClassifier dan VotingRegressor: Menggabungkan prediksi beberapa model.

9. sklearn.linear_model
    * Berisi berbagai algoritma regresi dan klasifikasi berbasis model linear.
    * Komponen penting:
        * LinearRegression: Regresi linear sederhana.
        * LogisticRegression: Klasifikasi untuk dua atau lebih kelas.
        * Ridge, Lasso: Model regresi dengan regularisasi.

10. sklearn.metrics
    * Modul untuk evaluasi model machine learning.
    * Komponen penting:
        * accuracy_score, precision_score, recall_score: Metrik evaluasi untuk klasifikasi.
        * mean_squared_error, r2_score: Metrik evaluasi untuk regresi.
        * roc_auc_score, confusion_matrix: Metrik untuk klasifikasi biner.

11. sklearn.svm
    * Implementasi Support Vector Machines (SVM) untuk klasifikasi dan regresi.
    * Komponen penting:
        * SVC: Support Vector Classification.
        * SVR: Support Vector Regression.

12. sklearn.neighbors
    * Modul yang menyediakan algoritma K-Nearest Neighbors (KNN).
    * Komponen penting:
        * KNeighborsClassifier: KNN untuk klasifikasi.
        * KNeighborsRegressor: KNN untuk regresi.

13. sklearn.tree
    * Berisi algoritma Decision Tree untuk klasifikasi dan regresi.
    * Komponen penting:
        * DecisionTreeClassifier: Pohon keputusan untuk klasifikasi.
        * DecisionTreeRegressor: Pohon keputusan untuk regresi.

14. sklearn.cluster
    * Alat untuk pengelompokan data (clustering).
    * Komponen penting:
        * KMeans: Algoritma clustering berbasis centroid.
        * DBSCAN: Algoritma clustering berbasis kepadatan.

15. sklearn.feature_selection
    * Modul untuk memilih fitur yang paling relevan.
    * Komponen penting:
        * SelectKBest: Memilih K fitur terbaik berdasarkan skor statistik.
        * RFE (Recursive Feature Elimination): Secara rekursif menghilangkan fitur paling tidak penting.

16. sklearn.utils.validation
    * Alat untuk validasi input data dan parameter.
    * Komponen penting:
        * check_X_y dan check_array: Memvalidasi input X dan y untuk memastikan bentuk data sesuai.


* Rangkuman Singkat Modul dan Fungsinya:
* Datasets: Memuat data bawaan atau eksternal.
* Preprocessing: Mengolah data (scaling, encoding).
* Imputation: Menangani nilai yang hilang.
* Model Selection: Pembagian data, cross-validation, tuning hyperparameter.
* Pipeline & Compose: Membuat alur praproses dan model yang terstruktur.
* Decomposition: Pengurangan dimensi (PCA).
* Ensemble & Trees: Model berbasis ensemble dan pohon keputusan.
* Linear Model: Model linear untuk regresi/klasifikasi.
* Metrics: Evaluasi model.
* SVM & Neighbors: Algoritma berbasis SVM dan KNN.
* Clustering: Pengelompokan data.
* Feature Selection: Memilih fitur penting.
* Validation: Validasi input dan parameter.