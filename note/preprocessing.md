# Scaler (Penskalaan)
| Jenis Scaler| Model yang Biasanya Menggunakannya |
|-------------|-------------|
| StandardScaler | Regresi Linear/Logistik, SVM, KNN, PCA, Neural Networks |
| MinMaxScaler | Neural Networks, K-Means Clustering, KNN, pohon keputusan (dalam kasus tertentu) |
| RobustScaler | Regresi Linear/Logistik, SVM, KNN (ketika ada outlier signifikan) |
| MaxAbsScaler | Neural Networks, model berbasis data sparse (klasifikasi teks, regresi logistik untuk data sparse) |


# Encoder (Pengkodean Kategorikal):
    * LabelEncoder: Mengubah label kategorikal menjadi angka integer (untuk variabel target).
    * OneHotEncoder: Mengubah variabel kategorikal menjadi format biner (one-hot encoded).
    * OrdinalEncoder: Mengubah fitur kategorikal menjadi angka integer, di mana urutan memiliki makna.

# Imputer (Pengganti Nilai Hilang):
    * SimpleImputer: Mengganti nilai hilang dengan strategi yang ditentukan (mean, median, mode, atau konstanta).
    * KNNImputer: Mengganti nilai hilang dengan menggunakan nilai terdekat berdasarkan algoritma K-Nearest Neighbors.

# Polynomial Features:
    * PolynomialFeatures: Menghasilkan fitur polinomial dari fitur yang ada, berguna untuk regresi polinomial.

# Binarizer:
    * Binarizer: Mengubah nilai menjadi 0 atau 1 berdasarkan threshold yang ditentukan.

# Function Transformer:
    * FunctionTransformer: Memungkinkan Anda untuk menggunakan fungsi kustom untuk transformasi data.

# Normalizer:
    * Normalizer: Mengubah fitur ke norma tertentu, misalnya L1 atau L2 norm, untuk mengubah setiap baris data menjadi unit vector.