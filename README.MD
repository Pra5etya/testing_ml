# Alur Preprocessing Optimal

Data preprocessing adalah langkah penting dalam analisis data untuk memastikan bahwa data siap digunakan untuk model pembelajaran mesin atau analisis lanjutan. Berikut adalah langkah-langkah optimal yang dapat dilakukan dalam proses ini:

## Script for Data Preparation Component (Preparation Data):
1. **Cek Duplikasi**:
   - Identifikasi dan hapus data yang duplikat untuk menghindari redundansi.
   - Contoh: Gunakan `data.duplicated()` pada pandas untuk mengecek duplikasi.

2. **Cek Missing Values**:
   - Tangani data yang hilang dengan metode seperti imputasi nilai rata-rata, median, modus, atau penghapusan baris/kolom yang terlalu banyak nilai hilangnya.
   - Contoh: `data.fillna()` atau `data.dropna()` pada pandas.

3. **Feature Engineering**:
   - Tambahkan fitur baru yang relevan atau ubah fitur yang ada untuk menambah nilai prediktif model.
   - Contoh: Membuat fitur `age` dari `year_built` atau mengekstraksi komponen tanggal dari kolom waktu.

## Script for Basic Visualization Component (Cleaned Data):
4. **Filter Outlier**:
   - Identifikasi dan atasi nilai ekstrem yang dapat memengaruhi model.
   - Metode: IQR, Z-Score, atau visualisasi seperti boxplot.
   - Contoh: `sns.boxplot(data=...)` menggunakan Seaborn.

5. **Cek Distribusi Data**:
   - Gunakan histogram atau uji statistik untuk memastikan normalitas data.
   - Metode: Histogram, Q-Q plot, uji Shapiro-Wilk.
   - Contoh: `sns.histplot(data=...)` atau `scipy.stats.shapiro(data)`.

6. **Cek Data Correlations**:
   - Lakukan pengecekan korelasi antar kolom untuk memahami hubungan antar variabel.
   - Metode: Heatmap, nilai korelasi Pearson atau Spearman.
   - Contoh: `sns.heatmap(data.corr(), annot=True)` menggunakan Seaborn.

7. **Transformasi Data**:
   - Perbaiki distribusi data yang skewed (positif/negatif) dengan transformasi log, square root, Box-Cox, atau Yeo-Johnson.
   - Contoh: `np.log1p(data)` untuk log transform.

8. **Encoding Variabel Kategorikal**:
    - Transformasikan kategori menjadi numerik menggunakan label encoding atau one-hot encoding.
    - Contoh: `pd.get_dummies(data)` untuk one-hot encoding.

9. **Normalisasi/Standardisasi**:
    - Skala data numerik untuk memastikan fitur memiliki skala yang sama.
    - Metode: Min-Max Scaling, Z-Score Standardization.
    - Contoh: `StandardScaler()` atau `MinMaxScaler()` dari scikit-learn.

10. **Cek Multikolinearitas**:
   - Identifikasi fitur yang memiliki korelasi tinggi untuk mengurangi redundansi.
   - Metode: Variance Inflation Factor (VIF).
   - Contoh: `statsmodels.stats.outliers_influence.variance_inflation_factor()`.

11. **Imbalanced Data**:
   - Tangani data tidak seimbang dengan metode oversampling (SMOTE, ADASYN) atau undersampling.
   - Contoh: Menggunakan `imblearn` untuk SMOTE atau kombinasi SMOTEENN.

## Sample Code: 
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from scipy.stats import boxcox
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Dataset contoh
data = {
    'Feature1': [1, 2, 3, 4, 5],
    'Feature2': [2, 4, 6, 8, 10],
    'Category': ['A', 'B', 'A', 'B', 'C'],
    'Target': [0, 1, 0, 0, 1]  # Imbalanced target
}
df = pd.DataFrame(data)

# 1. Transformasi fitur numerik
df['Feature1_Log'] = np.log(df['Feature1'] + 1)
df['Feature2_Log'] = np.log(df['Feature2'] + 1)

# 2. Encoding kategori
encoder = OneHotEncoder(sparse=False, drop='first')
category_encoded = encoder.fit_transform(df[['Category']])
encoded_df = pd.DataFrame(category_encoded, columns=encoder.get_feature_names_out(['Category']))
df = pd.concat([df, encoded_df], axis=1)

# 3. Normalisasi fitur numerik
scaler = MinMaxScaler()
df[['Feature1_Log', 'Feature2_Log']] = scaler.fit_transform(df[['Feature1_Log', 'Feature2_Log']])

# 4. Cek Multikolinearitas (VIF)
# Pilih fitur numerik untuk cek VIF
features_for_vif = ['Feature1_Log', 'Feature2_Log']
X = df[features_for_vif]

# Hitung VIF
vif = pd.DataFrame()
vif['Feature'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Hapus fitur dengan VIF > 10
vif_threshold = 10
selected_features = vif[vif['VIF'] < vif_threshold]['Feature']
df_vif_filtered = df[selected_features]

print("VIF Results:\n", vif)
print("Features selected after VIF:\n", selected_features)

# 5. Penanganan Imbalanced Data
X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(df_vif_filtered, df['Target'])

print("Class distribution before SMOTE:\n", df['Target'].value_counts())
print("Class distribution after SMOTE:\n", pd.Series(y_resampled).value_counts())

```


## Tips Tambahan:
- Gunakan pipeline untuk menyatukan proses preprocessing agar lebih efisien.
- Lakukan validasi setiap langkah dengan visualisasi atau uji statistik untuk memastikan kualitas data.
