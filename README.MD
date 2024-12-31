# Alur Preprocessing Optimal

Data preprocessing adalah langkah penting dalam analisis data untuk memastikan bahwa data siap digunakan untuk model pembelajaran mesin atau analisis lanjutan. Berikut adalah langkah-langkah optimal yang dapat dilakukan dalam proses ini:

## Script for Data Preparation Component:
1. **Cek Duplikasi**:
   - Identifikasi dan hapus data yang duplikat untuk menghindari redundansi.
   - Contoh: Gunakan `data.duplicated()` pada pandas untuk mengecek duplikasi.

2. **Cek Missing Values**:
   - Tangani data yang hilang dengan metode seperti imputasi nilai rata-rata, median, modus, atau penghapusan baris/kolom yang terlalu banyak nilai hilangnya.
   - Contoh: `data.fillna()` atau `data.dropna()` pada pandas.

3. **Feature Engineering**:
   - Tambahkan fitur baru yang relevan atau ubah fitur yang ada untuk menambah nilai prediktif model.
   - Contoh: Membuat fitur `age` dari `year_built` atau mengekstraksi komponen tanggal dari kolom waktu.

## Script for Basic Visualization Component:
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

8. **Imbalanced Data**:
   - Tangani data tidak seimbang dengan metode oversampling (SMOTE, ADASYN) atau undersampling.
   - Contoh: Menggunakan `imblearn` untuk SMOTE atau kombinasi SMOTEENN.

9. **Cek Multikolinearitas**:
   - Identifikasi fitur yang memiliki korelasi tinggi untuk mengurangi redundansi.
   - Metode: Variance Inflation Factor (VIF).
   - Contoh: `statsmodels.stats.outliers_influence.variance_inflation_factor()`.

## Script for Analyst Sections:
10. **Encoding Variabel Kategorikal**:
    - Transformasikan kategori menjadi numerik menggunakan label encoding atau one-hot encoding.
    - Contoh: `pd.get_dummies(data)` untuk one-hot encoding.

11. **Normalisasi/Standardisasi**:
    - Skala data numerik untuk memastikan fitur memiliki skala yang sama.
    - Metode: Min-Max Scaling, Z-Score Standardization.
    - Contoh: `StandardScaler()` atau `MinMaxScaler()` dari scikit-learn.

## Tips Tambahan:
- Gunakan pipeline untuk menyatukan proses preprocessing agar lebih efisien.
- Lakukan validasi setiap langkah dengan visualisasi atau uji statistik untuk memastikan kualitas data.
