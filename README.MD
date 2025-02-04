# Alur Preprocessing Optimal

Data preprocessing adalah langkah penting dalam analisis data untuk memastikan bahwa data siap digunakan untuk model pembelajaran mesin atau analisis lanjutan. Berikut adalah langkah-langkah optimal yang dapat dilakukan dalam proses ini:

## Script for Data Preparation Component (***Preparation Data***):

1. **Cek Duplikasi**:
   - Identifikasi dan hapus data yang duplikat untuk menghindari redundansi.
   - Contoh: Gunakan `data.duplicated()` pada pandas untuk mengecek duplikasi.

2. **Cek Missing Values**:
   - Tangani data yang hilang dengan metode seperti imputasi nilai rata-rata, median, modus, atau penghapusan baris/kolom yang terlalu banyak nilai hilangnya.
   - Contoh: `data.fillna()` atau `data.dropna()` pada pandas.

3. **Feature Engineering**:
   - Tambahkan fitur baru yang relevan atau ubah fitur yang ada untuk menambah nilai prediktif model.
   - Contoh: Membuat fitur `age` dari `year_built` atau mengekstraksi komponen tanggal dari kolom waktu.

## Script for Basic Visualization Component (***Exploration Data***):

4. **Cek Distribusi Data**:
   - Gunakan histogram atau uji statistik untuk memastikan normalitas data.
   - Metode: Histogram, Q-Q plot, uji Shapiro-Wilk.

5. **Transformasi Data**:
   - Perbaiki distribusi data yang skewed (positif/negatif) dengan transformasi log, square root, Box-Cox, atau Yeo-Johnson.
   - Contoh: `np.log1p(data)` untuk log transform.
   - Contoh: `sns.histplot(data=...)` atau `scipy.stats.shapiro(data)`.

6. **Filter Outlier**:
   - Identifikasi dan atasi nilai ekstrem yang dapat memengaruhi model.
   - Metode: IQR, Z-Score, atau visualisasi seperti boxplot.
   - Contoh: `sns.boxplot(data=...)` menggunakan Seaborn.

7. **Cek Multikolinearitas**:
   - Identifikasi fitur yang memiliki korelasi tinggi untuk mengurangi redundansi.
   - Metode: Variance Inflation Factor (VIF).
   - Contoh: `statsmodels.stats.outliers_influence.variance_inflation_factor()`.

8. **Imbalanced Data**:
   - Tangani data tidak seimbang dengan metode oversampling (SMOTE, ADASYN) atau undersampling.
   - Contoh: Menggunakan `imblearn` untuk SMOTE atau kombinasi SMOTEENN.

9. **Cek Data Correlations**:
   - Lakukan pengecekan korelasi antar kolom untuk memahami hubungan antar variabel.
   - Metode: Heatmap, nilai korelasi Pearson atau Spearman.
   - Contoh: `sns.heatmap(data.corr(), annot=True)` menggunakan Seaborn.

## Script for Basic model Component (Model Data):

10. **Menghapus Noise atau Data Tidak Relevan**:
    - Mengurangi Dimensi Data, Menghindari Kesalahan dalam Transformasi Data, serta Memastikan Hanya Fitur yang Berkontribusi yang Diproses Lebih Lanjut

   - Contoh: **Varianve Threshold**: 
   1. Jika ada fitur yang konstan atau memiliki sedikit variasi (misalnya credit_card_limit jika nilainya selalu 20000)
   2. Biasanya digunakan di awal preprocessing

   ```python
   from sklearn.feature_selection import VarianceThreshold

   # Hanya memilih fitur numerik
   X_numeric = df.select_dtypes(include=['number'])

   # Menghapus fitur dengan variansi rendah (konstan)
   selector = VarianceThreshold(threshold=0.01)
   X_selected = selector.fit_transform(X_numeric)
   ```

   - Contoh: **Select KBest** : 
   1. Jika ingin menentukan fitur mana yang paling relevan untuk prediksi
   2. Cocok untuk dataset dengan banyak fitur

   ```python
   from sklearn.feature_selection import SelectKBest, f_classif

   X = df.drop(columns=['fraud_status'])  # Fitur
   y = df['fraud_status']  # Target

   # Pilih 10 fitur terbaik berdasarkan ANOVA F-score
   selector = SelectKBest(score_func=f_classif, k=10)
   X_selected = selector.fit_transform(X, y)
   ```

   - Contoh: **Recursive Feature Elimination (RFE)** : 
   1. Saat sudah memilih model utama dan ingin menyaring fitur yang tidak penting
   2. Cocok jika ingin otomatis memilih fitur berdasarkan pengaruhnya terhadap model

   ```python
   from sklearn.feature_selection import RFE
   from sklearn.ensemble import RandomForestClassifier

   # Model dasar
   model = RandomForestClassifier()

   # Pilih 10 fitur terbaik
   selector = RFE(estimator=model, n_features_to_select=10, step=1)
   X_selected = selector.fit_transform(X, y)
   ```

11. **Encoding Variabel Kategorikal**:
    - Transformasikan kategori menjadi numerik menggunakan label encoding atau one-hot encoding.
    - Contoh: `pd.get_dummies(data)` untuk one-hot encoding.

12. **Normalisasi/Standardisasi**:
    - Skala data numerik untuk memastikan fitur memiliki skala yang sama.
    - Metode: Min-Max Scaling, Z-Score Standardization.
    - Contoh: `StandardScaler()` atau `MinMaxScaler()` dari scikit-learn.

## Tips Tambahan:
- Gunakan pipeline untuk menyatukan proses preprocessing agar lebih efisien.
- Lakukan validasi setiap langkah dengan visualisasi atau uji statistik untuk memastikan kualitas data.
