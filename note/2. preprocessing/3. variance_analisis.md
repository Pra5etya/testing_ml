# Analisis Univariate, Bivariate, dan Multivariate pada Dataset Besar

Melakukan analisis univariate, bivariate, dan multivariate pada dataset dengan jumlah kolom yang besar dan data yang masif memerlukan pendekatan yang terstruktur dan efisien. Berikut adalah langkah-langkahnya:

---

## **1. Analisis Univariate**
Univariate bertujuan memahami distribusi satu variabel pada satu waktu.

### **Langkah-langkah:**

- **Pilih Subset Variabel Penting**  
  Gunakan fitur engineering atau domain knowledge untuk memilih variabel penting sebelum analisis.

- **Gunakan Deskripsi Statistik Ringkas**  
  - Gunakan `.describe()` di Pandas untuk meringkas statistik seperti mean, median, dan standar deviasi.
  - Visualisasi distribusi data dengan histogram atau boxplot untuk kolom numerik.
  - Gunakan countplot atau pie chart untuk kolom kategorikal.

- **Automasi Visualisasi**  
  Looping untuk membuat plot untuk semua kolom numerik/kategorikal:

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Visualisasi univariate untuk kolom numerik
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()

# Visualisasi univariate untuk kolom kategorikal
for col in df.select_dtypes(include=['object']).columns:
    plt.figure(figsize=(8, 4))
    sns.countplot(x=df[col])
    plt.title(f'Countplot of {col}')
    plt.xticks(rotation=45)
    plt.show()
```

---

## **2. Analisis Bivariate**
Menganalisis hubungan dua variabel sekaligus.

### **Langkah-langkah:**

- **Numerik vs Numerik**  
  - Gunakan scatterplot atau heatmap untuk korelasi.  
  - Contoh:

```python
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()
```

- **Numerik vs Kategorikal**  
  - Gunakan boxplot untuk menganalisis distribusi numerik terhadap kategori.  
  - Contoh:

```python
for cat_col in df.select_dtypes(include=['object']).columns:
    for num_col in df.select_dtypes(include=['float64', 'int64']).columns:
        sns.boxplot(x=df[cat_col], y=df[num_col])
        plt.title(f'{num_col} by {cat_col}')
        plt.xticks(rotation=45)
        plt.show()
```

- **Kategorikal vs Kategorikal**  
  - Gunakan crosstab atau heatmap.  
  - Contoh:

```python
pd.crosstab(df['col1'], df['col2']).plot(kind='bar', stacked=True)
```

---

## **3. Analisis Multivariate**
Menganalisis hubungan banyak variabel sekaligus.

### **Langkah-langkah:**

- **Pairplot untuk Hubungan Numerik**  
  Gunakan `sns.pairplot()` untuk visualisasi antar-variabel numerik.  
  
```python
sns.pairplot(df.select_dtypes(include=['float64', 'int64']))
plt.show()
```

- **PCA (Principal Component Analysis)**  
  Untuk data dengan dimensi tinggi, PCA membantu mereduksi dimensi sambil mempertahankan informasi.

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data_scaled = scaler.fit_transform(df.select_dtypes(include=['float64', 'int64']))

pca = PCA(n_components=2)
principal_components = pca.fit_transform(data_scaled)

plt.scatter(principal_components[:, 0], principal_components[:, 1])
plt.title("PCA Scatter Plot")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()
```

- **Clustering untuk Relasi Multivariabel**  
  Gunakan algoritme seperti K-Means untuk melihat pola antar kelompok.  

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(data_scaled)

sns.scatterplot(x=principal_components[:, 0], y=principal_components[:, 1], hue=df['cluster'], palette='viridis')
plt.show()
```

---

## **Tips Efisiensi**

1. **Gunakan Sampling**  
   Jika data terlalu besar, gunakan sampling acak yang representatif sebelum analisis.

2. **Pipeline Otomasi**  
   Buat pipeline yang otomatis menganalisis setiap kolom dengan teknik tertentu.

3. **Gunakan Library EDA Otomatis**  
   Library seperti:
   - **Pandas Profiling**:  
     ```python
     from pandas_profiling import ProfileReport
     profile = ProfileReport(df)
     profile.to_notebook_iframe()
     ```
   - **Sweetviz**:  
     ```python
     import sweetviz as sv
     report = sv.analyze(df)
     report.show_html('report.html')
     ```

4. **Filter Variabel Penting**  
   Gunakan feature importance (misalnya, menggunakan Random Forest) untuk fokus pada variabel yang relevan.

5. **Distribusi di Komputasi Terdistribusi**  
   Untuk dataset besar, gunakan framework seperti **Dask** atau **PySpark**.

---

Dengan pendekatan ini, analisis tetap efisien dan memberikan wawasan yang bermakna meskipun dataset sangat besar.
