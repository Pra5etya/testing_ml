# Pengecekan data berdistribusi normal atau tidak
Jadi, urutan pengerjaannya adalah:
1. Tentukan jenis data: kategorikal atau numerik.
2. Jika kategorikal, langsung lakukan Chi-Square Test.
3. Jika numerik, mulai dengan uji normalitas, lanjutkan dengan uji homogenitas varians, dan akhirnya gunakan t-test atau Mann-Whitney U Test.


# Penerapan Analisa Statistik
## Shapiro (Menguji Normalitas Data)
1. jika value < 0.05 maka distribusi tidak normal
2. jika vaue > 0.05 maka distribusi normal

```python
from scipy.stats import shapiro

stat_a, p_a = shapiro(produk_a)
stat_b, p_b = shapiro(produk_b)

print(f"Produk A: Statistik={stat_a}, p-value={p_a}")
print(f"Produk B: Statistik={stat_b}, p-value={p_b}")

```

## Levene (Menguji varian antar group [homogen == sama])
1. jika value < 0.05 maka kelompok tidak homogen, perlu dilakuakn uji non parametrik seperti: **Mann-Whitney U Test**
2. jika value > 0.05 maka kelompok homogen

```python
from scipy.stats import levene

# Uji homogenitas varians (Levene's Test)
stat_levene, p_value_levene = levene(group1, group2)
print("\nUji Homogenitas Varians (Levene's Test):")
print(f"Statistik: {stat_levene}, P-Value: {p_value_levene}")

```

## Uji Perbandingan Rata-rata (T-Test atau Mann-Whitney U Test)
### jika data normal dan homogen (berlaku jika keduanya benar): 
1. Uji t-independen: 
```python
from scipy.stats import ttest_ind

# Uji t-independen (t-test)
stat_ttest, p_value_ttest = ttest_ind(group1, group2)
print("\nUji T-Independen (T-Test):")
print(f"Statistik: {stat_ttest}, P-Value: {p_value_ttest}")

```

### Jika data tidak normal atau varians tidak homogen: 
1. Uji Mann-Whitney U: 
```python
from scipy.stats import mannwhitneyu

# Uji Mann-Whitney U (non-parametrik)
stat_mannwhitney, p_value_mannwhitney = mannwhitneyu(group1, group2)
print("\nUji Mann-Whitney U (Non-Parametrik):")
print(f"Statistik: {stat_mannwhitney}, P-Value: {p_value_mannwhitney}")

```

## Uji Chi-Square (Jika data kategori): Data harus berbentuk array atau crosstab (pd.crosstab(col, col))
1. jika value < 0.05 maka tidak ada hubungan yang signifikan
2. jika value > 0.05 maka terdapat hubungan yang signifikan

```python
from scipy.stats import chi2_contingency
import pandas as pd
import numpy as np

# Misalkan kita punya dua kolom data kategorikal dalam format list atau DataFrame
jenis_kendaraan = ['Mobil', 'Mobil', 'Motor', 'Sepeda', 'Motor', 'Mobil', 'Sepeda']
status_rumah = ['Memiliki Rumah', 'Tidak Memiliki Rumah', 'Memiliki Rumah', 'Memiliki Rumah', 'Tidak Memiliki Rumah', 'Tidak Memiliki Rumah', 'Memiliki Rumah']

# Membuat tabel kontingensi
tabel_kontingensi = pd.crosstab(jenis_kendaraan, status_rumah)
print(type(tabel_kontingensi))
tabel_kontingensi

# Menggunakan chi2_contingency pada tabel kontingensi
chi2, p, dof, expected = chi2_contingency(tabel_kontingensi)

print(f"Chi-Square Statistic: {chi2}")
print(f"P-Value: {p}")

```