# GridSearchCV Scoring Metriks pada Berbagai Model

Berikut adalah penjelasan tentang **scoring** dalam `GridSearchCV` dan penerapannya pada berbagai jenis model, baik untuk klasifikasi maupun regresi.

## 1. Klasifikasi
Untuk tugas klasifikasi, berikut adalah model yang sering digunakan beserta metrik yang relevan:

### Model:
- **Logistic Regression** (`LogisticRegression`)
- **Support Vector Machine** (`SVC`)
- **Random Forest** (`RandomForestClassifier`)
- **Decision Tree** (`DecisionTreeClassifier`)
- **k-Nearest Neighbors** (`KNeighborsClassifier`)
- **Naive Bayes** (`GaussianNB`, `MultinomialNB`)
- **Gradient Boosting** (`GradientBoostingClassifier`)

### Metrik `scoring` yang disarankan:
- **`accuracy`**: Cocok untuk dataset yang seimbang (jumlah data di setiap kelas serupa).
- **`precision`**: Berguna untuk kasus di mana kesalahan positif palsu (false positives) perlu diminimalkan, misalnya, dalam deteksi penipuan.
- **`recall`**: Digunakan ketika kesalahan negatif palsu (false negatives) lebih mahal, misalnya, dalam diagnosis penyakit.
- **`f1`**: Kombinasi presisi dan recall, berguna jika dataset tidak seimbang.
- **`roc_auc`**: Bagus untuk mengukur performa pada model klasifikasi biner, khususnya saat mempertimbangkan keseimbangan antara true positive rate dan false positive rate.

## 2. Regresi
Untuk tugas regresi, berikut adalah model yang cocok dengan metrik evaluasi regresi:

### Model:
- **Linear Regression** (`LinearRegression`)
- **Ridge Regression** (`Ridge`)
- **Lasso Regression** (`Lasso`)
- **Support Vector Regressor** (`SVR`)
- **Random Forest Regressor** (`RandomForestRegressor`)
- **Decision Tree Regressor** (`DecisionTreeRegressor`)
- **Gradient Boosting Regressor** (`GradientBoostingRegressor`)

### Metrik `scoring` yang disarankan:
- **`r2`**: Menilai seberapa baik model memprediksi data (nilai koefisien determinasi). Cocok digunakan untuk kebanyakan model regresi.
- **`neg_mean_squared_error`**: Mengukur kesalahan kuadrat rata-rata; negatif karena GridSearchCV memaksimalkan nilai.
- **`neg_mean_absolute_error`**: Mengukur kesalahan absolut rata-rata; lebih baik untuk mendeteksi outlier.
- **`neg_root_mean_squared_error`**: Akar dari *mean squared error*, digunakan saat ingin menilai kesalahan dalam skala yang sama dengan target.

## 3. Pertimbangan Model Spesifik
- **Support Vector Machines (SVM)**:
  - Klasifikasi: `accuracy`, `precision`, `recall`, `f1`, `roc_auc`
  - Regresi: `neg_mean_squared_error`, `r2`
- **Tree-based Models (Decision Tree, Random Forest, Gradient Boosting)**:
  - Klasifikasi: Semua metrik klasifikasi tergantung pada kebutuhan
  - Regresi: `r2`, `neg_mean_squared_error`, `neg_mean_absolute_error`
- **Logistic Regression**: Sering digunakan dengan `roc_auc`, `f1`, atau `accuracy` untuk klasifikasi.
- **K-Nearest Neighbors**: Cocok untuk `accuracy`, `f1`, atau `precision` tergantung pada data yang tidak seimbang atau skenario khusus.

## Ringkasan Penggunaan Scoring:
- Untuk **klasifikasi dengan dataset seimbang**, pilih `accuracy`.
- Untuk **klasifikasi dengan dataset tidak seimbang**, pilih `f1`, `precision`, atau `recall` berdasarkan kebutuhan.
- Untuk **regresi**, pilih `r2` jika ingin memahami seberapa baik model menjelaskan variasi, atau gunakan kesalahan (MSE, MAE) untuk menilai performa lebih langsung.

Dengan pemilihan `scoring` yang tepat, Anda dapat mengoptimalkan model secara lebih efektif.