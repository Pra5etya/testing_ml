# Valuasi Performa Model

Gunakan metrik yang sesuai dengan tipe masalah Anda untuk mengevaluasi performa model dengan tepat:

## Regression

- **Mean Squared Error (MSE):** Sensitif terhadap outlier.
- **Mean Absolute Error (MAE):** Lebih robust terhadap outlier.
- **R² (Determination Coefficient):** Menilai seberapa baik model menjelaskan variabilitas data.

### Contoh Kode:
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Prediksi dan nilai aktual
y_true = [3.0, -0.5, 2.0, 7.0]
y_pred = [2.5, 0.0, 2.0, 8.0]

# Metrik regresi
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"MSE: {mse}")
print(f"MAE: {mae}")
print(f"R²: {r2}")
```

## Classification

- **Accuracy:** Proporsi prediksi yang benar.
- **Precision/Recall:** Penting jika Anda memiliki data tidak seimbang.
- **F1-Score:** Kombinasi dari precision dan recall.
- **ROC-AUC:** Evaluasi performa model secara keseluruhan.

### Contoh Kode:
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Prediksi dan nilai aktual
y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 1, 0, 0]
y_prob = [0.1, 0.9, 0.8, 0.2, 0.4]  # Probabilitas untuk ROC-AUC

# Metrik klasifikasi
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_prob)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")
print(f"ROC-AUC: {roc_auc}")
```

## Clustering

- **Silhouette Score:** Mengukur seberapa baik data terklasterisasi.
- **Calinski-Harabasz Index:** Mengukur rasio densitas intra-kluster terhadap inter-kluster.

### Contoh Kode:
```python
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Data untuk clustering
X, _ = make_blobs(n_samples=100, centers=3, random_state=42)

# Model clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_

# Metrik clustering
sil_score = silhouette_score(X, labels)
ch_score = calinski_harabasz_score(X, labels)

print(f"Silhouette Score: {sil_score}")
print(f"Calinski-Harabasz Index: {ch_score}")
```

## Anomaly Detection

- **Precision/Recall:** Efektif untuk kasus anomali jarang.
- **False Positive Rate:** Pastikan jumlah alarm palsu rendah.

### Contoh Kode:
```python
from sklearn.metrics import precision_score, recall_score

# Prediksi dan nilai aktual
y_true = [0, 0, 0, 1, 0, 1, 0, 0, 1]  # 1 adalah anomali
y_pred = [0, 0, 0, 1, 0, 0, 0, 0, 1]

# Metrik anomaly detection
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
```

## Optimisasi Model

### Grid Search dan Random Search CV
- **Grid Search:** Pada pencarian grid search, gunakan `n_jobs = -1` untuk memanfaatkan semua core CPU dan mempercepat proses.
- **Random Search CV:** Membantu dalam menemukan hyperparameter terbaik pada beberapa model dengan lebih cepat dibanding grid search.

### Hyperparameter Optimization Framework
Gunakan library yang lebih canggih untuk hyperparameter tuning:

- **Optuna:** Efisien dengan metode Bayesian Optimization.
- **Hyperopt:** Alternatif populer untuk pendekatan yang serupa.
- **Ray Tune:** Skalabel untuk distribusi besar.

### Contoh Kode (Optuna):
```python
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Fungsi objektif untuk tuning
def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 10, 200)
    max_depth = trial.suggest_int('max_depth', 2, 32)
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    return cross_val_score(clf, X, y, cv=3).mean()

# Tuning hyperparameter
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print("Best Parameters:", study.best_params)
```

## Early Stopping
Untuk model seperti Gradient Boosting atau XGBoost, gunakan fitur early stopping untuk menghentikan pelatihan jika performa model tidak meningkat setelah beberapa iterasi.

### Contoh Kode:
```python
from xgboost import XGBRegressor

model = XGBRegressor(n_estimators=1000, early_stopping_rounds=10, eval_metric='rmse', random_state=42)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)
```
