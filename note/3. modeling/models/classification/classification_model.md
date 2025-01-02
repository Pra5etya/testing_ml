# **Pipeline Klasifikasi Berdasarkan Level**

---

## **1. Level Pemula: Model Dasar**

### **Pipeline**
1. **Preprocessing**:
   - Impute missing values (`SimpleImputer`).
   - Scale numerical features (`StandardScaler` or `MinMaxScaler`).

2. **Models**:
   - `LogisticRegression`
   - `RidgeClassifier`
   - `KNeighborsClassifier`
   - `DummyClassifier`

### **Example Pipeline Implementation**
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier

# Example pipeline for Logistic Regression
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(random_state=42))
])

# Example pipeline for K-Nearest Neighbors
pipeline_knn = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('model', KNeighborsClassifier(n_neighbors=5))
])
```

---

## **2. Level Menengah: Model Intermediate**

### **Pipeline**
1. **Preprocessing**:
   - Impute missing values (`SimpleImputer`).
   - Scale or normalize features (`StandardScaler` or `Normalizer`).

2. **Models**:
   - `SVC`
   - `LinearSVC`
   - `NuSVC`
   - `DecisionTreeClassifier`
   - `RandomForestClassifier`

### **Example Pipeline Implementation**
```python
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Example pipeline for SVC
pipeline_svc = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('model', SVC(kernel='rbf', probability=True, random_state=42))
])

# Example pipeline for Random Forest
pipeline_rf = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
])
```

---

## **3. Level Lanjutan: Model Kompleks**

### **Pipeline**
1. **Preprocessing**:
   - Impute missing values (`SimpleImputer`).
   - Scale features (`StandardScaler`) or use advanced preprocessing if necessary.

2. **Models**:
   - `GradientBoostingClassifier`
   - `HistGradientBoostingClassifier`
   - `AdaBoostClassifier`
   - `MLPClassifier`

### **Example Pipeline Implementation**
```python
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

# Example pipeline for Gradient Boosting Classifier
pipeline_gb = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('model', GradientBoostingClassifier(random_state=42))
])

# Example pipeline for MLP Classifier
pipeline_mlp = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('model', MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42))
])
```

---

## **4. Level Ahli: Meta-Estimasi dan Custom Models**

### **Pipeline**
1. **Preprocessing**:
   - Combine multiple transformers (`ColumnTransformer`).
   - Use feature selection or dimensionality reduction (e.g., `PCA` or `SelectKBest`).

2. **Meta-Estimators**:
   - `VotingClassifier`
   - `StackingClassifier`
   - `CalibratedClassifierCV`

### **Example Pipeline Implementation**
```python
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Example pipeline for Voting Classifier
pipeline_voting = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('model', VotingClassifier(estimators=[
        ('lr', LogisticRegression(random_state=42)),
        ('dt', DecisionTreeClassifier(random_state=42)),
        ('svc', SVC(probability=True, random_state=42))
    ], voting='soft'))
])

# Example pipeline for Stacking Classifier
pipeline_stacking = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('model', StackingClassifier(estimators=[
        ('lr', LogisticRegression(random_state=42)),
        ('dt', DecisionTreeClassifier(random_state=42))
    ], final_estimator=SVC(random_state=42)))
])
```





# Kombinasi Model Lengkap pada VotingClassifier dan StackingClassifier

## 1. **Voting Classifier dengan GridSearchCV**

Pada Voting Classifier, kita akan menerapkan GridSearchCV untuk mencari hyperparameter terbaik dari model individual, seperti `LogisticRegression`, `SVC`, dan `RandomForestClassifier`. Selain itu, kita juga akan menggunakan **PCA** untuk reduksi dimensi.

### Contoh Kode:

```python
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Memuat dataset contoh
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Definisikan model individu
model1 = LogisticRegression(max_iter=1000)
model2 = SVC(probability=True)
model3 = RandomForestClassifier()

# Definisikan pipeline dengan PCA dan VotingClassifier
pipeline = Pipeline([
    ('pca', PCA()),
    ('voting', VotingClassifier(estimators=[
        ('lr', model1),
        ('svc', model2),
        ('rf', model3)
    ], voting='soft'))
])

# Hyperparameter grid for GridSearchCV
param_grid = [
    # Logistic Regression
    {
        'pca__n_components': [0.90, 0.95, 0.99],
        'voting__lr': [LogisticRegression(max_iter=1000)],
    },
    # Support Vector Classifier
    {
        'pca__n_components': [0.90, 0.95, 0.99],
        'voting__svc': [SVC(probability=True)],
        'voting__svc__C': [0.1, 1.0, 10.0],
        'voting__svc__kernel': ['linear', 'rbf'],
    },
    # Random Forest Classifier
    {
        'pca__n_components': [0.90, 0.95, 0.99],
        'voting__rf': [RandomForestClassifier()],
        'voting__rf__n_estimators': [50, 100, 200],
        'voting__rf__max_depth': [None, 10, 20],
    }
]

# GridSearchCV untuk mencari hyperparameter terbaik
grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Evaluasi model terbaik
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best accuracy: {grid_search.best_score_}")
```

## 2. **Stacking Classifier dengan GridSearchCV**
Pada Stacking Classifier, kita juga akan menggunakan GridSearchCV untuk mencari hyperparameter terbaik dari model dasar dan model meta.

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Memuat dataset contoh
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Definisikan model dasar
base_learners = [
    ('lr', LogisticRegression(max_iter=1000)),
    ('svc', SVC(probability=True)),
    ('rf', RandomForestClassifier())
]

# Model meta
meta_model = LogisticRegression()

# Definisikan pipeline untuk Stacking Classifier
pipeline = Pipeline([
    ('stacking', StackingClassifier(estimators=base_learners, final_estimator=meta_model))
])

# Hyperparameter grid for GridSearchCV
param_grid = [
    # Logistic Regression as base learner
    {
        'stacking__lr': [LogisticRegression(max_iter=1000)],
    },
    # Support Vector Classifier as base learner
    {
        'stacking__svc': [SVC(probability=True)],
        'stacking__svc__C': [0.1, 1.0, 10.0],
        'stacking__svc__kernel': ['linear', 'rbf'],
    },
    # Random Forest Classifier as base learner
    {
        'stacking__rf': [RandomForestClassifier()],
        'stacking__rf__n_estimators': [50, 100, 200],
        'stacking__rf__max_depth': [None, 10, 20],
    },
    # Hyperparameters for final estimator (meta model)
    {
        'stacking__final_estimator__C': [0.1, 1.0, 10.0],
        'stacking__final_estimator__max_iter': [100, 1000],
    }
]

# GridSearchCV untuk mencari hyperparameter terbaik
grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Evaluasi model terbaik
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best accuracy: {grid_search.best_score_}")
```