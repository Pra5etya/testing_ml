# Test and Split
1. Dilakukan ketika tidak ada dua file train dan test secara terpisah pada modul
```
from sklearn.model_selection import train_test_split
```


# Encoding
## Label Encoding
1. Model yang Tidak Sensitif terhadap Urutan Angka:
```
Decision Trees, 
Random Forest, 
Gradient Boosting Machines (GBM), 
Support Vector Machines (SVM), 
K-Nearest Neighbors (KNN), 
Naive Bayes
```

2. Data dengan Urutan Alami:
misalnya yang memiliki urutan Poor, Fair, Good, Excellent.

## One Hot Encoding
1. Model Sensitif terhadap Nilai Angka: 
```
Linear Regression, 
Logistic Regression, 
Neural Networks, 
Support Vector Machines (SVM) dengan kernel linear, 
AdaBoost
```

2. Data Nominal