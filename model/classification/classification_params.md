# Complete Hyperparameter Grid for Classifier Models in scikit-learn

## Logistic Regression
```python
param_grid_logreg = {
    'model__penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'model__C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
    'model__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'model__max_iter': [50, 100, 200, 500, 1000],
    'model__multi_class': ['auto', 'ovr', 'multinomial'],
    'model__fit_intercept': [True, False],
    'model__warm_start': [True, False],
    'model__l1_ratio': [None, 0.1, 0.5, 0.7, 1.0]  # Only for 'elasticnet' penalty
}
```

## Support Vector Machine (SVM)
```python
param_grid_svm = {
    'model__C': [0.01, 0.1, 1, 10, 100],
    'model__kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
    'model__degree': [2, 3, 4, 5],  # Only for 'poly'
    'model__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'model__coef0': [0.0, 0.1, 0.5, 1.0],  # Only for 'poly' and 'sigmoid'
    'model__shrinking': [True, False],
    'model__probability': [True, False],
    'model__class_weight': [None, 'balanced'],
    'model__decision_function_shape': ['ovo', 'ovr'],
    'model__max_iter': [-1, 100, 500, 1000]
}
```

## Decision Tree
```python
param_grid_dt = {
    'model__criterion': ['gini', 'entropy', 'log_loss'],
    'model__splitter': ['best', 'random'],
    'model__max_depth': [None, 5, 10, 20, 50, 100],
    'model__min_samples_split': [2, 5, 10, 20],
    'model__min_samples_leaf': [1, 2, 5, 10],
    'model__min_weight_fraction_leaf': [0.0, 0.1, 0.2],
    'model__max_features': [None, 'sqrt', 'log2'],
    'model__max_leaf_nodes': [None, 10, 20, 50, 100],
    'model__class_weight': [None, 'balanced'],
    'model__ccp_alpha': [0.0, 0.01, 0.1]
}
```

## Random Forest
```python
param_grid_rf = {
    'model__n_estimators': [50, 100, 200, 500, 1000],
    'model__criterion': ['gini', 'entropy', 'log_loss'],
    'model__max_depth': [None, 5, 10, 20, 50],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 5],
    'model__min_weight_fraction_leaf': [0.0, 0.1, 0.2],
    'model__max_features': ['sqrt', 'log2', None],
    'model__max_leaf_nodes': [None, 10, 20, 50],
    'model__bootstrap': [True, False],
    'model__oob_score': [True, False],
    'model__class_weight': [None, 'balanced', 'balanced_subsample'],
    'model__ccp_alpha': [0.0, 0.01, 0.1]
}
```

## Gradient Boosting
```python
param_grid_gb = {
    'model__loss': ['log_loss', 'deviance', 'exponential'],
    'model__learning_rate': [0.001, 0.01, 0.1, 0.2, 0.5],
    'model__n_estimators': [50, 100, 200, 500],
    'model__subsample': [0.5, 0.6, 0.8, 1.0],
    'model__criterion': ['friedman_mse', 'squared_error', 'mse'],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 5, 10],
    'model__min_weight_fraction_leaf': [0.0, 0.1],
    'model__max_depth': [3, 5, 10, 20],
    'model__max_features': ['sqrt', 'log2', None],
    'model__max_leaf_nodes': [None, 10, 20],
    'model__ccp_alpha': [0.0, 0.01, 0.1]
}
```

## AdaBoost
```python
param_grid_ab = {
    'model__n_estimators': [50, 100, 200, 500],
    'model__learning_rate': [0.001, 0.01, 0.1, 0.5, 1.0],
    'model__algorithm': ['SAMME', 'SAMME.R']
}
```

## Naive Bayes
### Gaussian NB
```python
param_grid_gnb = {
    'model__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
}
```
### MultinomialNB
```python
param_grid_mnb = {
    'model__alpha': [0.1, 0.5, 1.0, 2.0, 5.0],
    'model__fit_prior': [True, False]
}
```

## XGBoost
```python
param_grid_xgb = {
    'model__n_estimators': [50, 100, 200, 500],
    'model__learning_rate': [0.01, 0.1, 0.2, 0.3],
    'model__max_depth': [3, 5, 7, 10],
    'model__subsample': [0.6, 0.8, 1.0],
    'model__colsample_bytree': [0.6, 0.8, 1.0],
    'model__gamma': [0, 0.1, 0.5, 1],
    'model__reg_alpha': [0, 0.1, 1, 10],
    'model__reg_lambda': [0, 0.1, 1, 10],
    'model__scale_pos_weight': [1, 2, 5, 10]
}
```

## LightGBM
```python
param_grid_lgbm = {
    'model__n_estimators': [50, 100, 200, 500],
    'model__learning_rate': [0.001, 0.01, 0.1, 0.2],
    'model__max_depth': [-1, 5, 10, 20],
    'model__num_leaves': [31, 50, 100, 200],
    'model__min_child_samples': [5, 10, 20, 50],
    'model__min_child_weight': [0.001, 0.01, 0.1],
    'model__subsample': [0.6, 0.8, 1.0],
    'model__colsample_bytree': [0.6, 0.8, 1.0],
    'model__reg_alpha': [0.0, 0.1, 1.0],
    'model__reg_lambda': [0.0, 0.1, 1.0],
    'model__class_weight': [None, 'balanced']
}
```

## CatBoost
```python
param_grid_catboost = {
    'model__iterations': [100, 200, 500, 1000],
    'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'model__depth': [3, 5, 7, 10],
    'model__l2_leaf_reg': [1, 3, 5, 7],
    'model__border_count': [32, 64, 128],
    'model__bagging_temperature': [0, 1, 5],
    'model__class_weights': [None, 'balanced']
}
```

# Penerapan parameter yang baik
```python
param_grid = [
    {
        'param_1'
    }, 
    {
        'param_2'
    }
]
```