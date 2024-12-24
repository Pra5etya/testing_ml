# Complete Hyperparameter Grid for Regression Models in scikit-learn

## Ridge Regression (Checked)
```python
param_grid_ridge = {
    'model__alpha': [0.01, 0.1, 1, 10, 100],
    'model__max_iter': [50000, 100000, 200000],
    'model__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs'],
    'model__fit_intercept': [True, False],
    'model__tol': [1e-4, 1e-3, 1e-2],
    'model__random_state': [None, 42],
}
```

## Lasso Regression (Checked)
```python
param_grid_lasso = {
    'model__alpha': [0.01, 0.1, 1, 10, 100],
    'model__fit_intercept': [True, False],
    'model__max_iter': [50000, 100000, 200000],
    'model__tol': [1e-4, 1e-3, 1e-2],
    'model__selection': ['cyclic', 'random'],
    'model__random_state': [None, 42]
}
```

## ElasticNet (Checked)
```python
param_grid_elasticnet = {
    'model__alpha': [0.01, 0.1, 1, 10, 100],
    'model__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
    'model__fit_intercept': [True, False],
    'model__max_iter': [50000, 100000, 200000],
    'model__tol': [1e-4, 1e-3, 1e-2],
    'model__random_state': [None, 42]
}
```

## KNeighborsRegressor
```python
param_grid_knn = {
    'model__n_neighbors': [3, 5, 7, 9, 11, 15],
    'model__weights': ['uniform', 'distance'],
    'model__metric': ['minkowski', 'euclidean', 'manhattan', 'chebyshev'],
    'model__p': [1, 2],  # Power parameter for Minkowski metric.
    'model__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'model__leaf_size': [20, 30, 50, 100],
    'model__metric_params': [None, {'p': 3}]  # Additional metric arguments.
}
```

## DecisionTreeRegressor
```python
param_grid_dtr = {
    'model__criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
    'model__splitter': ['best', 'random'],
    'model__max_depth': [None, 5, 10, 20, 50],
    'model__min_samples_split': [2, 5, 10, 20],
    'model__min_samples_leaf': [1, 2, 4, 10],
    'model__min_weight_fraction_leaf': [0.0, 0.1, 0.2],
    'model__max_features': [None, 'sqrt', 'log2'],
    'model__max_leaf_nodes': [None, 10, 20, 50],
    'model__min_impurity_decrease': [0.0, 0.01, 0.1],
    'model__random_state': [None, 42],
    'model__ccp_alpha': [0.0, 0.01, 0.1]  # Complexity parameter for minimal cost-complexity pruning.
}
```

## RandomForestRegressor
```python
param_grid_rf = {
    'model__n_estimators': [100, 200, 500, 1000],
    'model__criterion': ['squared_error', 'absolute_error', 'poisson'],
    'model__max_depth': [None, 10, 20, 30, 50],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4],
    'model__min_weight_fraction_leaf': [0.0, 0.1],
    'model__min_impurity_decrease': [0.0, 0.01, 0.1],
    'model__max_features': ['auto', 'sqrt', 'log2', None],
    'model__max_leaf_nodes': [None, 10, 20, 50],
    'model__bootstrap': [True, False],
    'model__random_state': [None, 42],
    'model__ccp_alpha': [0.0, 0.01, 0.1],
    'model__warm_start': [True, False]
}
```

## GradientBoostingRegressor
```python
param_grid_gbr = {
    'model__loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
    'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'model__n_estimators': [100, 200, 300, 500],
    'model__subsample': [0.6, 0.8, 1.0],
    'model__criterion': ['friedman_mse', 'squared_error'],
    'model__max_depth': [3, 5, 7, 10],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4],
    'model__min_weight_fraction_leaf': [0.0, 0.1],
    'model__min_impurity_decrease': [0.0, 0.01, 0.1],
    'model__max_features': ['auto', 'sqrt', 'log2'],
    'model__tol': [1e-4, 1e-3, 1e-2],
    'model__random_state': [None, 42],
    'model__ccp_alpha': [0.0, 0.01, 0.1],
    'model__warm_start': [True, False]
}
```

## Support Vector Regressor (SVR)
```python
param_grid_svr = {
    'model__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'model__C': [0.1, 1, 10, 100, 1000],
    'model__epsilon': [0.01, 0.1, 0.2, 0.5, 1],
    'model__tol': [1e-4, 1e-3, 1e-2],
    'model__degree': [2, 3, 4],  # Only for 'poly' kernel.
    'model__gamma': ['scale', 'auto'],  # Kernel coefficient.
    'model__shrinking': [True, False],
    'model__max_iter': [-1, 1000, 5000]  # Maximum iterations (-1 means no limit).
}
```

## HistGradientBoostingRegressor
```python
param_grid_hgbr = {
    'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'model__max_iter': [100, 200, 300, 500],
    'model__max_depth': [None, 5, 10, 20],
    'model__min_samples_leaf': [10, 20, 50, 100],
    'model__max_leaf_nodes': [None, 10, 20, 50, 100],
    'model__l2_regularization': [0, 0.1, 0.5, 1.0],
    'model__min_impurity_decrease': [0.0, 0.01, 0.1],
    'model__random_state': [None, 42],
    'model__scoring': ['loss', 'neg_mean_squared_error'],
    'model__early_stopping': [True, False],
    'model__validation_fraction': [0.1, 0.2, 0.3],
    'model__tol': [1e-4, 1e-3, 1e-2]
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