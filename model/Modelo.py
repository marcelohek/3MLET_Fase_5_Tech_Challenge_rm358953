# -*- coding: utf-8 -*-
"""# 1. Importações
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, log_loss
import joblib

"""# 2. Carregar dados"""

X = pd.read_csv('features_preprocessed.csv')
y = pd.read_csv('target.csv')
if 'target' not in y.columns:
    y = y.iloc[:, 0]
print(f"Shape X: {X.shape}, y: {y.shape}")

"""# 3. Divisão em treino e teste"""

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Treino: {X_train.shape}, Teste: {X_test.shape}")

"""# 4. Grid Search para XGBoost"""

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.1, 0.01]
}
base_model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=3,
    scoring='roc_auc',
    verbose=1,
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
print("Melhores parâmetros:", grid_search.best_params_)
best_model = grid_search.best_estimator_

"""# 5. Avaliação do Modelo"""

# Previsões e probabilidades
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"Log Loss: {log_loss(y_test, y_proba):.4f}")


"""# 6. Salvando o Modelo Treinado"""

model_path = 'xgb_model.joblib'
joblib.dump(best_model, model_path)
print(f"Modelo XGBoost salvo em: {model_path}")