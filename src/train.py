import pandas as pd
import numpy as np
import yaml
import os
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from urllib.parse import urlparse

os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/SadabAli/Heart-Attack-Prediction.mlflow'
os.environ['MLFLOW_TRACKING_USERNAME'] = 'SadabAli'
os.environ['MLFLOW_TRACKING_PASSWORD'] = '95437498b5a15638d8aa35540cc26470566bc98d'

def hyperparameter(X_train, y_train, param_grid):
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search

params = yaml.safe_load(open('params.yaml'))['train']

def train(data_path, model_path, random_state, n_estimators, max_depth):
    data = pd.read_csv(data_path)
    X = data.drop(columns='Result')
    y = data['Result']

    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

    with mlflow.start_run():
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)

        signature = infer_signature(X_train, y_train)

        param_grid = {
            'n_estimators': [n_estimators, 200],  # using param passed + fixed
            'max_depth': [max_depth, 10, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }

        grid_search = hyperparameter(X_train, y_train, param_grid)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")

        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_param('best_n_estimators', grid_search.best_params_['n_estimators'])
        mlflow.log_param('best_max_depth', grid_search.best_params_['max_depth'])
        mlflow.log_param('best_min_samples_split', grid_search.best_params_['min_samples_split'])
        mlflow.log_param('best_min_samples_leaf', grid_search.best_params_['min_samples_leaf'])

        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)

        mlflow.log_text(str(cm), "confusion_matrix.txt")
        mlflow.log_text(cr, "classification_report.txt")

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store != 'file':
            mlflow.sklearn.log_model(best_model, "model", registered_model_name="Best_Model")
        else:
            mlflow.sklearn.log_model(best_model, "model", signature=signature)

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)

        print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train(params['data'], params['model'], params['random_state'], params['n_estimator'], params['max_depth'])
