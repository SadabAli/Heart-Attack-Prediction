import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import mlflow
from mlflow.models import infer_signature
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from urllib.parse import urlparse

# Set up MLflow tracking
os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/SadabAli/MachinrLerningPipeline.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = "SadabAli"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "95437498b5a15638d8aa35540cc26470566bc98d"

# Load parameters
params_all = yaml.safe_load(open('params.yaml'))
train_params = params_all['train']
eval_params = params_all.get('evaluate', {})  # in case you added target_col here

# Default target column
target_col = eval_params.get('target_col', 'Result')  # fallback if not defined

def evaluate(data_path, model_path):
    data = pd.read_csv(data_path)

    print("Available columns:", data.columns.tolist())  # For debugging

    X = data.drop(columns=target_col)
    y = data[target_col]

    # Load model
    model = pickle.load(open(model_path, 'rb'))

    # Predict and evaluate
    predictions = model.predict(X)
    acc = accuracy_score(y, predictions)

    # Log to MLflow
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
    mlflow.log_metric('accuracy', acc)

    print(f"Model Accuracy: {acc}")

if __name__ == '__main__':
    evaluate(train_params['data'], train_params['model'])
