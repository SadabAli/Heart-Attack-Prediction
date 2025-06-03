# 🩺 Heart Attack Prediction using ML Pipeline

This project aims to predict the likelihood of a heart attack using a machine learning model. The full development process follows an end-to-end ML pipeline with experiment tracking and data version control.

---

## 🚀 Objective

To build a robust machine learning pipeline that:
- Predicts heart attack risk
- Logs metrics to MLflow via Dagshub
- Is fully reproducible using DVC

---

## 🧰 Tools and Technologies

- **Python**: Core language for scripting
- **scikit-learn**: ML model (Random Forest)
- **pandas**: Data handling
- **DVC**: Version control for data & pipelines
- **MLflow**: Model tracking and logging
- **Dagshub**: Hosting and collaboration

---

## ⚙️ Pipeline Stages

| Stage       | Description                         | Script             |
|-------------|-------------------------------------|--------------------|
| Preprocess  | Cleans and prepares raw data        | `src/preprocess.py`|
| Train       | Trains RandomForest model           | `src/train.py`     |
| Evaluate    | Evaluates model, logs metrics       | `src/evaluate.py`  |



Run the full pipeline:

```bash
dvc repro
```
### Final Results

Model: ```RandomForestClassifier```

Tuned Hyperparameters:

```n_estimators = 100```

```max_depth = 10```

```min_samples_split = 2```

```min_samples_leaf = 1```

### Metrics:

![Screenshot 2025-06-03 202817](https://github.com/user-attachments/assets/acc1cca9-f5d7-409c-8992-6f78427c64d1)


### MLflow + Dagshub
All experiments and metrics are tracked on MLflow, hosted via Dagshub.

🔗 Dagshub Project Dashboard:
https://dagshub.com/SadabAli/MachinrLerningPipeline.mlflow


### 🖼️ Pipeline Visualization

![Screenshot 2025-06-03 202858](https://github.com/user-attachments/assets/35cd80b5-5bce-46cd-b3d0-a6db68d15d82)

### 🛠️ Setup & Run
1.Clone the repo:

```git clone https://github.com/SadabAli/Heart-Attack-Prediction.git```

```cd Heart-Attack-Prediction```

2.Install requirements:

```pip install -r requirements.txt```

3.Pull data and model using DVC:

```dvc pull```

4.Run the full pipeline:

```dvc repro```

# Author
### Mir Sadab Ali
### 🔗 GitHub: SadabAli
### 📧 Email: alisadab404@gmail.com
