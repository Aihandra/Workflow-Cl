import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 1. Set tracking URI dan experiment
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("cropclimate_model")

# 2. Aktifkan autologging (boleh dicampur manual logging juga)
mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)

# 3. Load data hasil preprocessing
X_train = pd.read_csv("cropclimate_preprocessing/X_train.csv")
X_test = pd.read_csv("cropclimate_preprocessing/X_test.csv")
y_train = pd.read_csv("cropclimate_preprocessing/y_train.csv").values.ravel()
y_test = pd.read_csv("cropclimate_preprocessing/y_test.csv").values.ravel()

# Pastikan label integer
y_train = y_train.astype(int)
y_test = y_test.astype(int)

# 4. Inisialisasi dan training model
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluasi
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# 6. Simpan model dan logging manual
mlflow.sklearn.log_model(model, artifact_path="model")
mlflow.log_metric("accuracy", accuracy)
