import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
import pickle
import os


train = pd.read_csv("TrainData.csv")
test = pd.read_csv("TestData.csv")

X_train = train.drop("quality", axis=1)
X_test = test.drop("quality", axis=1)
y_train = train["quality"]
y_test = test["quality"]

mlflow.set_experiment("RandomForestClassifier")
with mlflow.start_run(run_name="RandomForestClassifier"):
    n_estimators = 100
    max_depth = 10
    min_samples_split = 2
    min_samples_leaf = 5
    model = RandomForestClassifier(criterion = 'gini',n_estimators=100,max_depth=6,random_state=0)

    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("min_samples_split", min_samples_split)
    mlflow.log_param("min_samples_leaf", min_samples_leaf)
    mlflow.log_param("accuracy", accuracy)
    mlflow.sklearn.log_model(model, name = "model")

    mlflow.log_param("train_size",  len(X_train))
    mlflow.log_param("test_size", len(X_test))

    os.makedirs("models", exist_ok=True)
    with open("models/RandomForestModel.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f'accuracy {accuracy:.4f}')