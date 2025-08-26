import pickle

import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

train = pd.read_csv("TrainData.csv")
test = pd.read_csv("TestData.csv")

X_train = train.drop(["quality"], axis=1)
y_train = train["quality"]
X_test = test.drop(["quality"], axis=1)
y_test = test["quality"]

mlflow.set_experiment("GradientBoostingClassifier")
with mlflow.start_run(run_name="GradientBoostingClassifier"):
    n_estimators = 100
    learning_rate = 0.01
    max_depth = 10
    min_samples_split = 2
    min_samples_leaf = 5
    model = GradientBoostingClassifier(n_estimators=n_estimators,
                                       learning_rate=learning_rate,
                                       max_depth=max_depth,
                                       min_samples_split=min_samples_split,
                                       min_samples_leaf=min_samples_leaf)

    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    mlflow.log_param("model_type", "GradientBoostingClassifier")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("min_samples_split", min_samples_split)
    mlflow.log_param("min_samples_leaf", min_samples_leaf)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, name = "GradientBoostingClassifierModel")

    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("test_size", len(X_test))
    with open("models/GradientBoostingModel.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"accuracy: {accuracy:.4f}")