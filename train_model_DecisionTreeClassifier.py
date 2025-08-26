import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
import pickle

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

train_data = pd.read_csv('TrainData.csv')
test_data = pd.read_csv('TestData.csv')
X_train = train_data.drop("quality", axis=1)
X_test = test_data.drop("quality", axis=1)
y_train = train_data["quality"]
y_test = test_data["quality"]

mlflow.set_experiment("DecisionTreeClassifier")
with mlflow.start_run(run_name="DecisionTreeClassifier"):
    max_depth = 20
    min_samples_leaf = 5
    model = DecisionTreeClassifier(max_depth = max_depth, min_samples_leaf=min_samples_leaf,  random_state=0)

    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    mlflow.log_param("model_type", "DecisionTreeClassifier")
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("min_samples_leaf", min_samples_leaf)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, name = "DecisionTreeClassifier")

    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("test_size", len(X_test))

    with open("models/DecisionTreeClassifier.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"Accuracy: {accuracy:.4f}")


