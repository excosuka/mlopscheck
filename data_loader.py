import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



data = pd.read_csv('WineQT.csv')
data = data.drop("Id", axis=1)

X = data.drop("quality", axis=1)
X = StandardScaler().fit_transform(X)

y = data["quality"]

X_train, X_test, y_train, y_test = train_test_split(pd.DataFrame(X), y, test_size=0.25, random_state=3)

pd.concat([X_train, y_train], axis=1).to_csv("TrainData.csv", index=False)
pd.concat([X_test, y_test], axis=1).to_csv("TestData.csv", index=False)

