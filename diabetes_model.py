import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


df = pd.read_csv("diabetes.csv")


X = df.iloc[:, :-1]  
y = df.iloc[:, -1]   


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = LogisticRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")


if not os.path.exists("models"):
    os.makedirs("models")
if not os.path.exists("scaler"):
    os.makedirs("scaler")


joblib.dump(model, "models/diabetes_model.pkl")
joblib.dump(scaler, "scaler/diabetes_scaler.pkl")

print("Diabetes model saved successfully!")
