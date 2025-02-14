import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


model_dir = "models"
scaler_dir = "scalers"


os.makedirs(model_dir, exist_ok=True)
os.makedirs(scaler_dir, exist_ok=True)


df = pd.read_csv(r"C:\Users\ASUS\OneDrive\Desktop\Project\parkinsons.csv")


print("Dataset Columns:", df.columns)
print("Dataset Shape:", df.shape)


target_col = df.columns[-1]
print("Target Column:", target_col)


threshold = df[target_col].median()
df[target_col] = df[target_col].apply(lambda x: 1 if x >= threshold else 0)


X = df.drop(columns=[target_col]) 
y = df[target_col]


X = X.apply(pd.to_numeric, errors='coerce')


X.fillna(X.mean(), inplace=True)  


X = X.loc[:, X.std() > 0]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Parkinson’s Model Accuracy: {accuracy:.2f}")


model_path = os.path.join(model_dir, "parkinsons_model.pkl")
scaler_path = os.path.join(scaler_dir, "parkinsons_scaler.pkl")

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

print(f"Parkinson’s model saved successfully in {model_dir}!")
print(f"Parkinson’s scaler saved successfully in {scaler_dir}!")
