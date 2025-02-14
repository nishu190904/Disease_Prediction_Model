import pandas as pd
import os


base_path = r"C:\Users\ASUS\OneDrive\Desktop\Project"


diabetes_df = pd.read_csv(os.path.join(base_path, "diabetes.csv"))
heart_df = pd.read_csv(os.path.join(base_path, "heart.csv"))
parkinsons_df = pd.read_csv(os.path.join(base_path, "parkinsons.csv"))


print("Diabetes Data:\n", diabetes_df.head())
print("Heart Disease Data:\n", heart_df.head())
print("Parkinson’s Disease Data:\n", parkinsons_df.head())


print("Missing Values in Diabetes:\n", diabetes_df.isnull().sum())
print("Missing Values in Heart Disease:\n", heart_df.isnull().sum())
print("Missing Values in Parkinson’s:\n", parkinsons_df.isnull().sum())


diabetes_df.fillna(diabetes_df.mean(numeric_only=True), inplace=True)
heart_df.fillna(heart_df.mean(numeric_only=True), inplace=True)
parkinsons_df.fillna(parkinsons_df.mean(numeric_only=True), inplace=True)

print("Missing values after filling:")
print("Diabetes:", diabetes_df.isnull().sum().sum())
print("Heart:", heart_df.isnull().sum().sum())
print("Parkinson’s:", parkinsons_df.isnull().sum().sum())

