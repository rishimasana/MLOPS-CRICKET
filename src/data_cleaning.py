import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("data/raw/cricket.csv")

print("FIRST 5 ROWS")
print(df.head())
print("---------------------------------------------------")

print("\nDATA INFO")
print(df.info())
print("---------------------------------------------------")

print("\nMISSING VALUES")
print(df.isnull().sum())
print("---------------------------------------------------")

# Encoding categorical columns
encoder = LabelEncoder()

df["Home/Away"] = encoder.fit_transform(df["Home/Away"])
df["Pitch Condition"] = encoder.fit_transform(df["Pitch Condition"])
df["Weather"] = encoder.fit_transform(df["Weather"])

# Features and target
X = df.drop("Predicted Score", axis=1)
y = df["Predicted Score"]

print("\nFEATURES")
print(X.head())
print("---------------------------------------------------")

print("\nTARGET")
print(y.head())
print("---------------------------------------------------")

# Save cleaned dataset
df.to_csv("data/processed/cleaned_cricket.csv", index=False)

print("\nCLEANED DATA SAVED SUCCESSFULLY")