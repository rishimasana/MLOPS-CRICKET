import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_cricket.csv")

print("DATA LOADED SUCCESSFULLY")

# Features and target
X = df.drop("Predicted Score", axis=1)
y = df["Predicted Score"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTRAINING DATA SIZE:", X_train.shape)
print("TEST DATA SIZE:", X_test.shape)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

print("\nMODEL TRAINED SUCCESSFULLY")

# Predictions
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nMODEL PERFORMANCE")
print("Mean Absolute Error:", mae)
print("R2 Score:", r2)

# Save model
joblib.dump(model, "models/cricket_score_model.pkl")

print("\nMODEL SAVED SUCCESSFULLY")