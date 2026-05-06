import os
import pandas as pd

print("STARTING SCRIPT...")

# Get project root directory
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

file_path = os.path.join(base_dir, "data", "raw", "cricket.csv")

print("Trying to load from:", file_path)

df = pd.read_csv(file_path)

print("DATA LOADED SUCCESSFULLY ")
print(df.head())