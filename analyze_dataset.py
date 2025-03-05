import pandas as pd
import os

# Load the dataset
data_path = "Dataset.xlsx"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found at {data_path}")

df = pd.read_excel(data_path)

# Display basic information
print("Dataset Head:")
print(df.head())
print("\nDataset shape:", df.shape)
print("\nColumn names:", list(df.columns))
print("\nData types:")
print(df.dtypes)
print("\nSummary statistics:")
print(df.describe())

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())
