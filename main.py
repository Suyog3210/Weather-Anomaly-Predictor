import pandas as pd
from data_loader import load_rainfall_data, load_disaster_data
from preprocess import merge_datasets

# File paths
rainfall_file = "data/RF_AI_1901-2021.csv"
disaster_file = "data/Natural_Events(INDIA).xlsx"

# Load datasets
rainfall = load_rainfall_data(rainfall_file)
disaster = load_disaster_data(disaster_file)

# Merge datasets
final_data = merge_datasets(rainfall, disaster)

# Save merged dataset
final_data.to_csv("data/merged_rainfall_disaster.csv", index=False)

print("✅ Data processing completed. Merged dataset saved.")
print(final_data.head())
