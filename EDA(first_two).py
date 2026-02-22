import os
import pandas as pd
import matplotlib.pyplot as plt

# Current working directory
cwd = os.getcwd()
print("Current working directory:", cwd)

# List of possible file locations
possible_paths = [
    "violence_data_cleaned.csv",      # current directory
    os.path.join("data/processed", "violence_data_cleaned.csv")  # data folder
]

# Find the first existing file
for path in possible_paths:
    if os.path.exists(path):
        csv_path = path
        break
else:
    raise FileNotFoundError(
        "Could not find 'violence_data_cleaned.csv' in the project folder or 'data/' subfolder."
    )

# Read the CSV
df = pd.read_csv(csv_path)
print(f"Loaded CSV from: {csv_path}")
print("First 5 rows:")
print(df.head())

df['Value'].hist(bins=20)
edu_df = df[df['Demographics Question'] == 'Education']
edu_df.groupby('Demographics Response')['Value'].mean().sort_values()


