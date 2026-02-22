import pandas as pd

# Load the CSV
df = pd.read_csv("violence_data.csv")  # because it’s in the project root
df.head()   # see the first few rows
df.info()   # check column names, data types, missing values

print(df.columns)
# Check missing values
df.isnull().sum()


df = df.dropna(subset=['Value'])

import os

# Create folder if it doesn't exist
os.makedirs("data/processed", exist_ok=True)

# Save cleaned CSV
df.to_csv("data/processed/violence_data_cleaned.csv", index=False)

print("Data cleaned and saved!")

