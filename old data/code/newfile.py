import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 1. Load the dataset
# -------------------------------
df = pd.read_csv("violence_data.csv")

# Inspect first few rows
print("First 5 rows:")
print(df.head(), "\n")

# -------------------------------
# 2. Summarize violence by demographic
# -------------------------------
demographic_summary = df.groupby(
    ['Demographics Question', 'Demographics Response']
)['Value'].sum().sort_values(ascending=False)

print("Top 20 demographic groups reporting violence:")
print(demographic_summary.head(20), "\n")

# -------------------------------
# 3. Prepare data for plotting
# -------------------------------
demo_plot = demographic_summary.reset_index()

# For clarity, only show top 20 reporting groups in the plot
top_20 = demo_plot.head(20)

# -------------------------------
# 4. Visualization
# -------------------------------
plt.figure(figsize=(12,8))
sns.barplot(
    data=top_20,
    x='Value',
    y='Demographics Response',
    hue='Demographics Question',
    dodge=False
)
plt.title("Top 20 Demographic Groups Reporting Violence")
plt.xlabel("Number of Reports / Surveyed Value")
plt.ylabel("Demographic Response")
plt.legend(title="Demographic Question", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# -------------------------------
# 5. Optional: Save summary to CSV
# -------------------------------
demographic_summary.to_csv("demographic_violence_summary.csv")
print("Summary saved to 'demographic_violence_summary.csv'")




import pandas as pd

# Load data
df = pd.read_csv("violence_data.csv")

# Create a binary target: 1 if any violence reported, else 0
df['Violence_Occurred'] = df['Value'].apply(lambda x: 1 if x > 0 else 0)

# Pivot the demographics to wide format
df_wide = df.pivot_table(
    index=['Country', 'Survey Year', 'Question'],  # keep these as identifiers
    columns='Demographics Response',
    values='Violence_Occurred',
    fill_value=0
).reset_index()

print(df_wide.head())