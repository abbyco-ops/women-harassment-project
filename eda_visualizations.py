import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. Load Data
# -----------------------------
df = pd.read_csv("data/processed/violence_data_cleaned.csv")

# -----------------------------
# 2. Overall Distribution
# -----------------------------
plt.figure(figsize=(6,4))
df['Value'].hist(bins=20, color='skyblue', edgecolor='black')
plt.title("Distribution of Violence Values")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

# -----------------------------
# 3. Education Demographics
# -----------------------------
edu_df = df[df['Demographics Question'] == 'Education']
edu_table = edu_df.groupby('Demographics Response')['Value'].mean().sort_values().reset_index()

# Table visualization
fig, ax = plt.subplots(figsize=(6,2))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=edu_table.values, colLabels=edu_table.columns, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)
plt.title("Average Violence Value by Education Group")
plt.show()

# -----------------------------
# 4. Age Demographics
# -----------------------------
age_df = df[df['Demographics Question'] == 'Age']
age_table = age_df.groupby('Demographics Response')['Value'].mean().sort_values().reset_index()

# Table visualization
fig, ax = plt.subplots(figsize=(6,2))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=age_table.values, colLabels=age_table.columns, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)
plt.title("Average Violence Value by Age Group")
plt.show()

# -----------------------------
# 5. Country Variation
# -----------------------------
country_table = df.groupby('Country')['Value'].mean().sort_values(ascending=False).reset_index()

# Table visualization
fig, ax = plt.subplots(figsize=(8,4))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=country_table.values, colLabels=country_table.columns, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)
plt.title("Average Violence Value by Country")
plt.show()

# Optional: Add heatmap style for country variation
plt.figure(figsize=(8,6))
sns.heatmap(country_table.set_index('Country').T, annot=True, cmap="Reds", cbar=True)
plt.title("Heatmap of Average Violence Value by Country")
plt.show()