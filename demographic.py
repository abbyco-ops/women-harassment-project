import pandas as pd 
df = pd.read_csv("data/processed/violence_data_cleaned.csv")
df.head()
df.info()

# Filter only rows relevant to violence
# (Optional: depending on the dataset, all rows may already be violence reports)

# Group by demographic type and response, sum the reported violence
demographic_summary = df.groupby(
    ['Demographics Question', 'Demographics Response']
)['Value'].sum().sort_values(ascending=False)

print(demographic_summary)



import matplotlib.pyplot as plt
import seaborn as sns

# Reset index for plotting
demo_plot = demographic_summary.reset_index()

plt.figure(figsize=(12,6))
sns.barplot(
    data=demo_plot.head(20),  # top 20 reporting groups
    x='Value',
    y='Demographics Response',
    hue='Demographics Question'
)
plt.title("Top 20 Demographic Groups Reporting Violence")
plt.xlabel("Number of Reports")
plt.ylabel("Demographic Group")
plt.show()