import pandas as pd 
import matplotlib.pyplot as plt

df = pd.read_csv("data/processed/violence_data_cleaned.csv")


#View overall distribution
df['Value'].hist(bins=20)
plt.show()


#Compare demographic categories
edu_df = df[df['Demographics Question'] == 'Education']
print(edu_df.groupby('Demographics Response')['Value'].mean().sort_values())


#Compare age groups
age_df = df[df['Demographics Question'] == 'Age']
print(age_df.groupby('Demographics Response')['Value'].mean())


#Country variation
print(df.groupby('Country')['Value'].mean().sort_values(ascending=False))

#Visualizae country variation
df.groupby('Country')['Value'].mean().sort_values().plot(kind='barh', figsize=(8,6))
plt.xlabel("Average Value")
plt.title("Average Value by Country")
plt.show()
