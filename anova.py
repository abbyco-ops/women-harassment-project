import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# -----------------------------
# 1. Load Data
# -----------------------------
df = pd.read_csv("violence_data.csv")

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Remove rows with missing Value
df = df.dropna(subset=['Value'])

# -----------------------------
# 2. Define demographic questions to test
# -----------------------------
demographic_questions = df['Demographics Question'].unique()

# -----------------------------
# 3. Run ANOVA for each demographic question
# -----------------------------
anova_results = []

for question in demographic_questions:
    # Filter dataframe
    subset = df[df['Demographics Question'] == question].copy()
    
    # Strip whitespace from demographic responses
    subset['Demographics Response'] = subset['Demographics Response'].str.strip()
    
    # Rename column to simplify formula
    subset = subset.rename(columns={"Demographics Response": "Response"})
    
    # Only run ANOVA if more than 1 unique response exists
    if subset['Response'].nunique() > 1:
        model = ols('Value ~ C(Response)', data=subset).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        F = anova_table['F'][0]
        p = anova_table['PR(>F)'][0]
        anova_results.append({
            "Demographic Question": question,
            "F-statistic": F,
            "p-value": p
        })

# -----------------------------
# 4. Print Results
# -----------------------------
anova_df = pd.DataFrame(anova_results)
print("ANOVA Results by Demographic Question:")
print(anova_df.sort_values('p-value'))