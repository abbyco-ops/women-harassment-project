import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset
df = pd.read_csv("data/processed/violence_data_cleaned.csv")

# 2. Choose the violence question we want to model
target_question = df['Question'].unique()[0]  # pick the first question
df_target = df[df['Question'] == target_question].copy()
print("Targeting question:", target_question)
print("Filtered dataset shape:", df_target.shape)

# 3. Create binary target
df_target['Violence_Occurred'] = df_target['Value'].apply(lambda x: 1 if x > 0 else 0)


# Create country-level target BEFORE pivot
country_target = (
    df_target
    .groupby(['Country', 'Survey Year'])['Violence_Occurred']
    .max()
    .reset_index()
)

country_target.rename(columns={'Violence_Occurred': 'Country_Violence'}, inplace=True)


# 4. Pivot demographics to wide format
df_wide = df_target.pivot_table(
    index=['Country', 'Survey Year'],  # identifiers
    columns='Demographics Response',
    values='Violence_Occurred',
    fill_value=0
).reset_index()


print("Pivoted dataset shape:", df_wide.shape)


# Merge with wide dataframe
df_wide = df_wide.merge(country_target, on=['Country', 'Survey Year'])

X = df_wide.drop(columns=['Country', 'Survey Year', 'Country_Violence'])
y = df_wide['Country_Violence']



# 5. Define features and target
X = df_wide.drop(columns=['Country', 'Survey Year'])
y = df_wide.drop(columns=['Country', 'Survey Year']).any(axis=1).astype(int)  # same shape as X

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 8. Evaluate
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# 9. Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
}).sort_values(by='Coefficient', ascending=False)

print("\nTop 10 features increasing likelihood of violence reports:")
print(feature_importance.head(10))


#What This Script Does:
#Loads your Kaggle dataset

#Creates a binary target (Violence_Occurred)

#Pivots demographics into wide format so each response is a feature

#Selects a single violence question as the target

#Splits the data into train/test sets

#Trains a Logistic Regression model

#Evaluates performance with a classification report and confusion matrix

#Shows top predictive demographic groups