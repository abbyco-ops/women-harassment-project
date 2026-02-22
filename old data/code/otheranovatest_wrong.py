import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Load Data
# --------------------------------------------------
df = pd.read_csv("data/processed/violence_data_cleaned.csv")

# Keep only needed columns
df = df[['Value', 'Country', 'Demographics Question',
         'Demographics Response', 'Survey Year']].dropna()

# --------------------------------------------------
# 2. Split by Country (prevents leakage)
# --------------------------------------------------
countries = df['Country'].unique()

train_countries, test_countries = train_test_split(
    countries, test_size=0.2, random_state=42
)

train_df = df[df['Country'].isin(train_countries)]
test_df = df[df['Country'].isin(test_countries)]

# --------------------------------------------------
# 3. One-Hot Encode Categorical Variables
# --------------------------------------------------
train_df = pd.get_dummies(train_df, drop_first=True)
test_df = pd.get_dummies(test_df, drop_first=True)

# Align columns (important!)
train_df, test_df = train_df.align(test_df, join='left', axis=1, fill_value=0)

# Separate X and y
X_train = train_df.drop(columns=['Value'])
y_train = train_df['Value']

X_test = test_df.drop(columns=['Value'])
y_test = test_df['Value']

print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)

# --------------------------------------------------
# 4. Standardize Features
# --------------------------------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------------
# 5. Fit Ridge Regression
# --------------------------------------------------
model = Ridge(alpha=1.0)
model.fit(X_train_scaled, y_train)

# --------------------------------------------------
# 6. Evaluate Model
# --------------------------------------------------
y_pred = model.predict(X_test_scaled)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

print("\nModel Evaluation:")
print("R²:", round(r2, 3))
print("MAE:", round(mae, 3))
print("RMSE:", round(rmse, 3))

# --------------------------------------------------
# 7. Residual Plot
# --------------------------------------------------
residuals = y_test - y_pred

plt.figure(figsize=(6,4))
plt.hist(residuals, bins=20)
plt.title("Residual Distribution")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.show()

# --------------------------------------------------
# 8. Feature Importance
# --------------------------------------------------
coefficients = pd.DataFrame({
    "Feature": X_train.columns,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", ascending=False)

print("\nTop 10 Positive Associations:")
print(coefficients.head(10))

print("\nTop 10 Negative Associations:")
print(coefficients.tail(10))