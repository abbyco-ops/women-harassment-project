import pandas as pd 
import matplotlib.pyplot as plt

df = pd.read_csv("data/processed/violence_data_cleaned.csv")
#df['Value'].hist(bins=20)
#plt.show()




import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load and Clean Data
# -----------------------------
df = pd.read_csv("violence_data.csv")

# Remove missing values in Value
df = df.dropna(subset=['Value'])

# Convert Survey Year to numeric
df['Survey Year'] = pd.to_datetime(df['Survey Year'])
df['Year'] = df['Survey Year'].dt.year

print("Cleaned dataset shape:", df.shape)

# -----------------------------
# 2. Prepare Model Data
# -----------------------------
# We include:
# - Country (controls for country differences)
# - Demographics Question (type of grouping)
# - Demographics Response (actual group)
# - Year

df_model = df[['Value', 'Country', 'Demographics Question',
               'Demographics Response', 'Year']]

# Convert categorical variables into dummy variables
df_model = pd.get_dummies(df_model, drop_first=True)

# Separate features and target
X = df_model.drop(columns=['Value'])
y = df_model['Value']

print("Feature matrix shape:", X.shape)

# -----------------------------
# 3. Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 4. Fit Linear Regression
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# 5. Evaluate Model
# -----------------------------
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

print("\nModel Evaluation:")
print("R²:", r2)
print("MAE:", mae)
print("RMSE:", rmse)

# -----------------------------
# 6. Check Residual Distribution
# -----------------------------
residuals = y_test - y_pred

plt.hist(residuals, bins=20)
plt.title("Residual Distribution")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.show()

# -----------------------------
# 7. Extract Coefficients
# -----------------------------
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", ascending=False)

print("\nTop 10 Positive Associations:")
print(coefficients.head(10))

print("\nTop 10 Negative Associations:")
print(coefficients.tail(10))

