# Boston Housing Analysis - Complete Code

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load Dataset
boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['PRICE'] = boston.target

# 2. Initial Overview
print("First 5 rows:\n", df.head())
print("\nData Info:\n")
print(df.info())
print("\nDescriptive Statistics:\n")
print(df.describe())

# 3. Correlation Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# 4. Price Distribution
plt.figure(figsize=(6, 4))
sns.histplot(df['PRICE'], bins=30, kde=True)
plt.title("Housing Price Distribution")
plt.xlabel("Price ($1000s)")
plt.ylabel("Frequency")
plt.show()

# 5. Pairplot for Important Features
sns.pairplot(df[['PRICE', 'RM', 'LSTAT', 'PTRATIO']])
plt.show()

# 6. Split Features and Target
X = df.drop("PRICE", axis=1)
y = df["PRICE"]

# 7. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

print("\n--- Linear Regression Evaluation ---")
print("R² Score:", r2_score(y_test, lr_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, lr_pred)))

# 9. Random Forest Regressor Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print("\n--- Random Forest Evaluation ---")
print("R² Score:", r2_score(y_test, rf_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, rf_pred)))

# 10. Feature Importances (Random Forest)
feature_imp = pd.Series(rf_model.feature_importances_, index=X.columns)
plt.figure(figsize=(10, 6))
feature_imp.sort_values(ascending=True).plot(kind='barh')
plt.title("Feature Importances from Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()

# 11. Actual vs Predicted - Random Forest
plt.figure(figsize=(6, 6))
plt.scatter(y_test, rf_pred, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices - Random Forest")
plt.grid(True)
plt.show()
