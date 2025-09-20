# Boston Housing - improved end-to-end project
# Requirements: scikit-learn, pandas, numpy, matplotlib, seaborn, joblib
# pip install scikit-learn pandas matplotlib seaborn joblib

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
import joblib

# ---------------------------
# 1. Load dataset (modern approach)
# ---------------------------
boston = fetch_openml(name="Boston", version=1, as_frame=True)  # uses openml copy
df = boston.frame.copy()
df.rename(columns={boston.target.name: "PRICE"}, inplace=True)  # ensure target column is PRICE

# Quick checks
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nFirst 5 rows:")
display(df.head())

# ---------------------------
# 2. Basic info & summary
# ---------------------------
print("\nData types:")
print(df.dtypes)
print("\nMissing values per column:")
print(df.isna().sum())

print("\nDescriptive statistics (rounded):")
display(df.describe().round(3))

# ---------------------------
# 3. Exploratory Data Analysis (concise & meaningful)
# ---------------------------
# Correlation matrix (show only strong correlations)
plt.figure(figsize=(12, 10))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True, linewidths=.5)
plt.title("Correlation Heatmap (All features)")
plt.show()

# Price distribution + boxplot
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
sns.histplot(df["PRICE"], bins=30, kde=True, ax=axes[0])
axes[0].set_title("Housing Price Distribution (in $1000s)")
axes[0].set_xlabel("PRICE")

sns.boxplot(x=df["PRICE"], ax=axes[1])
axes[1].set_title("Price Boxplot (detect outliers)")
plt.show()

# Pairwise scatter for the most correlated features with PRICE
top_features = corr["PRICE"].abs().sort_values(ascending=False).index[1:6].tolist()
print("Top correlated features with PRICE:", top_features)
sns.pairplot(df[["PRICE"] + top_features], corner=True, diag_kind="kde", plot_kws={"alpha":0.6, "s":30})
plt.suptitle("Pairplot: PRICE vs Top features", y=1.02)
plt.show()

# ---------------------------
# 4. Prepare data for modeling
# ---------------------------
X = df.drop(columns="PRICE")
y = df["PRICE"]

# train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

# ---------------------------
# 5. Modeling pipelines
# ---------------------------
# Simple baseline: Linear Regression (with scaling)
lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LinearRegression())
])

# Regularized linear model (Ridge) for more robust baseline
ridge_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge())
])

# Random Forest (no scaling required but we'll use pipeline for interface)
rf_pipeline = Pipeline([
    ("scaler", StandardScaler()),   # scaling doesn't change RF but keeps interfaces consistent
    ("rf", RandomForestRegressor(random_state=42))
])

# ---------------------------
# 6. Quick cross-validated baselines
# ---------------------------
def cv_report(pipe, X, y, cv=5):
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="r2")
    print(f"CV R² mean: {scores.mean():.4f} ± {scores.std():.4f}")

print("\nBaseline CV results (5-fold):")
print("Linear Regression:")
cv_report(lr_pipeline, X_train, y_train)
print("Ridge:")
cv_report(ridge_pipeline, X_train, y_train)
print("Random Forest (default):")
cv_report(rf_pipeline, X_train, y_train)

# ---------------------------
# 7. Hyperparameter tuning for Random Forest (GridSearch)
# ---------------------------
param_grid = {
    "rf__n_estimators": [100, 200],
    "rf__max_depth": [None, 8, 12],
    "rf__min_samples_split": [2, 5],
    "rf__min_samples_leaf": [1, 2],
}

gs_rf = GridSearchCV(rf_pipeline, param_grid, cv=5, scoring="r2", n_jobs=-1, verbose=1)
gs_rf.fit(X_train, y_train)

print("\nBest RF params:", gs_rf.best_params_)
print("Best RF CV R²:", gs_rf.best_score_)

# ---------------------------
# 8. Fit final models on training set
# ---------------------------
# Fit best RF, and a tuned Ridge (simple small grid)
best_rf = gs_rf.best_estimator_
best_rf.fit(X_train, y_train)

# Tune Ridge alpha briefly
from sklearn.model_selection import GridSearchCV
ridge_params = {"ridge__alpha": [0.1, 1.0, 10.0, 50.0]}
gs_ridge = GridSearchCV(ridge_pipeline, ridge_params, cv=5, scoring="r2", n_jobs=-1)
gs_ridge.fit(X_train, y_train)
best_ridge = gs_ridge.best_estimator_
print("\nBest Ridge params and CV R²:", gs_ridge.best_params_, gs_ridge.best_score_)

# ---------------------------
# 9. Evaluate on test set
# ---------------------------
def evaluate(model, X_test, y_test, name="Model"):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    print(f"\n--- {name} Evaluation on TEST ---")
    print(f"R²: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    return y_pred

lr_pipeline.fit(X_train, y_train)
y_lr = evaluate(lr_pipeline, X_test, y_test, "Linear Regression (OLS)")
y_ridge = evaluate(best_ridge, X_test, y_test, "Ridge Regression")
y_rf = evaluate(best_rf, X_test, y_test, "Random Forest (Tuned)")

# ---------------------------
# 10. Residuals & diagnostic plots (for best model)
# ---------------------------
def residual_plots(y_true, y_pred, title_suffix=""):
    resid = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12,4))
    sns.scatterplot(x=y_pred, y=resid, ax=axes[0], alpha=0.7)
    axes[0].axhline(0, color='r', ls='--')
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residuals vs Predicted " + title_suffix)
    
    sns.histplot(resid, kde=True, ax=axes[1])
    axes[1].set_title("Residual Distribution " + title_suffix)
    plt.show()

residual_plots(y_test, y_rf, "(Random Forest)")

# Actual vs Predicted scatter
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_rf, alpha=0.7)
minv, maxv = min(y_test.min(), y_rf.min()), max(y_test.max(), y_rf.max())
plt.plot([minv, maxv], [minv, maxv], 'r--')
plt.xlabel("Actual PRICE")
plt.ylabel("Predicted PRICE")
plt.title("Actual vs Predicted (Random Forest)")
plt.grid(True)
plt.show()

# ---------------------------
# 11. Feature importances (permutation importance for reliability)
# ---------------------------
print("\nCalculating permutation importances (this may take a little while)...")
perm = permutation_importance(best_rf, X_test, y_test, n_repeats=30, random_state=42, n_jobs=-1)
perm_idx = perm.importances_mean.argsort()[::-1]
feat_names = X.columns

imp_df = pd.DataFrame({
    "feature": feat_names,
    "importance_mean": perm.importances_mean,
    "importance_std": perm.importances_std
}).sort_values(by="importance_mean", ascending=False)

display(imp_df.head(15))

plt.figure(figsize=(8,6))
sns.barplot(x="importance_mean", y="feature", data=imp_df.head(12), orient="h")
plt.title("Top 12 Features by Permutation Importance (Test set)")
plt.xlabel("Mean decrease in R² (importance)")
plt.ylabel("Feature")
plt.show()

# ---------------------------
# 12. Save final model & artifacts
# ---------------------------
model_filename = "boston_rf_model.joblib"
joblib.dump({"model": best_rf, "feature_names": list(X.columns)}, model_filename)
print(f"Saved tuned Random Forest pipeline to: {model_filename}")

# ---------------------------
# 13. Quick tips & next steps (printed)
# ---------------------------
print("\nQuick tips / next steps:")
print("- Consider using advanced models (XGBoost / LightGBM) and compare.")
print("- Use nested CV if you want an unbiased estimate of tuning + evaluation.")
print("- Explore interactions / polynomial features (careful with overfitting).")
print("- If you need explainability, consider SHAP for per-sample contributions.")
print("- If productionizing, containerize and provide API endpoints (Flask/FastAPI).")
