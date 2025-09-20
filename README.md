# Boston Housing Analysis - README

## Project Overview

This project is a comprehensive, end-to-end analysis of the Boston Housing dataset. It covers:

* Data loading using modern `fetch_openml` method.
* Exploratory Data Analysis (EDA) including correlations, distributions, and pairplots.
* Modeling using Linear Regression, Ridge Regression, and Random Forest.
* Hyperparameter tuning for Random Forest using GridSearchCV.
* Model evaluation using R², RMSE, and MAE.
* Diagnostic plots including residuals and Actual vs Predicted scatter.
* Feature importance using permutation importance.
* Saving trained models for future use.

## Requirements

Python packages required:

```
scikit-learn
pandas
numpy
matplotlib
seaborn
joblib
```

Install via:

```
pip install scikit-learn pandas numpy matplotlib seaborn joblib
```

## File Structure

* `boston_housing_analysis.py` : main Python script containing the full workflow.
* `boston_rf_model.joblib` : saved Random Forest model after tuning.
* `README.md` : this file.

## Usage

1. Clone or download the project.
2. Install the required packages.
3. Run the Python script:

```bash
python boston_housing_analysis.py
```

4. Follow printed outputs and generated plots for insights.

## Key Steps / Workflow

1. **Load Dataset**: Load Boston dataset from OpenML and rename target column as `PRICE`.
2. **EDA**: Check data types, missing values, descriptive statistics, correlations, distributions, and pairplots.
3. **Data Preparation**: Split dataset into train/test sets.
4. **Model Pipelines**: Set up pipelines for Linear Regression, Ridge Regression, and Random Forest.
5. **Baseline Evaluation**: Cross-validation for quick R² estimates.
6. **Hyperparameter Tuning**: GridSearchCV for Random Forest to improve performance.
7. **Final Model Training**: Train best Random Forest and Ridge models on full training data.
8. **Evaluation**: Evaluate models on the test set using R², RMSE, MAE.
9. **Diagnostics**: Residual plots, Actual vs Predicted plots.
10. **Feature Importance**: Permutation importance for robust feature ranking.
11. **Model Saving**: Save trained model using `joblib`.

## Notes & Next Steps

* For better performance, consider XGBoost or LightGBM with hyperparameter tuning.
* Nested cross-validation can provide more unbiased model estimates.
* For interpretability, integrate SHAP explanations.
* For deployment, wrap models into API endpoints using Flask/FastAPI.
* Explore feature engineering (interactions, polynomial features) carefully.

## Author

* Project developed by Rucha

* [scikit-learn documentation](https://scikit-learn.org/stable/)
* [Boston Housing Dataset - OpenML](https://www.openml.org/d/531)
