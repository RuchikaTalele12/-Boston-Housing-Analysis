*Boston Housing Price Analysis*


## 📌 Features

- 📊 Exploratory Data Analysis (EDA)
- 🔍 Correlation heatmaps and feature relationships
- 📈 Price distribution and important feature visualizations
- 🧠 Predictive modeling with:
  - Linear Regression
  - Random Forest Regressor
- ✅ Model evaluation using R² Score and RMSE
- 📉 Feature importance analysis
- 🖼️ Actual vs Predicted price plots

---

## 📂 Dataset

- Dataset Name: **Boston Housing**
- Source: `sklearn.datasets.load_boston()` *(deprecated in latest versions)*
- Alternative: `fetch_openml(name="boston", version=1)`

Each row describes a Boston suburb and includes 13 features:

- CRIM — Crime rate
- ZN — Proportion of residential land zoned for lots
- INDUS — Proportion of non-retail business acres per town
- CHAS — Charles River dummy variable
- NOX — Nitric oxide concentration
- RM — Average number of rooms per dwelling
- AGE — Proportion of owner-occupied units built before 1940
- DIS — Weighted distance to employment centers
- RAD — Accessibility to radial highways
- TAX — Property tax rate
- PTRATIO — Pupil-teacher ratio
- B — Proportion of people of African American descent
- LSTAT — % lower status of the population

---

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/boston-housing-analysis.git
cd boston-housing-analysis
pip install -r requirements.txt
````

---

## 🚀 How to Run

```bash
python boston_housing_analysis.py
```

Or run the Jupyter Notebook version:

```bash
jupyter notebook Boston_Housing_Analysis.ipynb
```

---

## 📈 Example Visualizations

* Heatmap of correlations
* Price distribution
* Pairplot of key features
* Feature importance from Random Forest
* Actual vs Predicted price scatter plot

---

## 🧪 Model Performance

| Model             | R² Score | RMSE  |
| ----------------- | -------- | ----- |
| Linear Regression | \~0.74   | \~4.9 |
| Random Forest     | \~0.87   | \~3.2 |

---

## 📌 Dependencies

* `pandas`
* `numpy`
* `matplotlib`
* `seaborn`
* `scikit-learn`

---

## 📜 License

This project is licensed under the MIT License.

--
