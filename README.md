# 🏡 House Price Prediction Project

This project aims to predict house prices using the **House Prices - Advanced Regression Techniques** dataset from Kaggle. We performed comprehensive Exploratory Data Analysis (EDA), thorough preprocessing, and applied several regression models to determine the best-performing model.

## 🚀 Project Objectives

1. **Exploratory Data Analysis (EDA):** Understand the distribution of house prices (`SalePrice`) and its relationship with other features.
2. **Data Preprocessing:** Handle missing values, convert data types, encode categorical variables (both nominal and ordinal), and standardize data.
3. **Model Training:** Evaluate the performance of Linear Regression models (Lasso, ElasticNet) and Tree-based models (Random Forest, XGBoost).
4. **Evaluation:** Compare models based on the **Root Mean Squared Error (RMSE)** on the log-transformed sale price and the original sale price.

## 🛠️ Technology & Libraries

* **Language:** Python
* **Data Analysis Libraries:** `pandas`, `numpy`, `scipy`
* **Visualization:** `matplotlib`, `seaborn`
* **Machine Learning:** `scikit-learn` (Linear Regression, Tree Models, Preprocessing), `XGBoost`
* **Tool:** Jupyter Notebook

## ⚙️ Key Execution Steps

The project was executed through the following steps, detailed in the `house-price-prediction.ipynb` file:

### 1. Data Loading and Exploration

* Load the training dataset (`train.csv`) and check its dimensions and general information (`info()`).
* Convert the `Id` and `MSSubClass` columns to the `object` (categorical) type for appropriate handling.

### 2. Missing Value Imputation

* Apply synchronized imputation methods:
    * **Numeric Variables:** Fill missing values with the **Median** from the training set.
    * **Categorical Variables:** Fill missing values with the constant **"None"** (as missing values often mean "absence" of the amenity, e.g., 'No Pool', 'No Garage').

### 3. Feature Transformation

* **Logarithmic Transformation (Target Transformation):** Apply **$\text{log}(1+x)$** to the target variable `SalePrice` to address its **skewness**, which helps improve the performance of linear regression models.
* **Ordinal Encoding:** Apply custom mapping (from 0 to 8) to 19 categorical features with inherent order (e.g., `ExterQual`, `BsmtQual`, `FireplaceQu`, `LotShape`, etc.) to convert them into numerical values that preserve their rank.
* **Nominal Label Encoding:** Apply **`LabelEncoder`** to all remaining categorical features (nominal) after ordinal encoding.
* **Scaling:** Apply **`MinMaxScaler`** to numerical features (Ordinal and Continuous) to normalize the value range.

### 4. Data Splitting and Model Training

* The training data is split into **Training (80%)** and **Validation (20%)** sets.
* Models are trained on the log-transformed sale price (`log(SalePrice)`):
    * **Linear Models:** ElasticNet and Lasso (using **`GridSearchCV`** for optimal hyperparameter search).
    * **Tree-based Models:** RandomForestRegressor and XGBRegressor (with pre-selected hyperparameters).

## 📊 Performance Results

Models are compared based on their performance on the Validation set:

| Model | RMSE (Log) | R2 Score | RMSE (Original $) |
| :--- | :--- | :--- | :--- |
| **XGBoost** | 0.1306 | 0.9086 | 27,268.57 |
| **RandomForest** | 0.1477 | 0.8831 | 30,088.10 |
| **Lasso\_Reg** | 0.1496 | 0.8800 | 28,985.76 |
| **ElasticNet** | 0.1503 | 0.8790 | 29,298.41 |

### Best Model

The **XGBoost** model showed the best performance with the lowest **RMSE (Log) of 0.1306**.

---

### Results on Test Set (If True Prices Exist)

The best model (**XGBoost**) was retrained on the full training set and used for prediction on the test set.

| Metric | Value |
| :--- | :--- |
| **RMSE** (Test set) | $68,001.71 |
| **MAE** (Test set) | $51,464.60 |
| **MAPE** (Test set) | 28.66% |

## 💡 Next Steps for Development

* Implement advanced **Feature Engineering** techniques (e.g., creating interaction features).
* Experiment with other models (e.g., SVR, LightGBM).
* Apply **Ensemble/Stacking** techniques to combine the best models.
* Deeper hyperparameter optimization for tree-based models.
