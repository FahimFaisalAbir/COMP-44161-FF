__Opportunity No. 44161__

# Automated Property Valuation Model 

### Overview
An **Automated Valuation Model (AVM)** was developed using **supervised machine learning** to predict the **assessed market value of residential and commercial properties**.  
The model provides a **data-driven, consistent, and transparent framework** for estimating property values based on a combination of physical, locational, and categorical features.

---

### Objectives
- To automate the property valuation process using statistical and machine learning methods.  
- To improve the **accuracy**, **efficiency**, and **fairness** of property assessments.  
- To provide an adaptable modeling pipeline for use by analysts, municipalities, and research teams.

---

### Data Preparation
- Data were preprocessed by handling missing values, scaling numeric features, and encoding categorical variables.  
- Both **Label Encoding** and **One-Hot Encoding** options were implemented for flexibility.  
- Mode imputation was applied to ensure consistent treatment of missing information.

---

### Machine Learning Approach
- Multiple supervised learning algorithms were implemented, including:
  - Linear Regression (Ridge, Lasso, ElasticNet)
  - Decision Tree Regressor
  - Random Forest Regressor
  - XGBoost Regressor
- Each model was evaluated using **R²** and **RMSE** metrics with optional **Grid Search** and **Randomized Search** for hyperparameter optimization.

---

### Results
- The **XGBoost model** achieved the highest performance, demonstrating strong predictive accuracy and robustness across diverse property types.  
- The model effectively captured non-linear relationships between property characteristics and assessed values.

---

### Impact
- Enables **automated, data-driven valuation** of properties.  
- Reduces manual workload for assessors and analysts.  
- Improves **consistency** and **fairness** in property value estimation across regions.  
- Provides a foundation for future integration into municipal assessment or decision-support systems.

---

### Repository Structure
## Environment

You will need to install several packages to run the existing code

# 🏠 Automated Property Valuation Model (AVM)

This project predicts the assessed market value of properties using supervised machine learning.

---

## 📊 Model Overview

![Feature importance](plot/xgb_feature_importance.png)

---

## 🧮 Model Comparison Results

| Model | R² | RMSE | Relative to best |
|-------|----|------|--------------|
| base R LM model	| 0.003| 0 |
| OLS LM model | 0.137 | 50,200 | 14 |
| Linear Regression | 0.783 | 83 | |
| Random Forest | 0.935 | 27394.84 | 100 |
| XGBoost | 0.934 | 27375.93 | 99|

The **XGBoost model** achieved the highest accuracy, effectively capturing complex relationships between property features and assessed value.


```python
pip install pandas numpy scipy matplotlib seaborn scikit-learn matplotlib xgboost joblib typing-extensions notebook ipython tqdm

