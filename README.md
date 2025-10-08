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
- Median / Mode imputation was applied to ensure consistent treatment of missing information.

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
- Enables **automated, fair, data-driven valuation** of properties.  
- Reduces manual workload for assessors and analysts and help them to detect important features for operational decision making.  
- Improves **consistency** and **fairness** in property value estimation across regions.  
- Provides a foundation for future integration into municipal assessment or decision-support systems.

---
# Automated Property Valuation Model (AVM)

This project predicts the assessed market value of properties using supervised machine learning.

### Repository Structure
- DataUtils - Handle data loading, pre-processing , outlier detection, feature type detection and ETL pipeline
- Ensemble Model - Ensemble of tree based models
- PropertyEvalModeling V3.ipynb - Perform exploratory data analysis, pre-processing and train suprevised models for property assesment, justification of modelling approach.

## Environment

You will need to install several packages to run the existing code



---

## 📊 Model Overview

![Feature importance](plot/xgb_feature_importance.png)
![Ensemble (Decision tree, Random Forest, XGB average) Feature importance](plot/ensemble_feature_importance.png)

---

## Model Comparison Results

| Model | R² | RMSE | Relative to baseline | Relative to Best|
|-------|----|------|--------------|--------------|
| base R LM model	| 0.0034 | 2767000| 0 % | 3.21 % |
| OLS LM model | 0.137 | 3929 | 356.6 % | 14.66 %|
| Linear Regression | 0.784 | 22958 | 2,513 % | 83.86 %|
| Random Forest | 0.935 | 27394.84 | 2,948 % | 97.83 % |
| XGBoost | 0.934 | 27375.93 | 3,016 % | 100 % |

The **XGBoost model** achieved the highest accuracy, effectively capturing complex relationships between property features and assessed value.


```python
pip install pandas numpy scipy matplotlib seaborn scikit-learn matplotlib xgboost joblib typing-extensions notebook ipython tqdm statsmodels

