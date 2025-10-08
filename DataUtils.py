# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 21:49:54 2025

@author: Fahim
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from itertools import islice
import pandas as pd
from Utils import categorical_columns,categorical_columns_summary
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error

from scipy import stats
import pandas as pd
import numpy as np

import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr
from xgboost import XGBRegressor
from typing import Tuple, List, Dict, Any, Optional



def load_data(file: str) -> pd.DataFrame:
    """
    Load Data
    
    Parameters
    ----------
    file : path string
    
    Returns
    -------
    pd.DataFrame
    """
    # Read CSV
    Data = pd.read_csv(file)

    # Convert TotalLivingArea to numeric, replace NaN with 0
    Data['Total Living Area_Num'] = pd.to_numeric(Data['Total Living Area'], errors='coerce').fillna(0)

    # Convert AssessedLandArea to numeric, replace NaN with 0
    Data['Assessed Land Area_Num'] = pd.to_numeric(Data['Assessed Land Area'], errors='coerce').fillna(0)

    # Parse numbers from Total.AssessedValue (removes $ signs, commas, etc.), replace NaN with 0
    Data['Total Assessed Value_Num'] = pd.to_numeric(Data['Total Assessed Value'].replace('[^0-9.]', '', regex=True),
                                                    errors='coerce').fillna(0)

    return Data


# Identify categorical columns
def categorical_columns(df, describe=True):
    categorical_cols = [col for col in df.columns if df[col].dtype in ['object', 'category', 'bool']]
    print(f"Categorical columns: {categorical_cols}")
    if(describe==True):
        # Describe all columns, including categorical
        
        print("\nDescriptive statistics for all columns:")
        print(df.describe(include='all'))

def categorical_columns_summary(df):
    # Select only object or categorical dtype columns
    cat_cols = df.select_dtypes(include=['object','category']).columns
    
    for col in cat_cols:
        if col in ['ResidencePostalCode']:
            pass
        else:
            print("==============================================Column name: ",col,"==================================================================")
            # Absolute counts
            unique_values=df[col].nunique()
            counts = df[col].value_counts()
            
            # Relative frequencies (proportions)
            props =df[col].value_counts(normalize=True)
            print("---------Unique values-------",unique_values)
            print("---------Counts---------",counts)
            print("-------Proportion-------",props)
            



def missing_value_summary(df: pd.DataFrame):
    """
    Display the percentage of missing values in data column.
    
    Parameters
    ----------
        Input DataFrame
    
    Returns
    -------
        Table of columns with count and percentage of missing values
    """
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100

    summary = pd.DataFrame({
        "MissingCount": missing_count,
        "MissingPercent": missing_percent.round(2)
    }).sort_values(by="MissingPercent", ascending=False)
    list=summary[summary['MissingPercent']<25]
    return summary,list.index



def detect_outliers(df: pd.DataFrame, method: str = "zscore", threshold: float = 3.0) -> pd.DataFrame:
    """
    Detect outliers in a pandas DataFrame using Z-score or IQR method.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (numeric columns will be used)
    method : str, optional
        Outlier detection method. 'zscore' or 'inter quratile range'
    threshold : float, optional
        Z-score threshold (default=3.0) or IQR multiplier (default=3)
        
    Returns
    -------
    pd.DataFrame
        A DataFrame of the same shape as `df` with boolean mask (True = outlier)
    """
    numeric_df = df.select_dtypes(include=[np.number])

    # use z score
    if method.lower() == "zscore":
        # Compute Z-scores
        z_scores = (numeric_df - numeric_df.mean()) / numeric_df.std(ddof=0)
        outliers = np.abs(z_scores) > threshold
    
    # use percentile 
    elif method.lower() == "iqr":
        # Compute IQR
        Q1 = numeric_df.quantile(0.25)
        Q3 = numeric_df.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - (threshold * IQR)
        upper_bound = Q3 + (threshold * IQR)
        outliers = (numeric_df < lower_bound) | (numeric_df > upper_bound)

    else:
        raise ValueError("Invalid method. Choose either 'zscore' or 'iqr'.")

    return outliers.fillna(False)


def detect_feature_types(df: pd.DataFrame, threshold_setup: bool = False,threshold_unique: int = 30):
    """
    Detect numeric (int/float) and categorical features automatically.
    
    Parameters
    ============
    df : pd.DataFrame
        Input dataframe
    threshold_unique : int
        Max unique values allowed for a numeric column to be treated as categorical/ordinal feature
    
    Returns
    ==========
    dict
        Dictionary with 'numeric' and 'categorical' feature lists
    """
    numeric_features = []
    categorical_features = []
    
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # numeric type, but might behave like ordinals (e.g., 0/1 or 1–5 codes)
            if df[col].nunique() <= threshold_unique:
                if(threshold_setup):
                    categorical_features.append(col)
            else:
                numeric_features.append(col)
        else:
            categorical_features.append(col)
    
    return numeric_features,categorical_features


# ========= Helpers =========
def detect_features(df: pd.DataFrame, target_col: str) -> Tuple[List[str], List[str]]:
    assert target_col in df.columns, f"Target '{target_col}' not found."
    X = df.drop(columns=[target_col])
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    return num_cols, cat_cols


# ========= LabelEncoder wrapper for multiple columns =========
# class MultiColumnLabelEncoder:
#     def __init__(self, missing_token: str = "MISSING"):
#         self.encoders: Dict[str, LabelEncoder] = {}
#         self.missing_token = missing_token

#     def fit(self, X: pd.DataFrame):
#         for col in X.columns:
#             le = LabelEncoder()
#             #cat_modes = X[col].mode().iloc[0]
#             vals = X[col].astype(str).fillna(self.missing_token)
#             #vals = X[col].astype(str).fillna(cat_modes)
#             le.fit(vals)
#             # if self.missing_token not in le.classes_:
#             #     le.classes_ = np.append(le.classes_, self.missing_token)
#             self.encoders[col] = le
#         return self

#     def transform(self, X: pd.DataFrame) -> pd.DataFrame:
#         X = X.copy()
#         for col, le in self.encoders.items():
#             cat_modes = X[col].mode().iloc[0]
#             vals = X[col].astype(str).fillna(self.missing_token)
#             #vals = X[col].astype(str).fillna(cat_modes)
#             mask_known = np.isin(vals, le.classes_)
#             #vals = np.where(mask_known, vals, cat_modes)
#             vals = np.where(mask_known, vals, self.missing_token)
#             X[col] = le.transform(vals)
#         return X

#     def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
#         return self.fit(X).transform(X)
