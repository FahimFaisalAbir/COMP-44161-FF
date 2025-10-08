# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 17:59:19 2025

@author: Fahim
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

            

## Sales model for Redemption forecast

class EnsembleModel:
    def __init__(self):
        self.models = {

            'XGB':XGBRegressor(objective="reg:squarederror", random_state=42, subsample=0.9,reg_lambda=2,
                         n_estiamtor=400, max_depth=9,learning_rate=0.1,colsample_bytree=0.7,n_jobs=-1),

            
            'DecisionTree': DecisionTreeRegressor(max_depth=6,min_samples_leaf=2,
                                                 random_state=42),
        
            
            'RandomForest': RandomForestRegressor(max_features='sqrt', min_samples_split=5,
                                                  min_samples_leaf=1,
                       n_estimators=600, n_jobs=-1, random_state=42)
        }

    def fit(self, X_train, y_train):
        # train multiple models
        for name, model in self.models.items():
            model.fit(X_train, y_train)

    def predict(self, X_test):
        # predict
        if X_test.isnull().any().any():
            X_test = X_test.fillna(0)
            
        preds = pd.DataFrame()
        for name, model in self.models.items():
            preds[name] = model.predict(X_test)
            
        # Fill any NaNs if needed 
        preds_filled = preds.copy()
        preds_filled = preds_filled.apply(lambda row: row.fillna(row.mean()), axis=1)
 
        preds_filled = preds_filled.fillna(0)
        preds_filled['Ensemble'] = preds_filled.mean(axis=1)

        return preds_filled['Ensemble'].values
    
    

    def get_avg_feature_importances(self,  X_train,normalize=True):
        """
        Return feature importances for each model that supports it.
        Also returns an 'Average' column (mean across available models).
        """
        data = {}
        for name, model in self.models.items():
            if hasattr(model, "feature_importances_"):
                imp = np.asarray(model.feature_importances_, dtype=float)
                if normalize and imp.sum() > 0:
                    imp = imp / imp.sum()
                data[name] = imp

        if not data:
            return None
        feature_names = X_train.columns
        
        imp_df = pd.DataFrame(data,index= feature_names)
        imp_df["Average"] = imp_df.mean(axis=1)
        
        return imp_df
