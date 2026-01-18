"""Trend score prediction model"""
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score

class TrendPredictor:
    def __init__(self):
        self.model = XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42)
        
    def train(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X, y, cv=5):
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='r2')
        return {'mean_r2': scores.mean(), 'std_r2': scores.std()}
    
    def get_feature_importance(self):
        return self.model.feature_importances_
