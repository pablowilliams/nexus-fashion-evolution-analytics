"""Trend analysis and pattern detection for fashion data"""
import pandas as pd
import numpy as np

class TrendAnalyzer:
    def __init__(self):
        self.trend_cache = {}
        
    def compute_trend_velocity(self, df, attribute, time_col='year'):
        grouped = df.groupby([time_col, attribute]).size().unstack(fill_value=0)
        velocity = grouped.diff().fillna(0)
        return velocity
    
    def identify_rising_trends(self, df, attribute, threshold=0.1):
        velocity = self.compute_trend_velocity(df, attribute)
        recent = velocity.iloc[-3:].mean()
        rising = recent[recent > threshold * recent.max()].index.tolist()
        return rising
    
    def seasonal_decomposition(self, df, metric='trend_score'):
        seasonal = df.groupby('quarter')[metric].mean()
        return seasonal.to_dict()
    
    def correlation_analysis(self, df, features, target='trend_score'):
        correlations = df[features + [target]].corr()[target].drop(target)
        return correlations.sort_values(ascending=False)
