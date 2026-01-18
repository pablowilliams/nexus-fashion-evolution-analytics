"""Data loading and preprocessing for fashion evolution analysis"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.encoders = {}
        self.scaler = StandardScaler()
        
    def load_data(self):
        df = pd.read_csv(self.filepath)
        return self._preprocess(df)
    
    def _preprocess(self, df):
        categorical_cols = ['category', 'silhouette', 'neckline', 'sleeve_type', 
                           'hemline', 'primary_color', 'pattern', 'fabric', 
                           'sustainability_label', 'price_segment']
        for col in categorical_cols:
            self.encoders[col] = LabelEncoder()
            df[f'{col}_encoded'] = self.encoders[col].fit_transform(df[col].astype(str))
        return df
    
    def get_temporal_features(self, df):
        df['year_normalized'] = (df['year'] - df['year'].min()) / (df['year'].max() - df['year'].min())
        df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'].str[1].astype(int) / 4)
        df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'].str[1].astype(int) / 4)
        return df
