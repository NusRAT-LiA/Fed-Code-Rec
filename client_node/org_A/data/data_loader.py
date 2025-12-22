
"""Data loading utilities for ML pipelines"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_dataset(filepath, normalize=True):
    """Load and preprocess dataset"""
    df = pd.read_csv(filepath)
    
    if normalize:
        scaler = StandardScaler()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df

def split_features_target(df, target_col):
    """Split dataframe into features and target"""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

class DataPipeline:
    """End-to-end data processing pipeline"""
    
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
    
    def fit_transform(self, data):
        """Fit and transform data"""
        return self.scaler.fit_transform(data)
    
    def transform(self, data):
        """Transform new data"""
        return self.scaler.transform(data)
