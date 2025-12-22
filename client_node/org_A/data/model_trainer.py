
"""ML model training utilities"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import joblib

def train_model(X_train, y_train, model_type='rf'):
    """Train a machine learning model"""
    if model_type == 'rf':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X, y, cv=5):
    """Evaluate model using cross-validation"""
    scores = cross_val_score(model, X, y, cv=cv)
    return {
        'mean_score': scores.mean(),
        'std_score': scores.std()
    }

class ModelManager:
    """Manage model lifecycle"""
    
    def __init__(self, model_path='models/'):
        self.model_path = model_path
    
    def save_model(self, model, filename):
        """Save trained model"""
        path = f"{self.model_path}/{filename}"
        joblib.dump(model, path)
    
    def load_model(self, filename):
        """Load saved model"""
        path = f"{self.model_path}/{filename}"
        return joblib.load(path)
