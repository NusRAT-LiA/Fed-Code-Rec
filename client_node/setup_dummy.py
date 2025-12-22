"""
Dummy Data Generator
Creates test organizations with sample Python code
"""

import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Sample code for Org A (Data Science focused)
ORG_A_CODE = {
    "data_loader.py": '''
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
''',
    
    "model_trainer.py": '''
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
'''
}

# Sample code for Org B (Web Development focused)
ORG_B_CODE = {
    "auth_service.py": '''
"""User authentication service"""
from datetime import datetime, timedelta
import hashlib
import secrets

class AuthService:
    """Handle user authentication and sessions"""
    
    def __init__(self, secret_key):
        self.secret_key = secret_key
        self.sessions = {}
    
    def hash_password(self, password, salt=None):
        """Hash password with salt"""
        if salt is None:
            salt = secrets.token_hex(16)
        
        pwd_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        )
        return salt, pwd_hash.hex()
    
    def verify_password(self, password, salt, hash_stored):
        """Verify password against stored hash"""
        _, pwd_hash = self.hash_password(password, salt)
        return pwd_hash == hash_stored
    
    def create_session(self, user_id, expiry_hours=24):
        """Create new user session"""
        token = secrets.token_urlsafe(32)
        expiry = datetime.now() + timedelta(hours=expiry_hours)
        
        self.sessions[token] = {
            'user_id': user_id,
            'expires': expiry
        }
        
        return token
    
    def validate_session(self, token):
        """Validate session token"""
        if token not in self.sessions:
            return None
        
        session = self.sessions[token]
        if datetime.now() > session['expires']:
            del self.sessions[token]
            return None
        
        return session['user_id']

def register_user(username, password, email):
    """Register new user"""
    auth = AuthService(secret_key='app_secret')
    salt, pwd_hash = auth.hash_password(password)
    
    user_data = {
        'username': username,
        'email': email,
        'password_salt': salt,
        'password_hash': pwd_hash,
        'created_at': datetime.now()
    }
    
    return user_data
''',
    
    "api_routes.py": '''
"""REST API route handlers"""
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel

app = FastAPI()

class UserLogin(BaseModel):
    """Login request model"""
    username: str
    password: str

class UserResponse(BaseModel):
    """User response model"""
    id: int
    username: str
    email: str

def get_current_user(token: str):
    """Dependency to get current user from token"""
    # Validate token logic here
    return {"id": 1, "username": "demo"}

@app.post("/api/login")
async def login(credentials: UserLogin):
    """User login endpoint"""
    # Authentication logic here
    return {"token": "sample_token", "expires": 3600}

@app.get("/api/users/me")
async def get_user_profile(current_user = Depends(get_current_user)):
    """Get current user profile"""
    return current_user

@app.post("/api/data")
async def create_data(data: dict, current_user = Depends(get_current_user)):
    """Create new data entry"""
    return {"id": 123, "status": "created"}

class DatabaseConnection:
    """Database connection manager"""
    
    def __init__(self, connection_string):
        self.connection_string = connection_string
    
    def connect(self):
        """Establish database connection"""
        pass
    
    def execute_query(self, query, params=None):
        """Execute database query"""
        pass
'''
}

# Sample code for Org C (Enterprise Java-style Python)
ORG_C_CODE = {
    "business_logic.py": '''
"""Core business logic layer"""
from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Order:
    """Order entity"""
    order_id: int
    customer_id: int
    items: List[dict]
    total_amount: float
    status: str
    created_at: datetime

class OrderService(ABC):
    """Abstract order service interface"""
    
    @abstractmethod
    def create_order(self, customer_id: int, items: List[dict]) -> Order:
        """Create new order"""
        pass
    
    @abstractmethod
    def get_order(self, order_id: int) -> Optional[Order]:
        """Retrieve order by ID"""
        pass
    
    @abstractmethod
    def update_order_status(self, order_id: int, status: str) -> bool:
        """Update order status"""
        pass

class OrderServiceImpl(OrderService):
    """Order service implementation"""
    
    def __init__(self, repository):
        self.repository = repository
    
    def create_order(self, customer_id: int, items: List[dict]) -> Order:
        """Create new order"""
        total = sum(item['price'] * item['quantity'] for item in items)
        
        order = Order(
            order_id=self._generate_id(),
            customer_id=customer_id,
            items=items,
            total_amount=total,
            status='pending',
            created_at=datetime.now()
        )
        
        self.repository.save(order)
        return order
    
    def get_order(self, order_id: int) -> Optional[Order]:
        """Retrieve order by ID"""
        return self.repository.find_by_id(order_id)
    
    def update_order_status(self, order_id: int, status: str) -> bool:
        """Update order status"""
        order = self.get_order(order_id)
        if order:
            order.status = status
            self.repository.update(order)
            return True
        return False
    
    def _generate_id(self) -> int:
        """Generate unique order ID"""
        return hash(datetime.now()) % 1000000

def calculate_discount(order: Order, discount_rate: float) -> float:
    """Calculate discount for order"""
    return order.total_amount * discount_rate
''',
    
    "data_repository.py": '''
"""Data repository layer"""
from typing import List, Optional, TypeVar, Generic
from abc import ABC, abstractmethod

T = TypeVar('T')

class Repository(ABC, Generic[T]):
    """Generic repository interface"""
    
    @abstractmethod
    def save(self, entity: T) -> T:
        """Save entity"""
        pass
    
    @abstractmethod
    def find_by_id(self, entity_id: int) -> Optional[T]:
        """Find entity by ID"""
        pass
    
    @abstractmethod
    def find_all(self) -> List[T]:
        """Find all entities"""
        pass
    
    @abstractmethod
    def update(self, entity: T) -> bool:
        """Update entity"""
        pass
    
    @abstractmethod
    def delete(self, entity_id: int) -> bool:
        """Delete entity"""
        pass

class InMemoryRepository(Repository[T]):
    """In-memory repository implementation"""
    
    def __init__(self):
        self.storage = {}
        self.next_id = 1
    
    def save(self, entity: T) -> T:
        """Save entity to memory"""
        entity_id = getattr(entity, 'id', self.next_id)
        self.storage[entity_id] = entity
        self.next_id += 1
        return entity
    
    def find_by_id(self, entity_id: int) -> Optional[T]:
        """Find entity by ID"""
        return self.storage.get(entity_id)
    
    def find_all(self) -> List[T]:
        """Find all entities"""
        return list(self.storage.values())
    
    def update(self, entity: T) -> bool:
        """Update entity"""
        entity_id = getattr(entity, 'id', None)
        if entity_id and entity_id in self.storage:
            self.storage[entity_id] = entity
            return True
        return False
    
    def delete(self, entity_id: int) -> bool:
        """Delete entity"""
        if entity_id in self.storage:
            del self.storage[entity_id]
            return True
        return False

class TransactionManager:
    """Manage database transactions"""
    
    def __init__(self):
        self.active = False
    
    def begin_transaction(self):
        """Begin new transaction"""
        self.active = True
    
    def commit(self):
        """Commit transaction"""
        self.active = False
    
    def rollback(self):
        """Rollback transaction"""
        self.active = False
'''
}


def create_org_structure(org_name: str, code_files: dict, base_dir: str = "."):
    """
    Create organization directory structure with code files.
    
    Args:
        org_name: Name of the organization
        code_files: Dict mapping filenames to code content
        base_dir: Base directory for creation
    """
    org_path = Path(base_dir) / org_name
    data_path = org_path / "data"
    
    # Create directories
    data_path.mkdir(parents=True, exist_ok=True)
    
    # Create code files
    for filename, content in code_files.items():
        filepath = data_path / filename
        with open(filepath, 'w') as f:
            f.write(content)
        logger.info(f"Created: {filepath}")
    
    logger.info(f"Organization '{org_name}' created with {len(code_files)} files")


def main():
    """Create all dummy organizations."""
    logger.info("Creating dummy organizations...")
    
    # Create three organizations
    create_org_structure("org_A", ORG_A_CODE)
    create_org_structure("org_B", ORG_B_CODE)
    create_org_structure("org_C", ORG_C_CODE)
    
    logger.info("\n" + "="*50)
    logger.info("Dummy data setup complete!")
    logger.info("="*50)
    logger.info("\nCreated organizations:")
    logger.info("  - org_A: Data Science (ML pipelines, model training)")
    logger.info("  - org_B: Web Development (API routes, authentication)")
    logger.info("  - org_C: Enterprise (Business logic, repositories)")
    logger.info("\nNext steps:")
    logger.info("  1. cd org_A && python ../indexer.py")
    logger.info("  2. cd org_B && python ../indexer.py")
    logger.info("  3. cd org_C && python ../indexer.py")


if __name__ == "__main__":
    main()