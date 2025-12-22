
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
