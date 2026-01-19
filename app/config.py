import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Flask API Configuration"""
    
    # Flask settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    HOST = os.getenv('FLASK_HOST', '0.0.0.0')
    PORT = int(os.getenv('FLASK_PORT', 8000))
    
    # Corvo path configuration
    CORVO_PATH = Path(os.getenv('CORVO_PATH', '../corvo')).resolve()
    
    # Milvus configuration
    MILVUS_URI = os.getenv('MILVUS_URI', 'http://localhost:19530')
    MILVUS_TOKEN = os.getenv('MILVUS_TOKEN', '')
    MILVUS_COLLECTION = os.getenv('MILVUS_COLLECTION', 'corvo')
    
    # CORS configuration
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*').split(',')
    
    # Validate Corvo path exists
    @classmethod
    def validate(cls):
        if not cls.CORVO_PATH.exists():
            raise ValueError(f"Corvo path does not exist: {cls.CORVO_PATH}")
        if not (cls.CORVO_PATH / 'app').exists():
            raise ValueError(f"Corvo app directory not found: {cls.CORVO_PATH / 'app'}")
        return True

