# Configuration settings for CDR RAG Pipeline
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
STATIC_DIR = PROJECT_ROOT / "static"

# Data file path
ES_DATA_FILE = DATA_DIR / "es_data.json"

# ChromaDB settings
CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"
CHROMA_COLLECTION_NAME = "cdr_records"

# Embedding model settings
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Ollama/Mistral settings
OLLAMA_BASE_URL = "http://localhost:11434"
MISTRAL_MODEL_NAME = "mistral"

# API settings
API_HOST = "127.0.0.1"
API_PORT = 8000

# Retrieval settings
DEFAULT_RETRIEVAL_COUNT = 10
MAX_CONTEXT_LENGTH = 4000

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
