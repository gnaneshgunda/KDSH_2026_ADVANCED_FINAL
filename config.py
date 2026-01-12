"""
Configuration and setup module for Advanced Narrative Consistency RAG
"""

import os
import logging
import nltk
import spacy
from pathlib import Path
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# API Configuration
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

# Path Configuration
DEFAULT_BOOKS_DIR = Path(os.getenv("BOOKS_DIR", "./db/books"))
DEFAULT_CSV_PATH = Path(os.getenv("CSV_PATH", "./db/train.csv"))
DEFAULT_DB_PATH = Path(os.getenv("DB_PATH", "./db"))  # Per-book pkl storage
DEFAULT_OUTPUT_FILE = os.getenv("OUTPUT_FILE", "./db/results.csv")

# Ensure output and DB directories exist
Path(DEFAULT_OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
DEFAULT_DB_PATH.mkdir(parents=True, exist_ok=True)

# Chunking Configuration
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "400"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))

# Embedding Configuration
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))

# Retrieval Configuration
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
MAX_TOP_K = int(os.getenv("MAX_TOP_K", "20"))
RETRIEVAL_STEP = int(os.getenv("RETRIEVAL_STEP", "3"))

# Fallback Loop Configuration
MAX_RETRIEVAL_ROUNDS = int(os.getenv("MAX_RETRIEVAL_ROUNDS", "3"))
FALLBACK_ENABLED = os.getenv("FALLBACK_ENABLED", "true").lower() == "true"

# Graph RAG Configuration
DEFAULT_MIN_EDGE_DENSITY = float(os.getenv("MIN_EDGE_DENSITY", "0.3"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))
MULTI_HOP_DEPTH = int(os.getenv("MULTI_HOP_DEPTH", "2"))

# Evidence Configuration
MAX_SUPPORTING_CHUNKS = int(os.getenv("MAX_SUPPORTING_CHUNKS", "6"))
MAX_OPPOSING_CHUNKS = int(os.getenv("MAX_OPPOSING_CHUNKS", "4"))

# Negation Configuration
NEGATION_THRESHOLD = float(os.getenv("NEGATION_THRESHOLD", "0.7"))

# LLM Configuration
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logger.setLevel(getattr(logging, LOG_LEVEL))

# NLP Setup
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("Loaded spaCy model: en_core_web_sm")
except OSError:
    logger.warning("spaCy model 'en_core_web_sm' not found. Downloading...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# NLTK Setup
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("Downloading NLTK data...")
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('universal_tagset')