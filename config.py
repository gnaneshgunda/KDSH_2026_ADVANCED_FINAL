"""
Configuration for LangGraph-based Advanced Narrative Consistency RAG
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

# Load environment variables
load_dotenv()

from api_key_manager import get_next_api_key

# Model Configuration - LangChain NVIDIA Endpoints
EMBEDDING_CONFIG = {
    "model": "nvidia/llama-3.2-nv-embedqa-1b-v2",
    "truncate": "NONE"
}

CHAT_CONFIG = {
    "model": "moonshotai/kimi-k2-instruct-0905",
    "temperature": 0.6,
    "top_p": 0.9,
    "max_tokens": 4096
}

# NLP Model Configuration
def setup_nltk():
    """Download and setup required NLTK resources"""
    for resource in ["punkt", "punkt_tab", "averaged_perceptron_tagger"]:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)
    logger.info("NLTK resources ready")

def load_spacy_model():
    """Load spaCy model with fallback"""
    try:
        nlp = spacy.load("en_core_web_md")
    except OSError:
        logger.warning("spaCy model not found. Install: python -m spacy download en_core_web_md")
        nlp = spacy.load("en_core_web_sm")
    logger.info("spaCy model loaded")
    return nlp

# Initialize models
setup_nltk()
nlp = load_spacy_model()

# Path Configuration
DB_BOOKS_DIR = Path("./db/books")
DB_CSV_PATH = Path("./db/train.csv")
DEFAULT_INDEX_PATH = Path("langgraph_index.pkl")
DEFAULT_OUTPUT_FILE = "results_langgraph.csv"

# Embedding Configuration
EMBEDDING_DIM = 2048  # llama-3.2-nv-embedqa-1b-v2 dimension

# Chunking Configuration
DEFAULT_CHUNK_SIZE = 200
DEFAULT_MIN_EDGE_DENSITY = 0.3

# RAG Configuration
DEFAULT_TOP_K = 5
SIMILARITY_THRESHOLD = 0.65
NEGATION_THRESHOLD = 0.15
MULTI_HOP_DEPTH = 2
MAX_SUPPORTING_CHUNKS = 5
MAX_OPPOSING_CHUNKS = 5

# LangGraph Configuration
MAX_ITERATIONS = 10
RECURSION_LIMIT = 25

logger.info(f"LangGraph Configuration loaded | Embedding: {EMBEDDING_CONFIG['model']}")
