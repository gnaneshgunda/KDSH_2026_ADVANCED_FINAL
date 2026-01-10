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

# Load environment variables
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# API Configuration
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

if not NVIDIA_API_KEY:
    logger.warning("NVIDIA_API_KEY not found in environment; running with local stub client")

if not NVIDIA_API_KEY:
    logger.warning("NVIDIA_API_KEY not found in environment; running with local stub client")

# NLP Model Configuration
def setup_nltk():
    """Download and setup required NLTK resources"""
    for resource in ["punkt", "punkt_tab", "averaged_perceptron_tagger"]:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            try:
                logger.info(f"Downloading NLTK resource: {resource}")
                nltk.download(resource, quiet=True, raise_errors=False)
            except Exception as e:
                logger.warning(f"Failed to download NLTK resource '{resource}': {e}. Continuing...")
    logger.info("NLTK resources setup complete")


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
DEFAULT_BOOKS_DIR = Path("./db/books")
DEFAULT_CSV_PATH = Path("./db/train.csv")
DEFAULT_INDEX_PATH = Path("./db/advanced_index.pkl")
DEFAULT_OUTPUT_FILE = "./db/results_advanced.csv"

# Model Configuration
EMBEDDING_DIM = 1024
EMBEDDING_MODEL = "nvidia/nv-embed-qa"
CHAT_MODEL = "meta/llama-3.1-8b-instruct"

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

logger.info(f"Configuration loaded | NVIDIA NIM: {NVIDIA_BASE_URL}")
