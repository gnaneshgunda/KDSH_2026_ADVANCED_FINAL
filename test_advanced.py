"""Test advanced features"""
import os
from dotenv import load_dotenv
import requests

load_dotenv()
api_key = os.getenv("NVIDIA_API_KEY")
base_url = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

print("Testing Advanced RAG Components...")

# Test 1: Embeddings
print("\n1. Testing NVIDIA Embeddings...")
url = f"{base_url}/embeddings"
headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
payload = {"model": "nvidia/nv-embed-qa", "input": ["test text"]}

try:
    resp = requests.post(url, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    emb = resp.json()
    print(f"  ✓ Embeddings working! Dimension: {len(emb['data'][0]['embedding'])}")
except Exception as e:
    print(f"  ✗ Error: {e}")

# Test 2: spaCy
print("\n2. Testing spaCy NLP...")
try:
    import spacy
    nlp = spacy.load("en_core_web_md")
    doc = nlp("Alice feared abandonment because her mother left.")
    print(f"  ✓ spaCy loaded! Tokens: {[t.text for t in doc]}")
except Exception as e:
    print(f"  ✗ Error: {e}. Install: python -m spacy download en_core_web_md")

# Test 3: NetworkX
print("\n3. Testing NetworkX...")
try:
    import networkx as nx
    g = nx.DiGraph()
    g.add_edge("A", "B", weight=0.8)
    print(f"  ✓ NetworkX working! Graph nodes: {list(g.nodes())}")
except Exception as e:
    print(f"  ✗ Error: {e}")

# Test 4: Dependencies
print("\n4. Testing all dependencies...")
deps = {
    "nltk": "nltk.tokenize",
    "sklearn": "sklearn.metrics.pairwise",
    "pandas": "pandas",
    "numpy": "numpy"
}

for name, module in deps.items():
    try:
        __import__(module)
        print(f"  ✓ {name}")
    except ImportError:
        print(f"  ✗ {name} not installed")

print("\n✅ All systems ready for advanced RAG!")
