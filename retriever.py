import logging
import math
import re
import numpy as np
from collections import Counter, defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Set
from models import ChunkMetadata
from nvidia_client import NVIDIAClient
from config import nlp

logger = logging.getLogger(__name__)

# ---------- Inline BM25-like scorer (no external deps) ----------

# Common English stop words to exclude from BM25 scoring
_STOP_WORDS: Set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
    "neither", "each", "every", "all", "any", "few", "more", "most",
    "other", "some", "such", "no", "only", "own", "same", "than", "too",
    "very", "just", "because", "as", "until", "while", "of", "at", "by",
    "for", "with", "about", "against", "between", "through", "during",
    "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further",
    "then", "once", "here", "there", "when", "where", "why", "how",
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves", "he", "him",
    "his", "himself", "she", "her", "hers", "herself", "it", "its",
    "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def _tokenize(text: str) -> List[str]:
    """Lowercase tokenize, strip stop words."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP_WORDS]


class _BM25Scorer:
    """Simple BM25 (Okapi BM25) scorer built over a list of chunk texts."""

    def __init__(self, documents: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.N = len(documents)

        # Tokenize all documents
        self.doc_tokens: List[List[str]] = [_tokenize(d) for d in documents]
        self.doc_lens = [len(dt) for dt in self.doc_tokens]
        self.avgdl = sum(self.doc_lens) / max(self.N, 1)

        # Document frequency for each term
        self.df: Dict[str, int] = Counter()
        for dt in self.doc_tokens:
            for term in set(dt):
                self.df[term] += 1

    def score(self, query: str) -> np.ndarray:
        """Return BM25 scores for every document given the *query*."""
        query_tokens = _tokenize(query)
        scores = np.zeros(self.N, dtype=np.float64)
        for qt in query_tokens:
            if qt not in self.df:
                continue
            idf = math.log((self.N - self.df[qt] + 0.5) / (self.df[qt] + 0.5) + 1.0)
            for idx, dt in enumerate(self.doc_tokens):
                tf = dt.count(qt)
                denom = tf + self.k1 * (1 - self.b + self.b * self.doc_lens[idx] / max(self.avgdl, 1))
                scores[idx] += idf * (tf * (self.k1 + 1)) / max(denom, 1e-8)
        return scores


# ---------- Multi-query expansion helpers ----------

def _extract_key_nouns(text: str) -> List[str]:
    """Use spaCy to pull key nouns and named entities from *text*."""
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]
    nouns = [token.text for token in doc if token.pos_ in ("NOUN", "PROPN") and not token.is_stop]
    # Deduplicate while preserving order
    seen: set = set()
    result: List[str] = []
    for w in entities + nouns:
        w_lower = w.lower()
        if w_lower not in seen:
            seen.add(w_lower)
            result.append(w)
    return result


def _expand_queries(query: str, character_name: str = None) -> List[str]:
    """Generate 2-3 query variants for multi-query retrieval.

    Variants:
      1. Original claim text
      2. Character-focused: prepend character name
      3. Entity-focused: key nouns / entities from the claim
    """
    variants: List[str] = [query]

    # Character-focused variant
    if character_name:
        variants.append(f"{character_name}: {query}")

    # Entity-focused variant
    key_nouns = _extract_key_nouns(query)
    if key_nouns:
        entity_query = " ".join(key_nouns[:6])
        if entity_query.lower() != query.lower():
            variants.append(entity_query)

    return variants


# ---------- Fuzzy character matching helpers ----------

# Common name abbreviations / nicknames mapping
_COMMON_NICKNAMES: Dict[str, List[str]] = {
    "william": ["will", "bill", "billy", "willy"],
    "robert": ["rob", "bob", "bobby", "robbie"],
    "richard": ["rick", "dick", "rich"],
    "james": ["jim", "jimmy", "jamie"],
    "john": ["johnny", "jon"],
    "thomas": ["tom", "tommy"],
    "edward": ["ed", "eddie", "ted", "teddy"],
    "elizabeth": ["liz", "lizzy", "beth", "eliza", "betty"],
    "margaret": ["maggie", "meg", "peggy", "marge"],
    "catherine": ["kate", "cathy", "cat", "katie"],
    "katherine": ["kate", "kathy", "kat", "katie"],
    "alexander": ["alex", "xander"],
    "benjamin": ["ben", "benji", "benny"],
    "christopher": ["chris", "kit"],
    "nicholas": ["nick", "nicky"],
    "jonathan": ["jon", "jonny"],
    "michael": ["mike", "mikey"],
    "joseph": ["joe", "joey"],
    "charles": ["charlie", "chuck"],
    "patrick": ["pat", "paddy"],
    "daniel": ["dan", "danny"],
    "matthew": ["matt", "matty"],
    "andrew": ["andy", "drew"],
    "timothy": ["tim", "timmy"],
    "samuel": ["sam", "sammy"],
    "nathaniel": ["nate", "nathan", "nat"],
    "theodore": ["theo", "ted", "teddy"],
    "victoria": ["vicky", "tori"],
    "jennifer": ["jen", "jenny"],
    "jessica": ["jess", "jessie"],
    "rebecca": ["becca", "becky"],
    "abigail": ["abby", "abbie"],
    "alexandra": ["alex", "lexi"],
    "caroline": ["carrie", "carol"],
}

# Build reverse mapping as well (nickname -> canonical)
_NICKNAME_REVERSE: Dict[str, str] = {}
for _canon, _nicks in _COMMON_NICKNAMES.items():
    for _n in _nicks:
        _NICKNAME_REVERSE[_n] = _canon


def _character_name_variants(name: str) -> Set[str]:
    """Generate all plausible variants of a character name for matching."""
    name = name.strip()
    parts = name.split()
    variants: Set[str] = {name.lower()}

    # Individual parts (first name, last name, middle names)
    for p in parts:
        p_lower = p.lower()
        variants.add(p_lower)
        # Check nickname mappings in both directions
        if p_lower in _COMMON_NICKNAMES:
            variants.update(_COMMON_NICKNAMES[p_lower])
        if p_lower in _NICKNAME_REVERSE:
            canonical = _NICKNAME_REVERSE[p_lower]
            variants.add(canonical)
            variants.update(_COMMON_NICKNAMES.get(canonical, []))

    # Full name without middle parts (first + last)
    if len(parts) > 2:
        variants.add(f"{parts[0].lower()} {parts[-1].lower()}")

    return variants


# ---------- Main Retriever ----------

class HybridRetriever:
    """
    Multi-stage hybrid retrieval:
    1. Metadata filtering (character, temporal, location) with fuzzy matching
    2. Multi-query expansion (original + character-focused + entity-focused)
    3. Hybrid scoring: weighted combination of semantic cosine + BM25 keyword
    4. Reranking (NVIDIA rerank API)
    5. Context expansion (neighboring chunks)
    """

    # Weights for hybrid scoring (semantic vs keyword)
    SEMANTIC_WEIGHT = 0.7
    BM25_WEIGHT = 0.3

    def __init__(self, chunks: List[ChunkMetadata], client: NVIDIAClient):
        self.chunks = chunks
        self.client = client
        self.embeddings = np.array([c.embedding for c in chunks])

        # Pre-build BM25 index over chunk texts
        self._bm25 = _BM25Scorer([c.text for c in chunks])
        logger.info(f"HybridRetriever initialized with {len(chunks)} chunks (BM25 index built)")

    # -------- public API (signature unchanged) --------

    def retrieve(
        self,
        query: str,
        character_name: str = None,
        top_k: int = 7,
        context_window: int = 1,
        use_rerank: bool = True,
    ) -> List[ChunkMetadata]:
        """
        Retrieve relevant chunks with metadata filtering, multi-query expansion,
        hybrid BM25 + semantic scoring, and optional reranking.

        Args:
            query: Search query (claim text)
            character_name: Filter by character mentions
            top_k: Number of chunks to return
            context_window: Number of neighboring chunks to include
            use_rerank: Whether to use NVIDIA rerank API
        """
        if not self.chunks:
            return []

        # STEP 1: Metadata-based filtering (fuzzy character matching)
        filtered_chunks = self._filter_by_metadata(character_name)

        # If too few results, fall back to all chunks
        search_pool = filtered_chunks if len(filtered_chunks) >= top_k else self.chunks
        logger.debug(
            f"Search pool: {len(search_pool)} chunks "
            f"(filtered: {len(filtered_chunks)}, total: {len(self.chunks)})"
        )

        # Build a set of pool indices (relative to self.chunks) for BM25 subsetting
        pool_index_map: Dict[int, int] = {}  # pool_pos -> original_idx
        for pool_pos, chunk in enumerate(search_pool):
            orig_idx = self.chunks.index(chunk)
            pool_index_map[pool_pos] = orig_idx

        # STEP 2: Multi-query expansion
        query_variants = _expand_queries(query, character_name)
        logger.debug(f"Multi-query: {len(query_variants)} variants")

        # STEP 3: Hybrid scoring for each variant, merge by max score per chunk
        best_scores: Dict[int, float] = defaultdict(float)  # pool_pos -> best score

        pool_embeddings = np.array([c.embedding for c in search_pool])
        effective_dim = self.client.get_embedding_dim()

        # Pre-compute BM25 scores for all variants (full corpus then subset)
        for variant in query_variants:
            # --- Semantic component ---
            query_emb = np.array(self.client.embed([variant])[0])
            if effective_dim < len(query_emb):
                query_emb_trunc = query_emb[:effective_dim]
                pool_emb_trunc = pool_embeddings[:, :effective_dim]
            else:
                query_emb_trunc = query_emb
                pool_emb_trunc = pool_embeddings

            semantic_sims = cosine_similarity([query_emb_trunc], pool_emb_trunc)[0]

            # Normalise semantic scores to [0, 1]
            sem_min, sem_max = semantic_sims.min(), semantic_sims.max()
            if sem_max - sem_min > 1e-8:
                semantic_norm = (semantic_sims - sem_min) / (sem_max - sem_min)
            else:
                semantic_norm = np.zeros_like(semantic_sims)

            # --- BM25 component ---
            full_bm25 = self._bm25.score(variant)
            # Subset to pool
            bm25_pool = np.array([full_bm25[pool_index_map[p]] for p in range(len(search_pool))])

            bm25_min, bm25_max = bm25_pool.min(), bm25_pool.max()
            if bm25_max - bm25_min > 1e-8:
                bm25_norm = (bm25_pool - bm25_min) / (bm25_max - bm25_min)
            else:
                bm25_norm = np.zeros_like(bm25_pool)

            # --- Combine ---
            hybrid = self.SEMANTIC_WEIGHT * semantic_norm + self.BM25_WEIGHT * bm25_norm

            for pool_pos in range(len(search_pool)):
                if hybrid[pool_pos] > best_scores[pool_pos]:
                    best_scores[pool_pos] = hybrid[pool_pos]

        # Sort by best hybrid score descending
        ranked_pool_positions = sorted(best_scores, key=best_scores.get, reverse=True)

        # Take top 2*top_k for reranking
        initial_k = min(top_k * 2, len(ranked_pool_positions))
        ranked_indices = ranked_pool_positions[:initial_k]

        # STEP 4: Reranking (optional)
        if use_rerank and len(ranked_indices) > top_k:
            try:
                candidates = [
                    {"text": search_pool[i].text, "index": i}
                    for i in ranked_indices
                ]
                reranked = self.client.rerank(query, candidates)
                ranked_indices = [r["index"] for r in reranked[:top_k]]
                logger.debug(f"Reranked {len(candidates)} candidates to {len(ranked_indices)}")
            except Exception as e:
                logger.warning(f"Reranking failed: {e}, using hybrid scores")
                ranked_indices = ranked_indices[:top_k]
        else:
            ranked_indices = ranked_indices[:top_k]

        # Map back to original indices
        selected_indices: Set[int] = set()
        for pool_idx in ranked_indices:
            original_idx = pool_index_map[pool_idx]
            selected_indices.add(original_idx)

        # STEP 5: Context expansion
        expanded_indices: Set[int] = set()
        for idx in selected_indices:
            start = max(0, idx - context_window)
            end = min(len(self.chunks) - 1, idx + context_window)
            for i in range(start, end + 1):
                expanded_indices.add(i)

        # Return in narrative order
        result = [self.chunks[i] for i in sorted(expanded_indices)]

        logger.debug(
            f"Retrieved {len(selected_indices)} core + "
            f"{len(expanded_indices) - len(selected_indices)} context chunks"
        )
        return result

    # -------- private helpers --------

    def _filter_by_metadata(self, character_name: str = None) -> List[ChunkMetadata]:
        """
        Filter chunks by metadata (character mentions) with fuzzy name matching.

        Matches first names, last names, and common abbreviations / nicknames.
        """
        if not character_name:
            return self.chunks

        name_variants = _character_name_variants(character_name)
        filtered: List[ChunkMetadata] = []

        for chunk in self.chunks:
            # Check entity list
            entities_lower = [e.lower() for e in chunk.entities]

            # Fuzzy match: any name variant appears in any entity or raw text
            text_lower = chunk.text.lower()

            match = False
            for variant in name_variants:
                # Check entities
                if any(variant in e or e in variant for e in entities_lower):
                    match = True
                    break
                # Check raw text
                if variant in text_lower:
                    match = True
                    break

            if match:
                filtered.append(chunk)

        logger.debug(
            f"Filtered {len(filtered)}/{len(self.chunks)} chunks "
            f"for character '{character_name}' ({len(name_variants)} name variants)"
        )
        return filtered
