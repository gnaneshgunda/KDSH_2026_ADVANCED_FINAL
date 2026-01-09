"""
LangGraph nodes for Advanced Narrative Consistency RAG
"""

import logging
from typing import Dict, List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from models import GraphState, BackstoryClaim
from config import (
    nlp, DEFAULT_TOP_K, SIMILARITY_THRESHOLD, 
    NEGATION_THRESHOLD, MAX_SUPPORTING_CHUNKS, MAX_OPPOSING_CHUNKS
)

logger = logging.getLogger(__name__)


def load_corpus_node(state: GraphState) -> GraphState:
    """Node: Load corpus for the specified book"""
    logger.info(f"[Node] Loading corpus for book: {state['book_key']}")
    
    if state['book_key'] not in state['corpus']:
        state['error'] = f"Book '{state['book_key']}' not found in corpus"
        logger.warning(state['error'])
    
    state['chunks'] = state['corpus'].get(state['book_key'], [])
    logger.info(f"Loaded {len(state['chunks'])} chunks")
    return state


def extract_claims_node(state: GraphState) -> GraphState:
    """Node: Extract structured claims from backstory"""
    logger.info(f"[Node] Extracting claims for: {state['character']}")
    
    backstory_text = state['backstory_text']
    if not backstory_text or len(backstory_text.strip()) < 10:
        state['claims'] = []
        state['claim_embeddings'] = []
        logger.warning("Empty backstory")
        return state
    
    # Parse backstory into sentences
    doc = nlp(backstory_text)
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]
    
    state['claims'] = sentences[:10]  # Limit to 10 claims
    logger.info(f"Extracted {len(state['claims'])} claims")
    return state


def embed_claims_node(state: GraphState) -> GraphState:
    """Node: Generate embeddings for claims"""
    logger.info(f"[Node] Embedding {len(state['claims'])} claims")
    
    if not state['claims']:
        state['claim_embeddings'] = []
        return state
    
    # Get client from state (injected)
    client = state.get('_client')
    if not client:
        state['error'] = "Client not available"
        return state
    
    try:
        embeddings = client.embed_texts(state['claims'])
        state['claim_embeddings'] = embeddings
        logger.info(f"Generated {len(embeddings)} embeddings")
    except Exception as e:
        state['error'] = f"Embedding failed: {e}"
        logger.error(state['error'])
    
    return state


def retrieve_supporting_node(state: GraphState) -> GraphState:
    """Node: Retrieve supporting evidence chunks"""
    logger.info(f"[Node] Retrieving supporting chunks")
    
    if not state['claim_embeddings'] or not state['chunks']:
        state['supporting_chunks'] = []
        return state
    
    supporting = []
    chunks = state['chunks']
    
    for claim_emb in state['claim_embeddings']:
        # Compute similarities
        chunk_embeddings = np.array([c.embedding for c in chunks])
        similarities = cosine_similarity([claim_emb], chunk_embeddings)[0]
        
        # Get top-k
        top_indices = np.argsort(similarities)[-DEFAULT_TOP_K:][::-1]
        
        for idx in top_indices:
            if similarities[idx] >= SIMILARITY_THRESHOLD:
                supporting.append((chunks[idx].chunk_id, float(similarities[idx])))
    
    # Deduplicate and limit
    seen = set()
    unique_supporting = []
    for chunk_id, score in supporting:
        if chunk_id not in seen:
            seen.add(chunk_id)
            unique_supporting.append((chunk_id, score))
    
    state['supporting_chunks'] = unique_supporting[:MAX_SUPPORTING_CHUNKS]
    logger.info(f"Found {len(state['supporting_chunks'])} supporting chunks")
    return state


def retrieve_opposing_node(state: GraphState) -> GraphState:
    """Node: Retrieve opposing/contradicting evidence chunks"""
    logger.info(f"[Node] Retrieving opposing chunks")
    
    if not state['claim_embeddings'] or not state['chunks']:
        state['opposing_chunks'] = []
        return state
    
    opposing = []
    chunks = state['chunks']
    
    for claim_emb in state['claim_embeddings']:
        # Compute similarities
        chunk_embeddings = np.array([c.embedding for c in chunks])
        similarities = cosine_similarity([claim_emb], chunk_embeddings)[0]
        
        # Look for negations (low similarity or semantic opposition)
        for idx, sim in enumerate(similarities):
            if sim < NEGATION_THRESHOLD:
                opposing.append((chunks[idx].chunk_id, float(sim)))
    
    # Sort by lowest similarity (strongest opposition)
    opposing.sort(key=lambda x: x[1])
    
    # Deduplicate and limit
    seen = set()
    unique_opposing = []
    for chunk_id, score in opposing:
        if chunk_id not in seen:
            seen.add(chunk_id)
            unique_opposing.append((chunk_id, score))
    
    state['opposing_chunks'] = unique_opposing[:MAX_OPPOSING_CHUNKS]
    logger.info(f"Found {len(state['opposing_chunks'])} opposing chunks")
    return state


def analyze_consistency_node(state: GraphState) -> GraphState:
    """Node: LLM-based consistency analysis"""
    logger.info(f"[Node] Analyzing consistency with LLM")
    
    client = state.get('_client')
    if not client:
        state['error'] = "Client not available"
        return state
    
    # Build prompt
    prompt = _build_analysis_prompt(state)
    
    try:
        messages = [{"role": "user", "content": prompt}]
        response = client.chat_completion(messages)
        
        # Parse response
        prediction, confidence, reasoning = _parse_llm_response(response)
        
        state['prediction'] = prediction
        state['confidence'] = confidence
        state['reasoning'] = reasoning
        
        logger.info(f"Prediction: {prediction} | Confidence: {confidence:.2f}")
    except Exception as e:
        state['error'] = f"Analysis failed: {e}"
        logger.error(state['error'])
        state['prediction'] = 0
        state['confidence'] = 0.0
        state['reasoning'] = f"Error: {e}"
    
    return state


def _build_analysis_prompt(state: GraphState) -> str:
    """Build prompt for LLM consistency analysis"""
    chunks = state['chunks']
    corpus_map = {c.chunk_id: c.text for c in chunks}
    
    supporting_text = "\n".join([
        f"- [{cid}] {corpus_map.get(cid, 'N/A')[:200]}"
        for cid, _ in state['supporting_chunks'][:3]
    ])
    
    opposing_text = "\n".join([
        f"- [{cid}] {corpus_map.get(cid, 'N/A')[:200]}"
        for cid, _ in state['opposing_chunks'][:3]
    ])
    
    prompt = f"""Analyze if the character backstory is CONSISTENT with the narrative evidence.

Book: {state['book_key']}
Character: {state['character']}

Backstory Claims:
{chr(10).join(f"{i+1}. {claim}" for i, claim in enumerate(state['claims'][:5]))}

Supporting Evidence:
{supporting_text if supporting_text else "None"}

Opposing Evidence:
{opposing_text if opposing_text else "None"}

Determine:
1. Is the backstory CONSISTENT (1) or CONTRADICTORY (0)?
2. Confidence score (0.0-1.0)
3. Brief reasoning (2-3 sentences)

Format your response as:
PREDICTION: [0 or 1]
CONFIDENCE: [0.0-1.0]
REASONING: [your reasoning]
"""
    return prompt


def _parse_llm_response(response: str) -> tuple:
    """Parse LLM response into prediction, confidence, reasoning"""
    lines = response.strip().split('\n')
    
    prediction = 1
    confidence = 0.5
    reasoning = response
    
    for line in lines:
        line = line.strip()
        if line.startswith("PREDICTION:"):
            try:
                prediction = int(line.split(":")[-1].strip())
            except:
                pass
        elif line.startswith("CONFIDENCE:"):
            try:
                confidence = float(line.split(":")[-1].strip())
            except:
                pass
        elif line.startswith("REASONING:"):
            reasoning = line.split(":", 1)[-1].strip()
    
    return prediction, confidence, reasoning


def error_handler_node(state: GraphState) -> GraphState:
    """Node: Handle errors gracefully"""
    if state.get('error'):
        logger.error(f"[Node] Error handler: {state['error']}")
        state['prediction'] = 0
        state['confidence'] = 0.0
        state['reasoning'] = state['error']
    return state
