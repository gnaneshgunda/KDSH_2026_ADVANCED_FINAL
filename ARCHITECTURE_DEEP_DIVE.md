# Advanced RAG Architecture - Deep Dive

## 1. Dependency-Based Intelligent Chunking

### Problem
- Naive word-count chunking breaks at semantically important boundaries
- Splits clauses, breaks causal relationships, loses context

### Solution: Dependency Parsing Chunking
```
Input: "Alice had always feared abandonment. Her mother left when she was eight, 
        and she never fully recovered from the trauma."

Dependency tree:
                    feared
                   /      \
                Alice     abandonment

                left
               /    \
            mother   when
                    /
                  she
                 /
              was
              |
            eight

Chunk boundary: AFTER a sentence if word count > 200
Keeps semantic units intact: "fear → abandonment" stays together
```

### Implementation
- Uses spaCy: `doc.to_dependency_structure()`
- Builds NetworkX DiGraph per chunk
- Tracks entity relationships
- Stores: text + graph + entities + POS tags

## 2. Context Vector Construction

### Traditional Approach
```python
embedding = embed(text)  # [1024 floats]
similarity = cosine(embedding1, embedding2)  # Just embeddings
```

**Problem:** Loses temporal/emotional/causal information

### Advanced Approach: Multi-Signal Context Vector
```python
base_embedding = embed(text)  # [900 dims] - original

sentiment = analyze_sentiment(text)  # -1 to 1
sentiment_vec = [sentiment] * 32  # [32 dims]

temporal = extract_temporal_markers(text)  # "past"/"present"/"future"
temporal_vec = [temporal_score] * 32  # [32 dims]

causal = extract_causal_indicators(text)  # "because", "led to", etc
causal_vec = [causal_score] * 32  # [32 dims]

context_vector = concat([
    base_embedding,   # [900]
    sentiment_vec,    # [32] - emotional context
    temporal_vec,     # [32] - temporal context  
    causal_vec        # [32] - causal context
])  # = [1024] total

# Now cosine_similarity uses ALL information!
```

### Example
```
Backstory: "She feared abandonment" 
  → sentiment: -0.8 (fear)
  → temporal: "past" (happened before)
  → causal: "led to over-cautiousness"

Narrative chunk 1: "Alice hesitated to trust."
  → sentiment: -0.3
  → temporal: "present"
  → causal: "because of past trauma"

Narrative chunk 2: "She made reckless decisions."
  → sentiment: +0.2
  → temporal: "present"
  → causal: none

Similarity with context:
  Chunk 1: 0.92 (emotion + temporal + causal all align!)
  Chunk 2: 0.35 (opposite emotional signal)
```

## 3. Semantic Negation (Geometrical Opposites)

### Problem
How to find contradicting evidence? Simple: look for opposites.

### Solution: LLM Negation Generator

```python
backstory_claim = "She grew up in poverty and learned to distrust wealth"

negated = llm.negate(backstory_claim)
# Output: "She grew up in privilege and embraced wealth as virtue"

# Now search narrative for chunks similar to NEGATED claim
# If found: contradiction!
# If not: no contradiction
```

### Algorithm
```
1. For each backstory claim C:

2. Generate semantic opposite: C_neg = LLM(negate(C))

3. Embed both: E_c, E_neg

4. Search narrative:
   - Find chunks similar to E_c: Supporting chunks
   - Find chunks similar to E_neg: Opposing chunks

5. If opposing chunks exist AND are high confidence:
   → Contradiction detected!

6. LLM sees both supporting + opposing:
   → Makes informed consistency judgment
```

### Example
```
Backstory: "Character never trusted anyone"
Negated:   "Character readily trusted everyone"

Supporting chunks found:
  - "She kept her secrets well" (0.89 similarity to original)
  - "Even her best friend didn't know about..." (0.87)

Opposing chunks found:
  - "She confided in the first stranger" (0.76 similarity to negation)

LLM conclusion: "Some contradiction. Character's trust behavior is selective,
not absolute. Backstory oversimplified."
```

## 4. Graph-RAG Multi-hop Reasoning

### Problem
Single-hop retrieval misses contextual chains:
```
Fact A: "She was abandoned"
Fact B: "She developed trust issues"  
Fact C: "She later sabotaged relationships"

A → B → C (causal chain, but separated by narrative distance)

Single-hop: May miss A when searching for C
Multi-hop: Finds A → B → C pathway
```

### Solution: Knowledge Graph + Multi-hop Search

```python
# Build graph
graph = nx.DiGraph()
for chunk in chunks:
    graph.add_node(chunk.id)

# Add edges: high semantic similarity (cosine > 0.65)
for i, j in pairs:
    if cosine(chunk_i.embedding, chunk_j.embedding) > 0.65:
        graph.add_edge(i, j)

# Multi-hop search: DFS up to depth K
def multi_hop(start_node, query_embedding, hops=2):
    visited = set()
    results = []

    def dfs(node, depth):
        if depth == 0 or node in visited:
            return
        visited.add(node)
        results.append(node)
        for neighbor in graph.neighbors(node):
            dfs(neighbor, depth - 1)

    dfs(start_node, hops)
    return results
```

### Example
```
Query: "abandonment fear"

Single-hop (k=1):
  Direct matches: [chunk_5, chunk_8]

Multi-hop (k=2):
  chunk_5 → [chunk_12, chunk_3]
  chunk_8 → [chunk_15]
  chunk_12 → [chunk_7]

Result: [chunk_5, chunk_8, chunk_12, chunk_3, chunk_15, chunk_7]
Much richer context for reasoning!
```

## 5. Supporting + Opposing Retrieval

### Algorithm
```python
def retrieve_supporting_and_opposing(backstory_claim):

    # Supporting: High cosine similarity
    supporting = top_k(cosine_similarity(
        claim_embedding, 
        narrative_embeddings
    ), k=5)

    # Opposing: High similarity to semantic negation
    negated_claim = llm.negate(backstory_claim)
    opposing = top_k(cosine_similarity(
        negate_embedding,
        narrative_embeddings
    ), k=5)

    return supporting, opposing
```

### What LLM Sees
```
BACKSTORY CLAIMS:
- She feared abandonment
- She distrusted authority
- She was resourceful despite hardship

SUPPORTING CHUNKS (>0.85 similarity):
1. "Alice never allowed herself to depend on others" (0.91)
2. "She learned early to fend for herself" (0.89)
3. "Authority figures had never protected her" (0.87)

OPPOSING CHUNKS (high to negation, <0.35 to original):
1. "She entrusted her future to the council" (0.79 to negation)
2. "She readily accepted help from strangers" (0.82 to negation)

LLM REASONING:
"Mostly consistent. Supporting chunks confirm backstory.
However, some decisions (trusting council) contradict
the 'extreme distrust' claim. Predict: MOSTLY CONSISTENT (0.78)"
```

## 6. LLM Reasoning Layer

### Input to LLM
```
Backstory Claims: [structured list]
Supporting Chunks: [top 5 with similarity scores]
Opposing Chunks: [top 5 with similarity scores]
Character Context: [entities, temporal markers, causal chains]

Prompt: "Is this backstory consistent with the narrative?"
```

### LLM Output
```json
{
  "consistent": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "Multi-sentence explanation connecting evidence",
  "key_contradictions": ["list"],
  "causal_chain": "temporal sequence of events"
}
```

## Performance Characteristics

| Component | Time | Space | Quality |
|-----------|------|-------|---------|
| Dependency parsing | O(n) | O(n) | High |
| Context vectors | O(n·d) | O(n·d) | High |
| Semantic negation | O(k·m) | O(m) | Medium |
| Graph-RAG | O(n + e) | O(n + e) | Very High |
| LLM reasoning | O(l) | O(m) | Very High |

## Example End-to-End

```
INPUT CSV:
id | book_name | content
46 | In Search × Thalcave | Thalcave's people faded consistent

PROCESSING:
1. Parse book "In Search"
   - Extract sentences
   - Build dependency trees
   - Chunk intelligently: [chunk_0, chunk_1, ..., chunk_47]

2. Build context vectors
   - embed(chunk_0) → [1024]
   - context_vector(chunk_0) → [1024] with temporal/sentiment/causal

3. Build Graph-RAG
   - 47 chunks, connected by similarity edges
   - Ready for multi-hop queries

4. Extract backstory claim
   - "Thalcave's people faded" → claim_embedding, context_vector

5. Retrieve supporting/opposing
   - Supporting: [chunk_5, chunk_12, chunk_23] (0.89, 0.87, 0.85)
   - Opposing: [] (none found with high negation similarity)

6. LLM Reasoning
   - Input: claims + supporting (3) + opposing (0)
   - Output: consistent=True, confidence=0.92

OUTPUT CSV:
id | prediction | confidence | rationale
46 | 1 | 0.92 | Multiple narrative chunks (5, 12, 23) substantiate
   |   |      | Thalcave's role and his influence fading. No
   |   |      | contradicting evidence. Temporal arc aligns with
   |   |      | character development in chapters 7-12.
```

---

**This is production-grade narrative reasoning!** ✓
