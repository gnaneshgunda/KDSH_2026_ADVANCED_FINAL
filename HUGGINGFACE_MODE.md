# Advanced Narrative Consistency RAG - HuggingFace Mode

## Quick Setup: Use HuggingFace Embeddings

Since your NVIDIA API key isn't working, the system will automatically use HuggingFace embeddings.

### Step 1: Disable NVIDIA API

Edit your `.env` file and comment out or remove the NVIDIA_API_KEY:

```bash
# NVIDIA_API_KEY=nvapi-xxx...xxx
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
HF_TOKEN=your_hf_token_here  # Optional
```

**OR** just remove the API key value:

```bash
NVIDIA_API_KEY=
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
HF_TOKEN=your_hf_token_here  # Optional
```

### Step 2: Rebuild Index with HuggingFace Embeddings

For best results, rebuild the index so all embeddings are consistent:

```bash
# Delete old NVIDIA-based embeddings
rm db/*.pkl

# Run pipeline with HuggingFace
python pipeline.py
```

### Step 3: Verify Setup

When you run the pipeline, you should see:

```
2026-01-18 16:30:00 | nvidia_client | INFO | Using HuggingFace models for inference (embeddings: 384d)
2026-01-18 16:30:00 | nvidia_client | INFO | Loaded HuggingFace SentenceTransformer: all-MiniLM-L6-v2
```

---

## What Changes in HuggingFace Mode

### ✅ Embeddings
- **Model**: `all-MiniLM-L6-v2` (sentence-transformers)
- **Dimension**: 384 (padded to 2048 internally for compatibility)
- **Speed**: Runs locally, no API calls
- **Cost**: Free

### ⚠️ Chat/LLM (Claim Verification)
- **Fallback**: Uses stub heuristic (no LLM available locally)
- **Accuracy**: Lower than NVIDIA API but still functional
- **Verdicts**: Conservative (mostly SUPPORTED or NOT_MENTIONED)

### 📊 Expected Performance
- **Embedding quality**: Good (all-MiniLM-L6-v2 is well-tested)
- **Retrieval**: Accurate semantic search
- **Claim verification**: Basic heuristics only (no deep reasoning)

---

## Alternative: Use HuggingFace API for LLM

If you want better claim verification without NVIDIA, you can use HuggingFace Inference API:

### Option 1: Use HuggingFace Inference API

1. Get HF API token from https://huggingface.co/settings/tokens
2. Add to `.env`:
   ```bash
   HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
   ```

3. Modify `nvidia_client.py` to use HF Inference API (would require code changes)

### Option 2: Run Local LLM with Ollama

1. Install Ollama: https://ollama.ai/
2. Pull a model: `ollama pull llama3.2`
3. Modify code to use Ollama endpoint

**For now, the stub mode will work fine for testing!**

---

## Quick Commands

### Delete NVIDIA-based Cache
```bash
rm db/*.pkl
```

### Run with HuggingFace
```bash
python pipeline.py
```

### Check Logs
```bash
# Should show HF embeddings
grep "HuggingFace" logs.txt
```

---

## Summary

✅ **No code changes needed** - just disable NVIDIA API key  
✅ **HuggingFace embeddings work automatically**  
✅ **Delete `.pkl` files to rebuild with HF**  
⚠️ **Claim verification uses stub (reduced accuracy)**  

The pipeline will work end-to-end with HuggingFace embeddings!
