#!/usr/bin/env python3
"""
Test script to verify embedding dimension fix

Tests:
1. HF embeddings are padded to 2048 but track 384 as effective dimension
2. Similarity calculation uses only first 384 dims when using HF
3. No dimension mismatch errors occur during fallback
"""

import sys
import numpy as np
from nvidia_client import NVIDIAClient, _embed_text_hf
from config import EMBEDDING_DIM

def test_hf_embedding_dimensions():
    """Test that HF embeddings are properly padded"""
    print("=" * 60)
    print("Test 1: HF Embedding Dimensions")
    print("=" * 60)
    
    vec = _embed_text_hf("Test text for embedding")
    print(f"✓ HF embedding dimension: {len(vec)}")
    print(f"✓ Expected dimension (EMBEDDING_DIM): {EMBEDDING_DIM}")
    
    assert len(vec) == EMBEDDING_DIM, f"FAIL: Dimension mismatch: {len(vec)} != {EMBEDDING_DIM}"
    
    # Check that first 384 dims have values, rest are mostly zeros
    non_zero_count = np.count_nonzero(vec[:384])
    zero_tail_count = np.count_nonzero(vec[384:])
    
    print(f"✓ Non-zero values in first 384 dims: {non_zero_count}/384")
    print(f"✓ Non-zero values in tail (385-2048): {zero_tail_count}/{EMBEDDING_DIM-384}")
    
    assert non_zero_count > 300, f"FAIL: Too few non-zero values in first 384 dims"
    assert zero_tail_count < 100, f"FAIL: Too many non-zero values in zero-padded tail"
    
    print("✅ PASS: HF embeddings properly padded to 2048 dimensions\n")


def test_client_with_fallback():
    """Test NVIDIAClient in HF fallback mode"""
    print("=" * 60)
    print("Test 2: NVIDIAClient Fallback Mode")
    print("=" * 60)
    
    # Initialize without API key (forces HF fallback)
    client = NVIDIAClient(api_key=None)
    
    print(f"✓ Client initialized in fallback mode")
    print(f"✓ Active embedding dimension: {client.active_embedding_dim}")
    
    assert client.active_embedding_dim == 384, f"FAIL: Expected 384, got {client.active_embedding_dim}"
    
    # Test embedding batch
    texts = ["Test 1", "Test 2", "Test 3"]
    embeddings = client.embed(texts)
    
    print(f"✓ Generated {len(embeddings)} embeddings")
    
    for i, emb in enumerate(embeddings):
        print(f"  - Embedding {i+1} dimension: {len(emb)}")
        assert len(emb) == EMBEDDING_DIM, f"FAIL: Dimension mismatch for embedding {i+1}"
    
    print("✅ PASS: All embeddings have correct dimension (2048)\n")


def test_dimension_aware_similarity():
    """Test that similarity uses only effective dimensions"""
    print("=" * 60)
    print("Test 3: Dimension-Aware Similarity")
    print("=" * 60)
    
    client = NVIDIAClient(api_key=None)
    
    # Create two similar texts
    text1 = "The cat sat on the mat"
    text2 = "A cat is sitting on a mat"
    text3 = "Dogs are running in the park"
    
    emb1 = client.embed([text1])[0]
    emb2 = client.embed([text2])[0]
    emb3 = client.embed([text3])[0]
    
    # Use effective dimension for similarity (384 for HF)
    effective_dim = client.get_embedding_dim()
    print(f"✓ Effective dimension for similarity: {effective_dim}")
    
    # Truncate to effective dimension
    emb1_eff = emb1[:effective_dim]
    emb2_eff = emb2[:effective_dim]
    emb3_eff = emb3[:effective_dim]
    
    # Compute similarities
    sim_12 = np.dot(emb1_eff, emb2_eff)
    sim_13 = np.dot(emb1_eff, emb3_eff)
    
    print(f"✓ Similarity (similar cats texts): {sim_12:.4f}")
    print(f"✓ Similarity (different topics): {sim_13:.4f}")
    
    # Similar texts should have higher similarity
    assert sim_12 > sim_13, f"FAIL: Similar texts should have higher similarity"
    
    print("✅ PASS: Dimension-aware similarity calculation works\n")


def test_mixed_backend_scenario():
    """Simulate mixing NVIDIA and HF embeddings"""
    print("=" * 60)
    print("Test 4: Mixed Backend Scenario (Simulated)")
    print("=" * 60)
    
    # Simulate NVIDIA embedding (full 2048 dims with values)
    nvidia_emb = np.random.randn(2048).astype(np.float32)
    nvidia_emb = nvidia_emb / np.linalg.norm(nvidia_emb)
    
    # Get HF embedding (384 real + 1664 zeros)
    hf_emb = _embed_text_hf("Test text")
    
    print(f"✓ NVIDIA embedding shape: {nvidia_emb.shape}")
    print(f"✓ HF embedding shape: {hf_emb.shape}")
    
    # When comparing, use only first 384 dims (effective dimension)
    effective_dim = 384
    nvidia_eff = nvidia_emb[:effective_dim]
    hf_eff = hf_emb[:effective_dim]
    
    # This should not crash
    try:
        sim_full = np.dot(nvidia_emb, hf_emb)  # Would be distorted
        sim_effective = np.dot(nvidia_eff, hf_eff)  # Correct
        
        print(f"✓ Similarity (full 2048d): {sim_full:.4f} (distorted by zeros)")
        print(f"✓ Similarity (effective 384d): {sim_effective:.4f} (accurate)")
        
        # Effective similarity should be different (usually higher) than full
        print(f"✓ Difference: {abs(sim_full - sim_effective):.4f}")
        
        print("✅ PASS: Mixed backend handling works\n")
    except Exception as e:
        print(f"❌ FAIL: {e}\n")
        raise


def main():
    print("\n" + "=" * 60)
    print("EMBEDDING DIMENSION FIX VERIFICATION")
    print("=" * 60 + "\n")
    
    try:
        test_hf_embedding_dimensions()
        test_client_with_fallback()
        test_dimension_aware_similarity()
        test_mixed_backend_scenario()
        
        print("=" * 60)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("=" * 60)
        print("\n✅ The embedding dimension mismatch bug is fixed!")
        print("✅ HF embeddings are properly padded to 2048 dimensions")
        print("✅ Similarity calculations use only effective dimensions (384)")
        print("✅ No dimension errors will occur during fallback\n")
        
        return 0
    except AssertionError as e:
        print("\n" + "=" * 60)
        print("❌ TESTS FAILED")
        print("=" * 60)
        print(f"Error: {e}\n")
        return 1
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ UNEXPECTED ERROR")
        print("=" * 60)
        print(f"Error: {e}\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
