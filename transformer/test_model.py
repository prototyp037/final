import torch
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import MusicGemma, MusicGemmaConfig
from vocab import MusicVocab

def test_model_initialization_and_forward():
    print("=== Testing MusicGemma Model ===")
    
    # 1. Setup Vocab
    vocab = MusicVocab()
    vocab.build_vocab()
    vocab_size = len(vocab.token_to_id)
    print(f"Vocab Size: {vocab_size}")

    # 2. Configure Model (Small for testing)
    config = MusicGemmaConfig(
        vocab_size=vocab_size,
        hidden_size=64,        # Small hidden size
        intermediate_size=256,
        num_hidden_layers=2,   # Few layers
        num_attention_heads=4,
        num_key_value_heads=2, # GQA
        head_dim=16,
        max_position_embeddings=128,
        pad_token_id=vocab.pad_token_id
    )
    
    print("Initializing model...")
    model = MusicGemma(config)
    print("Model initialized successfully.")
    
    # 3. Create Dummy Input
    batch_size = 2
    seq_len = 32
    
    # Random tokens within vocab range
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Create a dummy attention mask (optional, but good to test)
    # 1 for keep, 0 for mask (though our model expects additive mask usually, 
    # let's check how we implemented it. 
    # In model.py: attn_weights = attn_weights + attention_mask
    # So mask should be 0 for keep, -inf for mask.
    
    # Let's try without mask first
    print(f"Input shape: {input_ids.shape}")
    
    # 4. Forward Pass
    print("Running forward pass...")
    try:
        logits = model(input_ids)
        print("Forward pass successful.")
        print(f"Output logits shape: {logits.shape}")
        
        # 5. Checks
        expected_shape = (batch_size, seq_len, vocab_size)
        assert logits.shape == expected_shape, f"Expected shape {expected_shape}, got {logits.shape}"
        assert not torch.isnan(logits).any(), "Output contains NaNs!"
        assert not torch.isinf(logits).any(), "Output contains Infs!"
        print("✅ Shape check passed.")
        print("✅ NaN/Inf check passed.")
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        raise e

def test_attention_mask():
    print("\n=== Testing Attention Mask ===")
    vocab = MusicVocab()
    vocab.build_vocab()
    
    config = MusicGemmaConfig(
        vocab_size=len(vocab.token_to_id),
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        head_dim=16
    )
    model = MusicGemma(config)
    
    batch_size = 1
    seq_len = 10
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    
    # Create an additive mask: 0 for keep, -1e9 for mask
    # Let's mask the last 5 tokens
    mask = torch.zeros((batch_size, 1, 1, seq_len))
    mask[:, :, :, 5:] = -1e9
    
    logits = model(input_ids, attention_mask=mask)
    print(f"Masked forward pass successful. Shape: {logits.shape}")
    
    # We can't easily verify the masking effect without inspecting attention weights,
    # but if it runs without error, dimensions are likely correct.

if __name__ == "__main__":
    test_model_initialization_and_forward()
    test_attention_mask()
