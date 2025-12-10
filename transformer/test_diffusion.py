import torch
import sys
import os
import matplotlib.pyplot as plt

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import MusicGemma, MusicGemmaConfig
from vocab import MusicVocab
from diffusion import DiscreteDiffusion

def test_diffusion_schedule():
    print("\n=== Testing Diffusion Schedule ===")
    # Mock model not needed for schedule
    diffusion = DiscreteDiffusion(None, 100, 1, 0, timesteps=100)
    
    # Test points
    t_start = torch.tensor([0])
    t_mid = torch.tensor([50])
    t_end = torch.tensor([100])
    
    prob_start = diffusion.get_mask_prob(t_start.float() / 100).item()
    prob_mid = diffusion.get_mask_prob(t_mid.float() / 100).item()
    prob_end = diffusion.get_mask_prob(t_end.float() / 100).item()
    
    print(f"t=0   Mask Prob: {prob_start:.4f} (Expected ~0.0)")
    print(f"t=50  Mask Prob: {prob_mid:.4f}")
    print(f"t=100 Mask Prob: {prob_end:.4f} (Expected ~1.0)")
    
    assert 0 <= prob_start < 0.1, "Start probability should be low"
    assert 0.9 < prob_end <= 1.0, "End probability should be high"
    print("✅ Schedule looks correct.")

def test_forward_diffusion():
    print("\n=== Testing Forward Diffusion (q_sample) ===")
    vocab_size = 50
    mask_id = 49
    pad_id = 0
    seq_len = 20
    batch_size = 2
    
    diffusion = DiscreteDiffusion(None, vocab_size, mask_id, pad_id, timesteps=10)
    
    # Create dummy data (random tokens 1-48)
    x_start = torch.randint(1, mask_id, (batch_size, seq_len))
    
    # Test t=10 (Full Mask)
    t = torch.full((batch_size,), 10)
    x_t, mask = diffusion.q_sample(x_start, t, mask_strategy='random')
    
    masked_ratio = (x_t == mask_id).float().mean().item()
    print(f"t=10 Masked Ratio: {masked_ratio:.2f}")
    assert masked_ratio > 0.9, "Should be heavily masked at max timestep"
    
    # Test Span Masking
    print("Testing Span Masking...")
    t_mid = torch.full((batch_size,), 5) # 50%
    x_t_span, mask_span = diffusion.q_sample(x_start, t_mid, mask_strategy='span')
    
    # Check if we have contiguous masks
    # Just print a sample to visually verify
    print("Original:", x_start[0].tolist())
    print("Span Masked:", x_t_span[0].tolist())
    print("✅ Forward diffusion runs without error.")

def test_loss_computation():
    print("\n=== Testing Loss Computation ===")
    vocab_size = 100
    mask_id = 99
    pad_id = 0
    
    # Setup small model
    config = MusicGemmaConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        head_dim=16,
        max_position_embeddings=64
    )
    model = MusicGemma(config)
    diffusion = DiscreteDiffusion(model, vocab_size, mask_id, pad_id, timesteps=10)
    
    x_start = torch.randint(1, mask_id, (4, 32)) # Batch 4, Seq 32
    
    loss = diffusion.compute_loss(x_start)
    print(f"Loss: {loss.item()}")
    
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"
    
    loss.backward()
    print("✅ Backward pass successful.")

def test_sampling():
    print("\n=== Testing Sampling (Reverse Process) ===")
    vocab_size = 100
    mask_id = 99
    pad_id = 0
    
    config = MusicGemmaConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        head_dim=16,
        max_position_embeddings=64
    )
    model = MusicGemma(config)
    diffusion = DiscreteDiffusion(model, vocab_size, mask_id, pad_id, timesteps=5)
    
    shape = (2, 16) # Batch 2, Seq 16
    
    print("Starting sampling loop...")
    samples = diffusion.sample(shape, steps=5)
    
    print(f"Sample shape: {samples.shape}")
    print("Sample output:", samples[0].tolist())
    
    assert samples.shape == shape
    # Check if any masks remain (should be none or very few if steps are sufficient)
    masks_remaining = (samples == mask_id).sum().item()
    print(f"Masks remaining: {masks_remaining}")
    
    print("✅ Sampling successful.")

if __name__ == "__main__":
    test_diffusion_schedule()
    test_forward_diffusion()
    test_loss_computation()
    test_sampling()
