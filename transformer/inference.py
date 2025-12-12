import torch
import os
import argparse
from model import MusicGemma, MusicGemmaConfig
from diffusion import DiscreteDiffusion
from vocab import MusicVocab
from preprocessing import MIDIPreprocessor

# Configuration
CHECKPOINT_PATH = "checkpoints/model_latest.pt" # Default
OUTPUT_DIR = "generated"
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

def generate(num_samples=1, steps=100, temperature=1.0, output_prefix="gen"):
    print(f"Loading model from {CHECKPOINT_PATH}...")
    
    # 1. Load Vocab
    vocab = MusicVocab()
    vocab.build_vocab()
    
    # 2. Load Model
    config = MusicGemmaConfig(
        vocab_size=len(vocab.token_to_id),
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        max_position_embeddings=1024
    )
    
    transformer = MusicGemma(config)
    model = DiscreteDiffusion(
        model=transformer,
        vocab_size=config.vocab_size,
        mask_token_id=vocab.mask_token_id,
        pad_token_id=vocab.pad_token_id,
        timesteps=1000
    ).to(DEVICE)
    
    if os.path.exists(CHECKPOINT_PATH):
        state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print("Model loaded successfully.")
    else:
        print(f"Warning: Checkpoint {CHECKPOINT_PATH} not found. Generating with random weights.")

    model.eval()
    
    # 3. Generate
    print(f"Generating {num_samples} samples...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Shape: [batch, seq_len]
    shape = (num_samples, 1024)
    
    with torch.no_grad():
        # Sample tokens
        generated_ids = model.sample(shape, steps=steps, temperature=temperature)
        
    # 4. Decode to MIDI
    preprocessor = MIDIPreprocessor("", OUTPUT_DIR)
    
    for i in range(num_samples):
        ids = generated_ids[i].cpu().tolist()
        tokens = vocab.decode(ids)
        
        # Filter special tokens
        tokens = [t for t in tokens if t not in vocab.specials]
        token_str = " ".join(tokens)
        
        output_path = os.path.join(OUTPUT_DIR, f"{output_prefix}_{i+1}.mid")
        try:
            preprocessor.tokens_to_midi(token_str, output_path)
            print(f"Saved {output_path}")
        except Exception as e:
            print(f"Error saving {output_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--steps", type=int, default=50) # Faster sampling by default
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH)
    args = parser.parse_args()
    
    CHECKPOINT_PATH = args.checkpoint
    generate(args.num_samples, args.steps, args.temp)
