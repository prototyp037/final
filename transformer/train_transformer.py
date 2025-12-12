import os
import time
import glob
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Local imports
from model import MusicGemma, MusicGemmaConfig
from diffusion import DiscreteDiffusion
from vocab import MusicVocab
from preprocessing import MIDIPreprocessor

# ==========================================
# CONFIGURATION
# ==========================================

if os.name == 'nt':  # Windows
    DATA_DIR = "C:\\Users\\037co\\Downloads\\lmd_full" # Change this to your dataset path

else:  # macOS/Linux

    DATA_DIR = "/Users/gordonkim/Downloads/lmd_full" # Change this to your dataset path
    CHECKPOINT_DIR = "/Users/gordonkim/Desktop/MelosonProject/checkpoints"
    LOG_DIR = "/Users/gordonkim/Desktop/MelosonProject/logs"

# Model Hyperparameters
SEQ_LEN = 1024          # Sequence length for training
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
GRAD_ACCUM_STEPS = 4    # Effective batch size = BATCH_SIZE * GRAD_ACCUM_STEPS

# Hardware
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = max(1, os.cpu_count() - 2) # Leave some CPUs for the OS/Main thread
BUFFER_SIZE = 2000      # Number of sequences to keep in buffer

# ==========================================
# DATA LOADING WORKER
# ==========================================
def data_worker(file_queue, batch_queue, vocab_path, data_dir):
    """
    Worker process that:
    1. Fetches a MIDI file path from file_queue
    2. Processes it into tokens (No caching)
    3. Chunks it into sequences
    4. Puts sequences into batch_queue
    """
    # Re-initialize helper objects in the worker process
    vocab = MusicVocab()
    vocab.build_vocab() # Deterministic build
    
    preprocessor = MIDIPreprocessor(data_dir, output_dir=None)
    
    while True:
        try:
            # Get a file to process
            # timeout=5 allows worker to exit if queue is empty for a while (end of epoch)
            file_path = file_queue.get(timeout=5) 
        except:
            break # Queue empty
            
        if file_path is None: # Sentinel
            break
            
        try:
            # 1. Process MIDI to Token String
            # This is the heavy CPU operation
            token_str = preprocessor.process_file(file_path)
            if not token_str:
                continue
                
            # 2. Encode to IDs
            tokens = token_str.split()
            token_ids = vocab.encode(tokens)
            
            # 3. Chunk into sequences (Flyby method)
            # We create multiple sequences from one file
            # Stride = SEQ_LEN (non-overlapping) or SEQ_LEN // 2 (overlapping)
            stride = SEQ_LEN 
            
            for i in range(0, len(token_ids) - SEQ_LEN + 1, stride):
                seq = token_ids[i : i + SEQ_LEN]
                
                # Verify length (just in case)
                if len(seq) == SEQ_LEN:
                    # 4. Push to Buffer
                    # This blocks if buffer is full, effectively pausing this worker
                    # until the GPU consumes more data.
                    batch_queue.put(seq)
                    
        except Exception as e:
            # print(f"Error processing {file_path}: {e}")
            continue

# ==========================================
# TRAINING SCRIPT
# ==========================================
def train():
    print(f"Initializing Training on {DEVICE}...")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # 1. Setup Model
    print("Building Vocab...")
    vocab = MusicVocab()
    vocab.build_vocab()
    
    print("Initializing Model...")
    config = MusicGemmaConfig(
        vocab_size=len(vocab.token_to_id),
        hidden_size=512,
        num_hidden_layers=6,       # Reduced for faster iteration/stability
        num_attention_heads=8,
        max_position_embeddings=SEQ_LEN
    )
    
    transformer = MusicGemma(config)
    model = DiscreteDiffusion(
        model=transformer,
        vocab_size=config.vocab_size,
        mask_token_id=vocab.mask_token_id,
        pad_token_id=vocab.pad_token_id,
        timesteps=1000
    ).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    writer = SummaryWriter(LOG_DIR)
    
    # 2. Prepare Data Queues
    # file_queue: Holds paths to MIDI files
    # batch_queue: Holds processed tensor sequences (ready for GPU)
    file_queue = mp.Queue()
    batch_queue = mp.Queue(maxsize=BUFFER_SIZE)
    
    # 3. Find Files
    print(f"Scanning for MIDI files in {DATA_DIR}...")
    all_files = []
    for root, _, filenames in os.walk(DATA_DIR):
        for f in filenames:
            if f.lower().endswith(('.mid', '.midi')):
                all_files.append(os.path.join(root, f))
    
    print(f"Found {len(all_files)} files.")
    random.shuffle(all_files)
    
    if len(all_files) == 0:
        print("No MIDI files found! Please check DATA_DIR.")
        return

    # 4. Start Workers
    print(f"Starting {NUM_WORKERS} data workers...")
    workers = []
    for _ in range(NUM_WORKERS):
        p = mp.Process(
            target=data_worker,
            args=(file_queue, batch_queue, None, DATA_DIR)
        )
        p.daemon = True # Kill workers if main process dies
        p.start()
        workers.append(p)
        
    # 5. Training Loop
    global_step = 0
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n=== Epoch {epoch+1}/{NUM_EPOCHS} ===")
        
        # Refill file queue for this epoch
        random.shuffle(all_files)
        for f in all_files:
            file_queue.put(f)
            
        # Progress bar
        pbar = tqdm(total=len(all_files)) # Approx progress (files, not batches)
        
        model.train()
        epoch_loss = 0
        batch_buffer = []
        
        while True:
            # Check if we are done with files and buffer is empty
            if file_queue.empty() and batch_queue.empty():
                break
                
            try:
                # Fetch sequence from buffer
                # timeout=1 prevents infinite blocking if workers die
                seq_list = batch_queue.get(timeout=1)
                batch_buffer.append(seq_list)
                
                # If we have enough for a batch
                if len(batch_buffer) >= BATCH_SIZE:
                    # Convert to Tensor and move to GPU
                    batch_tensor = torch.tensor(batch_buffer, dtype=torch.long).to(DEVICE)
                    batch_buffer = [] # Clear buffer
                    
                    # Forward Pass
                    loss = model.compute_loss(batch_tensor)
                    loss = loss / GRAD_ACCUM_STEPS
                    
                    # Backward Pass
                    loss.backward()
                    
                    epoch_loss += loss.item() * GRAD_ACCUM_STEPS
                    
                    # Optimizer Step
                    if (global_step + 1) % GRAD_ACCUM_STEPS == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                        
                        # Logging
                        writer.add_scalar("Loss/train", loss.item() * GRAD_ACCUM_STEPS, global_step)
                        pbar.set_description(f"Loss: {loss.item() * GRAD_ACCUM_STEPS:.4f}")
                    
                    global_step += 1
                    pbar.update(1) # Update roughly per batch (not per file, but good enough visual)
                    
                    # Save Checkpoint
                    if global_step % 1000 == 0:
                        save_path = os.path.join(CHECKPOINT_DIR, f"model_step_{global_step}.pt")
                        torch.save(model.state_dict(), save_path)
                        
            except Exception as e:
                # Queue might be empty temporarily while workers catch up
                if file_queue.empty() and batch_queue.empty():
                    break
                continue
                
        # End of Epoch
        avg_loss = epoch_loss / (global_step + 1) # Approx
        print(f"Epoch {epoch+1} Finished. Avg Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "model_latest.pt"))

    # Cleanup
    for p in workers:
        p.terminate()

if __name__ == "__main__":
    # Required for multiprocessing on some platforms
    mp.set_start_method('spawn', force=True)
    train()