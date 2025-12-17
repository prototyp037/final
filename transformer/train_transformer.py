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

# CONFIGURATION

if os.name == 'nt':  # Windows
    DATA_DIR = "C:\\Users\\037co\\Downloads\\lmd_full"
    CHECKPOINT_DIR = "C:\\Users\\037co\\Desktop\\MelosonProject\\checkpoints"
    LOG_DIR = "C:\\Users\\037co\\Desktop\\MelosonProject\\logs"

    # Model Hyperparameters
    SEQ_LEN = 1024          # Sequence length for training
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 3
    GRAD_ACCUM_STEPS = 4    # Effective batch size = BATCH_SIZE * GRAD_ACCUM_STEPS
    STRIDE = SEQ_LEN // 2   # Sliding window stride (50% overlap)

else:  # macOS/Linux

    DATA_DIR = "/Users/gordonkim/Downloads/lmd_full" 
    CHECKPOINT_DIR = "/Users/gordonkim/Desktop/MelosonProject/checkpoints"
    LOG_DIR = "/Users/gordonkim/Desktop/MelosonProject/logs"

    # Model Hyperparameters
    SEQ_LEN = 1024          # Sequence length for training
    BATCH_SIZE = 2
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 5
    GRAD_ACCUM_STEPS = 4    # Effective batch size = BATCH_SIZE * GRAD_ACCUM_STEPS
    STRIDE = SEQ_LEN // 2   # Sliding window stride (50% overlap)

# Hardware
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = max(1, os.cpu_count() - 2) # Leave some CPUs for the OS/Main thread
BUFFER_SIZE = 128      # Number of sequences to keep in buffer

# DATA LOADING WORKER
def data_worker(file_queue, batch_queue, vocab_path, data_dir):
    """
    Worker process that:
    1. Fetches a MIDI file path from file_queue
    2. Processes it into tokens (No caching)
    3. Chunks it into sequences
    4. Puts sequences into batch_queue
    """
    # Re-initialize helper objects in the worker process (necessary for multiprocessing)
    vocab = MusicVocab()
    vocab.build_vocab()
    
    preprocessor = MIDIPreprocessor(data_dir, output_dir=None)
    
    while True:
        try:
            # Get a file to process
            # timeout=5 allows worker to exit if queue is empty for a while (end of epoch)
            file_path = file_queue.get(timeout=5) 
        except:
            break # Queue empty
            
        if file_path is None:
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
            
            # 3. Chunk into sequences (flyby method)
            # We create multiple sequences from one file
            # Stride = STRIDE (overlapping)
            
            total_len = len(token_ids)
            
            # If the file is empty or just has special tokens (unlikely due to check above), skip
            if total_len < 3: # <bos> <eos> and at least one event
                continue

            # If file is shorter than SEQ_LEN, take it all and pad
            if total_len <= SEQ_LEN:
                seq = token_ids + [vocab.pad_token_id] * (SEQ_LEN - total_len)
                batch_queue.put(seq)
            else:
                # Sliding window
                for i in range(0, total_len, STRIDE):
                    # Extract chunk
                    chunk = token_ids[i : i + SEQ_LEN]
                    
                    # Skip if chunk is too small (e.g. just <eos> left)
                    if len(chunk) < 10: 
                        continue
                        
                    # Pad if necessary (last chunk)
                    if len(chunk) < SEQ_LEN:
                        chunk = chunk + [vocab.pad_token_id] * (SEQ_LEN - len(chunk))
                        
                    batch_queue.put(chunk)
                    
        except Exception as e:
            # print(f"Error processing {file_path}: {e}")
            continue

import threading

# ... (imports)

# CONFIGURATION
# ... (existing config)
NUM_WORKERS = 4 # Reduced for stability
BUFFER_SIZE = 1000 # Reduced buffer

# ... (data_worker function remains same)

# TRAINING SCRIPT
def train():
    print(f"Initializing Training on {DEVICE}...")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # 1. Setup Model
    # ... (same as before)
    print("Building Vocab...")
    vocab = MusicVocab()
    vocab.build_vocab()
    

    #Increased model size
    print("Initializing Model...")
    config = MusicGemmaConfig(
        vocab_size=len(vocab.token_to_id),
        hidden_size=512,
        num_hidden_layers=6,
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
    # Limit file_queue size to prevent deadlock! (it kept freezing)
    file_queue = mp.Queue(maxsize=30000) 
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
        p.daemon = True
        p.start()
        workers.append(p)
        
    # Feeder Thread Function (SUPER IMPORTANT)
    def fill_file_queue(files, queue):
        for f in files:
            queue.put(f) # Blocks if queue is full
        # Signal end of epoch
        for _ in range(NUM_WORKERS):
            queue.put(None)

    # 5. Training Loop
    global_step = 0
    avg_batches_per_file = 2.0 # Initial estimate
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n=== Epoch {epoch+1}/{NUM_EPOCHS} ===")
        
        # Shuffle files
        random.shuffle(all_files)
        
        # Start Feeder Thread
        # Use a thread because queue.put might block, and we need the main thread to run the training loop (consuming batch_queue) to unblock the workers.
        feeder = threading.Thread(target=fill_file_queue, args=(all_files, file_queue))
        feeder.start()
            
        # Progress bar (Estimated Steps)
        estimated_steps = int((len(all_files) * avg_batches_per_file) / BATCH_SIZE)
        pbar = tqdm(total=estimated_steps, desc="Training Steps")
        
        model.train()
        epoch_loss = 0
        batch_buffer = []
        epoch_steps = 0
        
        # Just run until the feeder is done and queues are empty.
        
        while True:
            # Check if feeder is alive or queue has items
            if not feeder.is_alive() and file_queue.empty() and batch_queue.empty():
                break
                
            try:
                # Fetch sequence from buffer
                seq_list = batch_queue.get(timeout=1)
                batch_buffer.append(seq_list)
                
                # If we have enough for a batch
                if len(batch_buffer) >= BATCH_SIZE:
                    # ... (Training step)
                    batch_tensor = torch.tensor(batch_buffer, dtype=torch.long).to(DEVICE)
                    batch_buffer = [] 
                    
                    loss = model.compute_loss(batch_tensor)
                    loss = loss / GRAD_ACCUM_STEPS
                    loss.backward()
                    
                    epoch_loss += loss.item() * GRAD_ACCUM_STEPS
                    
                    if (global_step + 1) % GRAD_ACCUM_STEPS == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                        
                        writer.add_scalar("Loss/train", loss.item() * GRAD_ACCUM_STEPS, global_step)
                        pbar.set_description(f"Loss: {loss.item() * GRAD_ACCUM_STEPS:.4f}")
                    
                    global_step += 1
                    epoch_steps += 1
                    pbar.update(1) 
                    
                    # Extend pbar if we underestimated
                    if epoch_steps >= pbar.total:
                        pbar.total += 100
                        pbar.refresh()
                    
                    if global_step % 1000 == 0:
                        print(f"\nSaving checkpoint at step {global_step}...")
                        save_path = os.path.join(CHECKPOINT_DIR, f"model_step_{global_step}.pt")
                        torch.save(model.state_dict(), save_path)
                        
            except Exception as e:
                # Timeout
                continue
        
        feeder.join()
        pbar.close()
        
        # Update estimate for next epoch (Moving Average)
        actual_batches_per_file = (epoch_steps * BATCH_SIZE) / len(all_files)
        avg_batches_per_file = 0.9 * avg_batches_per_file + 0.1 * actual_batches_per_file
        
        # End of Epoch
        avg_loss = epoch_loss / (epoch_steps + 1) if epoch_steps > 0 else 0
        print(f"Epoch {epoch+1} Finished. Avg Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "model_latest.pt"))

    # Cleanup
    for p in workers:
        p.terminate()

if __name__ == "__main__":
    # Required for multiprocessing on some platforms
    mp.set_start_method('spawn', force=True)
    train()