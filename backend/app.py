from flask import Flask, send_from_directory, jsonify, request
from flask_cors import CORS
import os
import sys
import torch

# Add transformer directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../transformer'))

from model import MusicGemma, MusicGemmaConfig
from diffusion import DiscreteDiffusion
from vocab import MusicVocab
from preprocessing import MIDIPreprocessor

app = Flask(__name__, static_folder='../frontend')
CORS(app)

# global states/variables
model = None
vocab = None
preprocessor = None
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "/Users/gordonkim/Downloads/model_step_33000.pt"

def load_model():
    global model, vocab, preprocessor
    print(f"Loading model on {DEVICE}...")
    
    vocab = MusicVocab()
    vocab.build_vocab()
    
    preprocessor = MIDIPreprocessor("", "") # Dirs not needed for conversion
    
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
        print(f"Warning: Checkpoint {CHECKPOINT_PATH} not found.")
        #we use untrained/random weight as a placeholder for now
    
    model.eval()

#needed for flask serving
@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

@app.route('/api/generate', methods=['POST'])
def generate_music():
    if model is None:
        load_model()
        
    try:
        data = request.json
        current_notes = data.get('notes', [])
        
        #Convert Context to Tokens
        token_str = preprocessor.json_to_tokens(current_notes)
        token_ids = vocab.encode(token_str.split())
        
        #Prepare Input Tensor
        seq_len = 1024
        input_ids = torch.full((1, seq_len), vocab.pad_token_id, dtype=torch.long).to(DEVICE)
        
        known_len = min(len(token_ids), seq_len)
        if known_len > 0:
            input_ids[0, :known_len] = torch.tensor(token_ids[:known_len]).to(DEVICE)
            
        #Create Mask (for continuation generation)
        # Mask everything after the known notes
        mask = torch.zeros((1, seq_len), dtype=torch.bool).to(DEVICE)
        mask[0, known_len:] = True
        
        # If context is full, we can't generate more. 
        #For now, just return if full.
        if known_len >= seq_len:
            return jsonify({"status": "error", "message": "Context too long"})
            
        # Run Inference
        print(f"Generating with context length {known_len}...")
        with torch.no_grad():
            output_ids = model.infill(input_ids, mask, steps=50)
            
        # Decode
        out_tokens = vocab.decode(output_ids[0].cpu().tolist())
        # Filter specials
        out_tokens = [t for t in out_tokens if t not in vocab.specials]
        
        # Convert to JSON
        new_notes = preprocessor.tokens_to_json(" ".join(out_tokens))
        
        print(f"Generated {len(new_notes)} notes.")
        
        return jsonify({
            "status": "success",
            "notes": new_notes
        })
        
    except Exception as e:
        print(f"Generation Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/improve', methods=['POST'])
def improve_music():
    if model is None:
        load_model()
        
    try:
        data = request.json
        current_notes = data.get('notes', [])
        selection_range = data.get('range', {})
        start_time = selection_range.get('start', 0)
        end_time = selection_range.get('end', 0)
        
        print(f"Improving range: {start_time} to {end_time}")
        
        # Convert Context to Tokens with Mask
        token_str, mask_list = preprocessor.json_to_tokens_with_mask(current_notes, start_time, end_time)
        token_ids = vocab.encode(token_str.split())
        
        # Prepare Input Tensor
        seq_len = 1024
        input_ids = torch.full((1, seq_len), vocab.pad_token_id, dtype=torch.long).to(DEVICE)
        mask_tensor = torch.zeros((1, seq_len), dtype=torch.bool).to(DEVICE)
        
        # Fill tensors
        length = min(len(token_ids), seq_len)
        if length > 0:
            input_ids[0, :length] = torch.tensor(token_ids[:length]).to(DEVICE)
            # Fill mask from the boolean list we generated
            mask_tensor[0, :length] = torch.tensor(mask_list[:length], dtype=torch.bool).to(DEVICE)
            
        # Run Inference (Infill)
        print(f"Infilling with {mask_tensor.sum().item()} masked tokens...")
        with torch.no_grad():
            # We use the same infill method, but now the mask is in the middle!
            output_ids = model.infill(input_ids, mask_tensor, steps=50)
            
        # Decode
        out_tokens = vocab.decode(output_ids[0].cpu().tolist())
        out_tokens = [t for t in out_tokens if t not in vocab.specials]
        
        # Convert to JSON
        new_notes = preprocessor.tokens_to_json(" ".join(out_tokens))
        
        print(f"Improved to {len(new_notes)} notes.")
        
        return jsonify({
            "status": "success",
            "notes": new_notes
        })
        
    except Exception as e:
        print(f"Improve Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    print("Starting server at http://localhost:5000")
    # Pre-load model
    # load_model() 
    app.run(debug=True, port=5000)

