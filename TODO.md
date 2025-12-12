# Project Todo List

## Frontend
- [x] Basic Piano Roll (Canvas)
- [x] Multi-track support
- [x] Snap-to-grid
- [x] Undo/Redo
- [x] Playback with Tone.js
- [ ] Connect "AI Fill" button to Backend API

## Backend (Flask)
- [x] Basic Flask App serving static files
- [ ] API Endpoint for Generation (`/api/generate`)
- [ ] API Endpoint for In-filling (`/api/infill`)

## AI Model
- [x] Data Preprocessing (MIDI to Tokens)
- [x] Transformer Model Architecture (MusicGemma)
- [x] Discrete Diffusion Logic
- [x] Training Script (`train_transformer.py`)
- [~] Train Model on LMD Dataset (Running...)
- [x] Inference Script (`inference.py`)

## Integration
- [ ] Connect Frontend to Backend API
- [ ] Implement "AI Fill" logic in Backend to call Model
