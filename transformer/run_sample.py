import os
import sys
import miditoolkit
import random
import time
from tqdm import tqdm

# Add the current directory to sys.path so we can import preprocessing
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing import MIDIPreprocessor

# Path to LMD dataset
DATA_DIR = '/Users/gordonkim/Downloads/lmd_full'
OUTPUT_DIR = './processed_data_sample'

def analyze_file(file_path, preprocessor):
    """
    Process a single file and return analysis results.
    """
    try:
        # 1. Process to Tokens
        tokens = preprocessor.process_file(file_path)
        if not tokens:
            return {'status': 'skipped', 'reason': 'Preprocessing returned None'}

        # 2. Decode back to MIDI
        midi_obj = preprocessor.tokens_to_midi(tokens)
        
        # 3. Compare
        try:
            original_midi = miditoolkit.MidiFile(file_path)
        except:
            return {'status': 'error', 'reason': 'Original MIDI corrupt'}
        
        # Calculate original duration in beats
        orig_ticks = original_midi.max_tick
        orig_tpb = original_midi.ticks_per_beat
        if orig_tpb == 0: return {'status': 'skipped', 'reason': 'Original TPB is 0'}
        orig_beats = orig_ticks / orig_tpb
        
        # Calculate reconstructed duration
        recon_ticks = midi_obj.max_tick
        # Fallback: Calculate max tick manually
        if recon_ticks == 0:
            all_notes = [n for i in midi_obj.instruments for n in i.notes]
            if all_notes:
                recon_ticks = max(n.end for n in all_notes)
        
        recon_tpb = midi_obj.ticks_per_beat
        recon_beats = recon_ticks / recon_tpb
        
        # Note Counts
        orig_note_count = sum(len(i.notes) for i in original_midi.instruments)
        recon_note_count = sum(len(i.notes) for i in midi_obj.instruments)
        
        # Diff
        beat_diff = abs(orig_beats - recon_beats)
        note_diff = abs(orig_note_count - recon_note_count)
        
        # Success Criteria
        # Duration within 2 beats (allow for some quantization drift at end)
        # Note count within 5% (some short notes might be filtered)
        duration_ok = beat_diff < 2.0
        notes_ok = note_diff == 0 # Should be exact for our logic
        
        status = 'success' if (duration_ok and notes_ok) else 'warning'
        
        return {
            'status': status,
            'file': os.path.basename(file_path),
            'orig_beats': orig_beats,
            'recon_beats': recon_beats,
            'beat_diff': beat_diff,
            'orig_notes': orig_note_count,
            'recon_notes': recon_note_count,
            'note_diff': note_diff
        }
        
    except Exception as e:
        return {'status': 'error', 'reason': str(e)}

def run_batch_analysis(num_files=50):
    print(f"Scanning for MIDI files in {DATA_DIR}...")
    
    midi_files = []
    if os.path.exists(DATA_DIR):
        for root, _, filenames in os.walk(DATA_DIR):
            for filename in filenames:
                if filename.lower().endswith(('.mid', '.midi')):
                    midi_files.append(os.path.join(root, filename))
    else:
        print(f"Directory {DATA_DIR} does not exist.")
        return

    if not midi_files:
        print("No MIDI files found.")
        return

    print(f"Found {len(midi_files)} files. Selecting {num_files} random files for analysis...")
    
    # Select random files
    selected_files = random.sample(midi_files, min(num_files, len(midi_files)))
    
    preprocessor = MIDIPreprocessor(DATA_DIR, OUTPUT_DIR)
    
    results = []
    
    print("\nStarting Analysis...")
    for file_path in tqdm(selected_files):
        res = analyze_file(file_path, preprocessor)
        results.append(res)
        
    # Summary
    success = [r for r in results if r['status'] == 'success']
    warnings = [r for r in results if r['status'] == 'warning']
    errors = [r for r in results if r['status'] == 'error']
    skipped = [r for r in results if r['status'] == 'skipped']
    
    print("\n" + "="*50)
    print("ANALYSIS SUMMARY")
    print("="*50)
    print(f"Total Processed: {len(results)}")
    print(f"Success: {len(success)} ({len(success)/len(results)*100:.1f}%)")
    print(f"Warnings: {len(warnings)}")
    print(f"Errors: {len(errors)}")
    print(f"Skipped: {len(skipped)}")
    
    if warnings:
        print("\n--- WARNING DETAILS (Top 10) ---")
        print(f"{'File':<30} | {'Beat Diff':<10} | {'Note Diff':<10}")
        print("-" * 60)
        for w in warnings[:10]:
            print(f"{w['file'][:30]:<30} | {w['beat_diff']:<10.2f} | {w['note_diff']:<10}")
            
    if success:
        avg_beat_diff = sum(r['beat_diff'] for r in success) / len(success)
        print(f"\nAverage Beat Diff (Successes): {avg_beat_diff:.4f}")

if __name__ == "__main__":
    # Run batch analysis on 50 files
    run_batch_analysis(50)
