import os
import sys
import miditoolkit
import random

# Add the current directory to sys.path so we can import preprocessing
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing import MIDIPreprocessor

# Path to LMD dataset
DATA_DIR = '/Users/gordonkim/Downloads/lmd_full'
OUTPUT_DIR = './processed_data_sample'

def visualize_timeline(tokens, limit_events=50):
    """
    Parses tokens and prints a human-readable timeline to verify polyphony and dynamics.
    """
    token_list = tokens.split()
    
    current_tick = 0
    event_count = 0
    
    print(f"\n{'TICK':<8} | {'INSTRUMENT':<15} | {'NOTE':<10} | {'VEL':<5} | {'DUR':<5} | {'DETAILS'}")
    print("-" * 70)
    
    # State buffers for the current timestep
    current_events = []
    
    i = 0
    while i < len(token_list) and event_count < limit_events:
        token = token_list[i]
        
        if token.startswith("TimeShift_"):
            # Flush current events if we are moving time
            if current_events:
                for evt in current_events:
                    print(f"{current_tick:<8} | {evt}")
                current_events = []
                
            shift = int(token.split("_")[1])
            current_tick += shift
            # print(f"{current_tick:<8} | --- Time Shift (+{shift}) ---")
            i += 1
            
        elif token.startswith("Inst_"):
            # Start of a note definition
            # Expected format: Inst_X Pitch_Y Vel_Z Dur_W
            try:
                inst = token.split("_")[1]
                pitch = token_list[i+1].split("_")[1]
                vel = int(token_list[i+2].split("_")[1]) * 16
                dur = token_list[i+3].split("_")[1]
                
                # Note Name calculation
                p_val = int(pitch)
                note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                note_str = f"{note_names[p_val % 12]}{p_val // 12 - 1}"
                
                event_str = f"{inst:<15} | {note_str:<10} | {vel:<5} | {dur:<5} | MIDI {pitch}"
                current_events.append(event_str)
                
                event_count += 1
                i += 4 # Skip the 4 tokens we consumed
            except IndexError:
                print("Error: Incomplete token sequence at end.")
                break
        else:
            i += 1

    # Flush remaining
    for evt in current_events:
        print(f"{current_tick:<8} | {evt}")

def check_clipping(file_path, preprocessor):
    """
    Checks if any notes were clipped due to MAX_DURATION.
    """
    try:
        midi_obj = miditoolkit.MidiFile(file_path)
        original_tpb = midi_obj.ticks_per_beat
        if original_tpb == 0: return 0, 0
        
        clipped_count = 0
        total_notes = 0
        
        # Logic from preprocessing.py
        TICKS_PER_BEAT = 12
        
        for instrument in midi_obj.instruments:
            for note in instrument.notes:
                total_notes += 1
                start_tick = int(round((note.start / original_tpb) * TICKS_PER_BEAT))
                end_tick = int(round((note.end / original_tpb) * TICKS_PER_BEAT))
                duration = max(1, end_tick - start_tick)
                
                if duration > 96: # MAX_DURATION from preprocessing
                    clipped_count += 1
                    
        return clipped_count, total_notes
    except:
        return 0, 0

def run_deep_inspection():
    print(f"Scanning for MIDI files in {DATA_DIR}...")
    
    midi_files = []
    if os.path.exists(DATA_DIR):
        for root, _, filenames in os.walk(DATA_DIR):
            for filename in filenames:
                if filename.lower().endswith(('.mid', '.midi')):
                    midi_files.append(os.path.join(root, filename))
    
    if not midi_files:
        print("No files found.")
        return

    # Pick a random file
    sample_file = random.choice(midi_files)
    print(f"\nSelected File: {sample_file}")
    
    preprocessor = MIDIPreprocessor(DATA_DIR, OUTPUT_DIR)
    tokens = preprocessor.process_file(sample_file)
    
    if not tokens:
        print("Preprocessing failed.")
        return

    # 1. Visual Timeline
    print("\n=== VISUAL TIMELINE (First 50 Events) ===")
    print("Verifying Polyphony: Look for multiple instruments at the same TICK.")
    print("Verifying Velocity: Look for varying VEL values.")
    visualize_timeline(tokens, limit_events=50)
    
    # 2. Raw Token Stream
    print("\n=== RAW TOKEN STREAM (First 200 chars) ===")
    print(tokens[:200] + " ...")
    
    # 3. Clipping Analysis
    clipped, total = check_clipping(sample_file, preprocessor)
    print(f"\n=== DATA LOSS ANALYSIS ===")
    print(f"Total Notes: {total}")
    print(f"Clipped Durations (> 8 beats): {clipped}")
    if total > 0:
        print(f"Clipping Rate: {(clipped/total)*100:.2f}%")
    
    if clipped > 0:
        print("NOTE: Some notes were shortened to 96 ticks (8 beats). This is expected for a lightweight model.")

if __name__ == "__main__":
    run_deep_inspection()
