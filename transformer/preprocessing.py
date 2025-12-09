import os
import miditoolkit
import numpy as np
from tqdm import tqdm
import math

# ==========================================
# CONFIGURATION
# ==========================================
# Resolution: Ticks per beat.
# We use a "Performance" representation (TimeShift based) instead of "REMI" (Bar/Pos).
# This allows for ANY time signature and complex polyphony.
# 12 ticks per beat = 48 ticks per bar (in 4/4).
# Supports triplets (4 ticks) and 16ths (3 ticks).
TICKS_PER_BEAT = 12 

# Max TimeShift to include in vocabulary.
# If a pause is longer than this, we stack multiple TimeShift tokens.
# 4 beats * 12 ticks = 48 ticks.
MAX_TIME_SHIFT = 48

# Instrument Classes (Simplified General MIDI)
# We map 128 programs to fewer categories to reduce vocabulary size.
# Names must not contain spaces for tokenization safety.
INSTRUMENT_CLASSES = [
    (0, 8, 'Piano'),
    (8, 16, 'ChromaticPercussion'),
    (16, 24, 'Organ'),
    (24, 32, 'Guitar'),
    (32, 40, 'Bass'),
    (40, 48, 'Strings'),
    (48, 56, 'Ensemble'),
    (56, 64, 'Brass'),
    (64, 72, 'Reed'),
    (72, 80, 'Pipe'),
    (80, 88, 'SynthLead'),
    (88, 96, 'SynthPad'),
    (96, 104, 'SynthEffects'),
    (104, 112, 'Ethnic'),
    (112, 120, 'Percussive'),
    (120, 128, 'SoundEffects')
]

def get_instrument_class(program, is_drum):
    if is_drum:
        return 'Drums'
    for start, end, name in INSTRUMENT_CLASSES:
        if start <= program < end:
            return name
    return 'Unknown'

class MIDIPreprocessor:
    def __init__(self, data_dir, output_dir):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def quantize_time(self, time, ticks_per_beat, original_ticks_per_beat):
        # Convert original time to beats, then to quantized ticks
        # Note: This assumes constant tempo for the duration of the tick.
        # For MIDI files, ticks are already tempo-relative (musical time).
        if original_ticks_per_beat is None or original_ticks_per_beat == 0:
             return 0
        beats = time / original_ticks_per_beat
        quantized_ticks = int(round(beats * ticks_per_beat))
        return quantized_ticks

    def process_file(self, file_path):
        try:
            midi_obj = miditoolkit.MidiFile(file_path)
        except Exception as e:
            return None

        # Note: We REMOVED the 4/4 time signature check.
        # This implementation is now Time Signature Agnostic.
        
        notes = []
        original_ticks_per_beat = midi_obj.ticks_per_beat

        for instrument in midi_obj.instruments:
            inst_class = get_instrument_class(instrument.program, instrument.is_drum)
            
            for note in instrument.notes:
                # Quantize
                start_tick = self.quantize_time(note.start, TICKS_PER_BEAT, original_ticks_per_beat)
                end_tick = self.quantize_time(note.end, TICKS_PER_BEAT, original_ticks_per_beat)
                duration = max(1, end_tick - start_tick) 
                
                # Limit duration to reduce vocab (e.g., 8 beats = 96 ticks)
                if duration > 96:
                    duration = 96

                notes.append({
                    'start': start_tick,
                    'pitch': note.pitch,
                    'velocity': note.velocity, 
                    'duration': duration,
                    'instrument': inst_class
                })

        # Sort notes by start time, then pitch
        notes.sort(key=lambda x: (x['start'], x['pitch'], x['instrument']))

        if not notes:
            return None

        # Convert to "Performance" Token Sequence (TimeShift based)
        # [TimeShift] [Inst] [Pitch] [Vel] [Dur]
        
        events = []
        current_tick = 0
        
        for note in notes:
            # Calculate Delta
            delta = note['start'] - current_tick
            
            # If delta is positive, insert TimeShift tokens
            if delta > 0:
                # Handle large gaps by stacking max shifts
                while delta > MAX_TIME_SHIFT:
                    events.append(f"TimeShift_{MAX_TIME_SHIFT}")
                    delta -= MAX_TIME_SHIFT
                
                if delta > 0:
                    events.append(f"TimeShift_{delta}")
                
                current_tick = note['start']
            
            # Note Events
            events.append(f"Inst_{note['instrument']}")
            events.append(f"Pitch_{note['pitch']}")
            events.append(f"Vel_{note['velocity'] // 16}") # Quantize velocity
            events.append(f"Dur_{note['duration']}")

        return " ".join(events)

    def tokens_to_midi(self, token_str, output_path=None):
        """
        Converts a token string back to a miditoolkit.MidiFile object.
        Useful for validation and generation.
        """
        tokens = token_str.split()
        
        # Create a new MIDI object
        midi_obj = miditoolkit.MidiFile()
        midi_obj.ticks_per_beat = TICKS_PER_BEAT
        
        # Prepare tracks
        # Map class name -> (program, is_drum)
        # We pick the first program in the range for the class
        class_to_program = {}
        for start, end, name in INSTRUMENT_CLASSES:
            class_to_program[name] = (start, False)
        class_to_program['Drums'] = (0, True)
        
        # Dictionary to hold Instrument objects: key = class_name
        tracks = {}
        
        current_tick = 0
        current_inst = None
        current_pitch = None
        current_vel = None
        
        for token in tokens:
            if token.startswith("TimeShift_"):
                shift = int(token.split("_")[1])
                current_tick += shift
                
            elif token.startswith("Inst_"):
                inst_name = token.split("_")[1]
                current_inst = inst_name
                
                # Create track if not exists
                if inst_name not in tracks:
                    program, is_drum = class_to_program.get(inst_name, (0, False))
                    track = miditoolkit.Instrument(program=program, is_drum=is_drum, name=inst_name)
                    tracks[inst_name] = track
                    midi_obj.instruments.append(track)
                    
            elif token.startswith("Pitch_"):
                current_pitch = int(token.split("_")[1])
                
            elif token.startswith("Vel_"):
                current_vel = int(token.split("_")[1]) * 16
                # Ensure non-zero velocity if we want it to be audible, though 0 is technically NoteOff
                if current_vel == 0: current_vel = 16 
                
            elif token.startswith("Dur_"):
                duration = int(token.split("_")[1])
                
                # We have all info for a note now
                if current_inst and current_pitch is not None and current_vel is not None:
                    note = miditoolkit.Note(
                        velocity=current_vel,
                        pitch=current_pitch,
                        start=current_tick,
                        end=current_tick + duration
                    )
                    tracks[current_inst].notes.append(note)

        if output_path:
            midi_obj.dump(output_path)
            
        return midi_obj

    def process_dataset(self):
        files = []
        # Updated to user's path
        search_dir = self.data_dir if self.data_dir else '/Users/gordonkim/Downloads/lmd_full'
        
        print(f"Scanning {search_dir}...")
        for root, _, filenames in os.walk(search_dir):
            for filename in filenames:
                if filename.lower().endswith(('.mid', '.midi')):
                    files.append(os.path.join(root, filename))
        
        print(f"Found {len(files)} MIDI files.")
        
        processed_count = 0
        output_file = os.path.join(self.output_dir, 'dataset.txt')
        
        with open(output_file, 'w') as f:
            for file_path in tqdm(files):
                token_str = self.process_file(file_path)
                if token_str:
                    f.write(token_str + '\n')
                    processed_count += 1
        
        print(f"Successfully processed {processed_count} files to {output_file}")

# Example Usage
if __name__ == "__main__":
    # You can run this file directly to process the dataset
    preprocessor = MIDIPreprocessor('/Users/gordonkim/Downloads/lmd_full', './processed_data')
    # preprocessor.process_dataset() 

