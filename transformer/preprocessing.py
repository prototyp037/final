import os
import miditoolkit
import numpy as np
from tqdm import tqdm
import math

# CONFIGURATION
# Resolution: Ticks per beat.
# Use a "Performance" representation (TimeShift based) instead of "REMI" (Bar/Pos) (old project style)
# Allows for ANY time signature and complex polyphony.
# 12 ticks per beat = 48 ticks per bar (in 4/4).
# Supports triplets (4 ticks) and 16ths (3 ticks).
TICKS_PER_BEAT = 12 

# Max TimeShift to include in vocabulary.
# If a pause is longer than this, we stack multiple TimeShift tokens.
# 4 beats * 12 ticks = 48 ticks.
MAX_TIME_SHIFT = 48

# Instrument Classes (Simplified General MIDI)
# Map 128 programs to fewer categories to reduce vocabulary size.
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

def get_instrument_program_from_name(name):
    """
    Maps frontend instrument names to default MIDI programs.
    """
    name = name.lower()
    if name == 'piano': return 0
    if name == 'drums': return 'Drums' # Special token
    if name == 'bass': return 32 # Acoustic Bass
    if name == 'strings': return 40 # Violin
    return 0 # Default to Piano

def get_instrument_name_from_program(program):
    """
    Maps MIDI program (or 'Drums') back to frontend instrument name.
    """
    if program == 'Drums': return 'drums'
    try:
        prog = int(program)
        # Use INSTRUMENT_CLASSES to find the category
        for start, end, name in INSTRUMENT_CLASSES:
            if start <= prog < end:
                # Map category to frontend name
                if name == 'Piano': return 'piano'
                if name == 'Bass': return 'bass'
                if name == 'Strings' or name == 'Ensemble': return 'strings'
                # Map others to piano for now as we only have 4 synths
                return 'piano'
    except:
        pass
    return 'piano'

class MIDIPreprocessor:
    def __init__(self, data_dir, output_dir):
        self.data_dir = data_dir
        self.output_dir = output_dir
        if output_dir:
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

        
        notes = []
        original_ticks_per_beat = midi_obj.ticks_per_beat

        for instrument in midi_obj.instruments:
            # Use Program Number directly (0-127) or 'Drums'
            if instrument.is_drum:
                inst_token = 'Drums'
            else:
                inst_token = instrument.program
            
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
                    'instrument': inst_token
                })

        # Sort notes by start time, then pitch
        notes.sort(key=lambda x: (x['start'], x['pitch'], str(x['instrument'])))

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

        # Clean up: Remove leading/trailing TimeShifts
        # We want the music to start immediately and end immediately
        
        # Remove leading TimeShifts
        while events and events[0].startswith("TimeShift_"):
            events.pop(0)
            
        # Remove trailing TimeShifts
        while events and events[-1].startswith("TimeShift_"):
            events.pop()
            
        if not events:
            return None
            
        # Add Special Tokens
        events = ["<bos>"] + events + ["<eos>"]

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
        
        # Dictionary to hold Instrument objects: key = program_id (or 'Drums')
        tracks = {}
        
        current_tick = 0
        current_inst = None # This will be program ID or 'Drums'
        current_pitch = None
        current_vel = None
        
        for token in tokens:
            if token.startswith("TimeShift_"):
                shift = int(token.split("_")[1])
                current_tick += shift
                
            elif token.startswith("Inst_"):
                inst_val = token.split("_")[1]
                current_inst = inst_val
                
                # Create track if not exists
                if inst_val not in tracks:
                    if inst_val == 'Drums':
                        program = 0
                        is_drum = True
                        name = 'Drums'
                    else:
                        program = int(inst_val)
                        is_drum = False
                        name = f"Instrument_{program}"
                        
                    track = miditoolkit.Instrument(program=program, is_drum=is_drum, name=name)
                    tracks[inst_val] = track
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
                if current_inst is not None and current_pitch is not None and current_vel is not None:
                    # Check if track exists (it should)
                    if current_inst in tracks:
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

    def json_to_tokens(self, json_notes):
        """
        Converts Frontend JSON notes to Token String.
        json_notes: list of {pitch, startTime, duration, instrument}
        """
        notes = []
        for n in json_notes:
            # Map frontend instrument names to Program IDs
            inst_name = n.get('instrument', 'piano')
            inst_program = get_instrument_program_from_name(inst_name)
            
            # Quantize
            start_tick = int(round(n['startTime'] * TICKS_PER_BEAT))
            duration = int(round(n['duration'] * TICKS_PER_BEAT))
            duration = max(1, min(96, duration))
            
            notes.append({
                'start': start_tick,
                'pitch': n['pitch'],
                'velocity': 100, # Default velocity
                'duration': duration,
                'instrument': inst_program
            })
            
        notes.sort(key=lambda x: (x['start'], x['pitch'], str(x['instrument'])))
        
        events = []
        current_tick = 0
        
        for note in notes:
            delta = note['start'] - current_tick
            if delta > 0:
                while delta > MAX_TIME_SHIFT:
                    events.append(f"TimeShift_{MAX_TIME_SHIFT}")
                    delta -= MAX_TIME_SHIFT
                if delta > 0:
                    events.append(f"TimeShift_{delta}")
                current_tick = note['start']
            
            events.append(f"Inst_{note['instrument']}")
            events.append(f"Pitch_{note['pitch']}")
            events.append(f"Vel_{note['velocity'] // 16}")
            events.append(f"Dur_{note['duration']}")
            
        return " ".join(events)

    def json_to_tokens_with_mask(self, json_notes, start_time, end_time):
        """
        Converts Frontend JSON notes to Token String and returns a boolean mask.
        mask is True for tokens that correspond to notes within [start_time, end_time].
        """
        notes = []
        for n in json_notes:
            inst_name = n.get('instrument', 'piano')
            inst_program = get_instrument_program_from_name(inst_name)
            
            start_tick = int(round(n['startTime'] * TICKS_PER_BEAT))
            duration = int(round(n['duration'] * TICKS_PER_BEAT))
            duration = max(1, min(96, duration))
            
            notes.append({
                'start': start_tick,
                'pitch': n['pitch'],
                'velocity': 100,
                'duration': duration,
                'instrument': inst_program,
                'original_start': n['startTime'] # Keep original for masking check
            })
            
        notes.sort(key=lambda x: (x['start'], x['pitch'], str(x['instrument'])))
        
        events = []
        mask = [] # Boolean list
        current_tick = 0
        
        start_tick_selection = int(round(start_time * TICKS_PER_BEAT))
        end_tick_selection = int(round(end_time * TICKS_PER_BEAT))
        
        # Add BOS
        events.append("<bos>")
        mask.append(False)
        
        for note in notes:
            # Check if this note falls within the masking range
            is_note_masked = (note['start'] >= start_tick_selection and note['start'] < end_tick_selection)
            
            delta = note['start'] - current_tick
            if delta > 0:
                while delta > MAX_TIME_SHIFT:
                    shift_amount = MAX_TIME_SHIFT
                    
                    # Check if this TimeShift overlaps with the selection
                    shift_start = current_tick
                    shift_end = current_tick + shift_amount
                    # Overlap logic: start < end_sel AND end > start_sel
                    is_shift_masked = (shift_start < end_tick_selection and shift_end > start_tick_selection)
                    
                    events.append(f"TimeShift_{shift_amount}")
                    mask.append(is_shift_masked) 
                    
                    delta -= MAX_TIME_SHIFT
                    current_tick += shift_amount
                    
                if delta > 0:
                    shift_amount = delta
                    
                    shift_start = current_tick
                    shift_end = current_tick + shift_amount
                    is_shift_masked = (shift_start < end_tick_selection and shift_end > start_tick_selection)
                    
                    events.append(f"TimeShift_{shift_amount}")
                    mask.append(is_shift_masked)
                    
                    current_tick += shift_amount
            
            # Note tokens
            tokens = [
                f"Inst_{note['instrument']}",
                f"Pitch_{note['pitch']}",
                f"Vel_{note['velocity'] // 16}",
                f"Dur_{note['duration']}"
            ]
            
            events.extend(tokens)
            mask.extend([is_note_masked] * 4)
            
        # Add EOS
        events.append("<eos>")
        mask.append(False)
        
        # ELASTIC MASKING LOGIC
        # 1. Calculate Duration of Selection in Beats
        selection_duration_beats = (end_time - start_time)
        
        # 2. Dynamic Density Calculation
        # Calculate average density of the existing notes (unmasked context)
        # If notes list is empty, default to moderate density
        if len(notes) > 0:
            total_duration_beats = (notes[-1]['start'] + notes[-1]['duration']) / TICKS_PER_BEAT
            total_duration_beats = max(total_duration_beats, 1.0)
            avg_notes_per_beat = len(notes) / total_duration_beats
        else:
            avg_notes_per_beat = 4.0 # Default assumption
            
        # Each note is ~5 tokens (including shifts)
        # We want to allow for slightly higher density in the infill region to give freedom
        estimated_tokens_per_beat = avg_notes_per_beat * 6 
        
        # Clamp density to reasonable bounds (e.g. min 10 tokens/beat, max 50)
        target_density = max(15, min(60, estimated_tokens_per_beat * 1.5))
        
        target_count = int(selection_duration_beats * target_density)
        
        # 3. Count existing masked tokens
        current_masked_count = sum(mask)
        
        # 4. Inject Padding if needed
        needed = target_count - current_masked_count
        
        if needed > 0:
            # Find where to insert. 
            # Strategy: Insert after the FIRST masked token we find.
            # If no masked tokens (e.g. pure silence that wasn't caught by TimeShift logic?), 
            # we look for the TimeShift that covers the start time.
            
            insert_index = -1
            
            # First, try to find any masked token
            for i in range(len(mask)):
                if mask[i]:
                    insert_index = i + 1 # Insert after the first masked token
                    break
            
            # If still -1, it means we have a selection but NO tokens were masked.
            # This happens if the selection is in empty space at the end, or if my TimeShift logic above failed.
            # But with the new TimeShift logic, if the selection overlaps any time, it should be masked.
            # Unless the selection is AFTER the last note?
            if insert_index == -1:
                # Check if selection is after the last note
                last_note_end = notes[-1]['start'] + notes[-1]['duration'] if notes else 0
                if start_tick_selection >= last_note_end:
                    # We are appending to the end.
                    # Insert before EOS
                    insert_index = len(events) - 1
            
            if insert_index != -1:
                # Inject <pad> tokens
                pads = ["<pad>"] * needed
                pad_mask = [True] * needed
                
                events[insert_index:insert_index] = pads
                mask[insert_index:insert_index] = pad_mask
                
        return " ".join(events), mask
                
        return " ".join(events), mask

    def tokens_to_json(self, token_str):
        """
        Converts Token String back to Frontend JSON notes.
        """
        tokens = token_str.split()
        json_notes = []
        
        current_tick = 0
        current_inst = "piano" # Default frontend name
        current_pitch = 60
        current_vel = 100
        
        for token in tokens:
            if token.startswith("TimeShift_"):
                shift = int(token.split("_")[1])
                current_tick += shift
                
            elif token.startswith("Inst_"):
                inst_val = token.split("_")[1]
                # Map program ID back to frontend name
                current_inst = get_instrument_name_from_program(inst_val)
                
            elif token.startswith("Pitch_"):
                current_pitch = int(token.split("_")[1])
                
            elif token.startswith("Vel_"):
                current_vel = int(token.split("_")[1]) * 16
                
            elif token.startswith("Dur_"):
                duration_ticks = int(token.split("_")[1])
                
                # Create Note
                json_notes.append({
                    "pitch": current_pitch,
                    "startTime": current_tick / TICKS_PER_BEAT,
                    "duration": duration_ticks / TICKS_PER_BEAT,
                    "instrument": current_inst,
                    "velocity": current_vel / 127
                })
                
        return json_notes


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

