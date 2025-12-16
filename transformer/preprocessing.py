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

    def json_to_tokens(self, json_notes):
        """
        Converts Frontend JSON notes to Token String.
        json_notes: list of {pitch, startTime, duration, instrument}
        """
        notes = []
        for n in json_notes:
            # Map frontend instrument names to classes
            # Frontend: "piano", "drums", "bass", "strings"
            # Classes: "Piano", "Drums", "Bass", "Strings" (Capitalized)
            inst = n.get('instrument', 'piano').capitalize()
            if inst == 'Drums': inst = 'Drums' # Special case if needed
            
            # Quantize
            start_tick = int(round(n['startTime'] * TICKS_PER_BEAT))
            duration = int(round(n['duration'] * TICKS_PER_BEAT))
            duration = max(1, min(96, duration))
            
            notes.append({
                'start': start_tick,
                'pitch': n['pitch'],
                'velocity': 100, # Default velocity
                'duration': duration,
                'instrument': inst
            })
            
        notes.sort(key=lambda x: (x['start'], x['pitch'], x['instrument']))
        
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
            inst = n.get('instrument', 'piano').capitalize()
            if inst == 'Drums': inst = 'Drums'
            
            start_tick = int(round(n['startTime'] * TICKS_PER_BEAT))
            duration = int(round(n['duration'] * TICKS_PER_BEAT))
            duration = max(1, min(96, duration))
            
            notes.append({
                'start': start_tick,
                'pitch': n['pitch'],
                'velocity': 100,
                'duration': duration,
                'instrument': inst,
                'original_start': n['startTime'] # Keep original for masking check
            })
            
        notes.sort(key=lambda x: (x['start'], x['pitch'], x['instrument']))
        
        events = []
        mask = [] # Boolean list
        current_tick = 0
        
        # Add BOS
        events.append("<bos>")
        mask.append(False)
        
        for note in notes:
            # Check if this note falls within the masking range
            # We mask if the note starts within the range
            is_masked = (note['original_start'] >= start_time and note['original_start'] < end_time)
            
            delta = note['start'] - current_tick
            if delta > 0:
                while delta > MAX_TIME_SHIFT:
                    events.append(f"TimeShift_{MAX_TIME_SHIFT}")
                    # If the NEXT note is masked, we mask the time shift leading to it?
                    # Or if the PREVIOUS note was masked?
                    # TimeShift belongs to the gap.
                    # If we are entering a masked region, we should mask the shift so the model can decide when the next note starts.
                    mask.append(is_masked) 
                    delta -= MAX_TIME_SHIFT
                if delta > 0:
                    events.append(f"TimeShift_{delta}")
                    mask.append(is_masked)
                current_tick = note['start']
            
            # Note tokens
            tokens = [
                f"Inst_{note['instrument']}",
                f"Pitch_{note['pitch']}",
                f"Vel_{note['velocity'] // 16}",
                f"Dur_{note['duration']}"
            ]
            
            events.extend(tokens)
            mask.extend([is_masked] * 4)
            
        # Add EOS
        events.append("<eos>")
        mask.append(False)
        
        # ELASTIC MASKING LOGIC
        # We need to ensure there are enough slots in the masked region for the AI to write new music.
        # If the selection is empty or sparse, we inject <pad> tokens to give it "scratch space".
        
        # 1. Calculate Duration of Selection in Beats
        selection_duration_ticks = (end_time - start_time) * TICKS_PER_BEAT
        selection_duration_beats = selection_duration_ticks / TICKS_PER_BEAT
        
        # 2. Target Density
        # A complex 16th note melody needs ~5 tokens per 1/4 beat.
        # So ~20 tokens per beat is a safe upper bound for "busy" music.
        TOKENS_PER_BEAT = 20 
        target_count = int(selection_duration_beats * TOKENS_PER_BEAT)
        
        # 3. Count existing masked tokens
        current_masked_count = sum(mask)
        
        # 4. Inject Padding if needed
        needed = target_count - current_masked_count
        
        if needed > 0:
            # Find where to insert (at the end of the masked region, or just before the first unmasked token after the region)
            # We look for the transition from True to False
            insert_index = -1
            for i in range(len(mask) - 1, -1, -1):
                if mask[i]:
                    insert_index = i + 1
                    break
            
            # If no masked tokens found (e.g. selecting empty silence), find where the silence *would* be
            if insert_index == -1:
                # This is tricky. If we selected silence, we probably have TimeShifts that cover it.
                # But we didn't mask the TimeShifts unless they were attached to notes?
                # Actually, in the loop above, we mask TimeShifts if they lead to a masked note.
                # If there are NO notes, nothing is masked.
                
                # Fallback: Insert at the point corresponding to start_time?
                # For simplicity, let's insert after BOS if nothing else.
                insert_index = 1 
                
                # But wait, if we select silence in the middle of a song, we need to find that spot.
                # This simple logic might put it at the beginning.
                # Ideally, we should find the TimeShift that crosses 'start_time'.
                
                # IMPROVED LOGIC:
                # If we have a selection but no notes were captured, we likely want to insert 
                # right where the time cursor would be.
                # For now, let's just append to the end of the sequence if it's empty? No.
                
                # Let's stick to the simple case: If we have SOME masked tokens, we extend them.
                # If we have ZERO masked tokens (pure silence selection), we might need to mask the TimeShift that covers this gap.
                pass

            if insert_index != -1:
                # Inject <pad> tokens
                # We mark them as True (Masked) so the model can overwrite them
                pads = ["<pad>"] * needed
                pad_mask = [True] * needed
                
                events[insert_index:insert_index] = pads
                mask[insert_index:insert_index] = pad_mask
                
        return " ".join(events), mask

    def tokens_to_json(self, token_str):
        """
        Converts Token String back to Frontend JSON notes.
        """
        tokens = token_str.split()
        json_notes = []
        
        current_tick = 0
        current_inst = "Piano"
        current_pitch = 60
        current_vel = 100
        
        for token in tokens:
            if token.startswith("TimeShift_"):
                shift = int(token.split("_")[1])
                current_tick += shift
                
            elif token.startswith("Inst_"):
                current_inst = token.split("_")[1].lower()
                
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

