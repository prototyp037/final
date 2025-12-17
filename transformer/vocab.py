import json
import os

class MusicVocab:
    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}
        self.is_built = False
        
        # Special Tokens
        self.specials = ['<pad>', '<mask>', '<bos>', '<eos>', '<unk>']
        for token in self.specials:
            self.add_token(token)
            
    def add_token(self, token):
        if token not in self.token_to_id:
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            
    def build_vocab(self):
        """
        Manually constructs the vocabulary based on preprocessing rules.
        This ensures deterministic vocab without scanning the whole dataset.
        """
        # 1. TimeShift (1 to 48)
        for i in range(1, 49):
            self.add_token(f"TimeShift_{i}")
            
        # 2. Instruments
        # Expanded to full General MIDI (0-127) + Drums
        for i in range(128):
            self.add_token(f"Inst_{i}")
        self.add_token("Inst_Drums")
            
        # 3. Pitch (0 to 127)
        for i in range(128):
            self.add_token(f"Pitch_{i}")
            
        # 4. Velocity (0 to 7)
        for i in range(9): # 0-7 just to be safe
            self.add_token(f"Vel_{i}")
            
        # 5. Duration (1 to 96)
        for i in range(1, 97):
            self.add_token(f"Dur_{i}")
            
        self.is_built = True
        print(f"Vocabulary built with {len(self.token_to_id)} tokens.")

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.token_to_id, f, indent=2)
            
    def load(self, path):
        with open(path, 'r') as f:
            self.token_to_id = json.load(f)
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.is_built = True

    def encode(self, token_list):
        return [self.token_to_id.get(t, self.token_to_id['<unk>']) for t in token_list]

    def decode(self, id_list):
        return [self.id_to_token.get(i, '<unk>') for i in id_list]

    @property
    def pad_token_id(self):
        return self.token_to_id['<pad>']
    
    @property
    def mask_token_id(self):
        return self.token_to_id['<mask>']
    
    @property
    def bos_token_id(self):
        return self.token_to_id['<bos>']

    @property
    def eos_token_id(self):
        return self.token_to_id['<eos>']
