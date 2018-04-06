import os
import pickle
from dsrt.config.defaults import DataConfig

class Properties(dict):
    def __init__(self, dialogues, vocab_size, config=DataConfig()):
        self.config = config
        self.vocab_size = vocab_size
        self.analyze(dialogues)

    def analyze(self, dialogues):
        # get num dialogues and max dialogue length
        dialogue_lengths = [len(d) for d in dialogues]
        self['num-dialogues'] = len(dialogue_lengths)
        
        if self.config['filter-long-dialogues']:
            self['max-dialogue-length'] = self.config['max-dialogue-length']
        else:
            self['max-dialogue-length'] = max(dialogue_lengths)
        
        # get num utterances and max utterance length
        utterance_lengths = [len(u) for d in dialogues for u in d]
        self['num-utterances'] = len(utterance_lengths)
        
        if self.config['filter-dialogues-with-long-utterances']:
            self['max-utterance-length'] = self.config['max-utterance-length']
        else:
            self['max-utterance-length'] = max(utterance_lengths)
    
    def save_properties(self, path):
        with open(os.path.join(path, 'properties'), 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_properties(path):
        with open(os.path.join(path, 'properties'), 'rb') as f:
            return pickle.load(f)
