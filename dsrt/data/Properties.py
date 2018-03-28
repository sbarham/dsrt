from dsrt.config import DataConfig

class Properties(dict):
	def __init__(self, dialogues, config=DataConfig())
		self.config = config
		self.analyze(dialogues)
	
	def analyze(self, dialogues):
        dialogue_lengths = []
        utterance_lengths = []
        
        for dialogue in dialogues:
            # get dialogue length
            dialogue_lengths += [len(dialogue)]

            # get constituent utterances lengths
            utterance_lengths_flat += [len(u) for u in dialogue]
        
        self['max_dialogue_length'] = max(dialogue_lengths)
        self['max_utterance_length'] = max(utterance_lengths_flat)
        self['num-dialogues'] = len(dialogue_lengths)
        self['num-utterances'] = len(utterance_lengths)
        
        return
