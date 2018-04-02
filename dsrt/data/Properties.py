from dsrt.config.defaults import DataConfig

class Properties(dict):
    def __init__(self, dialogues, config=DataConfig()):
        self.config = config
        self.analyze(dialogues)

    def analyze(self, dialogues):
        dialogue_lengths = []
        utterance_lengths = []

        for dialogue in dialogues:
            # get dialogue length
            dialogue_lengths += [len(dialogue)]

            # get constituent utterances lengths
            utterance_lengths += [len(u) for u in dialogue]

        self['max-dialogue-length'] = max(dialogue_lengths)
        self['max-utterance-length'] = max(utterance_lengths)
        self['num-dialogues'] = len(dialogue_lengths)
        self['num-utterances'] = len(utterance_lengths)
        
        return
