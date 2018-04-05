import logging
from nltk import word_tokenize
import progressbar
from dsrt.config.defaults import DataConfig

class Tokenizer:
    def __init__(self, config=DataConfig()):
        self.config = config
        self.init_logger()

    def transform(self, dialogues):
        self.log('info', 'Tokenizing the dialogues (this may take a while) ...')
        
        # this can be a very long operation; we'll track progress with a progressbar ...
        bar = progressbar.ProgressBar(widgets=[
            ' [', progressbar.Timer(), '] ',
            progressbar.Bar(),
            ' (', progressbar.ETA(), ') ',
        ])
        
        num_dialogues = len(dialogues)
        
        res = []
        for i, d in enumerate(dialogues):
            res.append(self.tokenize_dialogue(d))
            bar.update(100 * (i / num_dialogues))
        
        print()
        
        return res

    def tokenize_dialogues(self, dialogues):
        return [self.tokenize_dialogue(d) for d in dialogues]

    def tokenize_dialogue(self, dialogue):
        utterances = dialogue.split('\t') # [:-1]
        return [self.tokenize_utterance(u) for u in utterances]

    def tokenize_utterance(self, utterance):
        return word_tokenize(utterance)
    
    def init_logger(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(self.config['logging-level'])

    def log(self, priority, msg):
        """
        Just a wrapper, for convenience.
        NB1: priority may be set to one of:
        - CRITICAL     [50]
        - ERROR        [40]
        - WARNING      [30]
        - INFO         [20]
        - DEBUG        [10]
        - NOTSET       [0]
        Anything else defaults to [20]
        NB2: the levelmap is a defaultdict stored in Config; it maps priority
             strings onto integers
        """
        self.logger.log(logging.CRITICAL, msg)