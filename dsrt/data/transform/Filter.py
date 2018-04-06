import tqdm
from multiprocessing import Pool
import logging

from dsrt.config.defaults import DataConfig

class Filter:
    def __init__(self, properties, parallel=True, config=DataConfig()):
        self.properties = properties
        self.config = config
        self.parallel = parallel
        self.init_logger()
        
    def init_logger(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(self.config['logging-level'])

    def transform(self, dialogues):
        chunksize=self.config['chunksize']
        p = Pool() if self.parallel else Pool(1)
        
        if self.config['filter-long-dialogues']:
            self.max_dl = self.config['max-dialogue-length']
            self.log('info', 'Filtering long dialogues (> {} utterances) ...'.format(self.max_dl))
            
            res = []
            total = len(dialogues)
            
            self.log('info', '[filter running on {} cores]'.format(p._processes))
            for d in tqdm.tqdm(p.imap(self.filter_long_dialogues, dialogues, chunksize=chunksize), total=total):
                res.append(d)
            
            dialogues = list(filter(None, res))
            
        if self.config['filter-dialogues-with-long-utterances']:
            self.max_ul = self.config['max-utterance-length']
            self.log('info', 'Filtering dialogues with long utterances (> {} tokens) ...'.format(self.max_ul))
            
            res = []
            total = len(dialogues)
            
            self.log('info', '[filter running on {} cores]'.format(p._processes))
            for d in tqdm.tqdm(p.imap(self.filter_dialogues_with_long_utterances, dialogues, chunksize=chunksize), total=total):
                res.append(d)
                
            dialogues = list(filter(None, res))

        p.close()
        p.join()

        return dialogues

    def filter_long_dialogues(self, dialogue):
        if len(dialogue) > self.max_dl:
            return None
        
    def filter_dialogues_with_long_utterances(self, dialogue):
        for utterance in dialogue:
            if len(utterance) > self.max_ul:
                return None
                
        return dialogue
    
    ####################
    #     UTILITIES    #
    ####################
    
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
