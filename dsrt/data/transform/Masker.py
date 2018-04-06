import logging
import tqdm
from multiprocessing import Pool
import numpy as np
from dsrt.config.defaults import DataConfig

class Masker:
    def __init__(self, vectorizer, properties=None, parallel=True, config=DataConfig()):
        self.properties = properties
        self.parallel = parallel
        self.config = config
        
        self.pad_u_index = vectorizer.word_to_index(config['pad-u'])
        
        self.init_logger()
    
    def init_logger(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(self.config['logging-level'])
    
    def transform(self, dialogues):
        '''Takes an integer encoded dialogue and swaps our padding with the 0-mask Keras expects'''
        self.log('info', 'Masking dialogues ...')

        chunksize=self.config['chunksize']
        p = Pool() if self.parallel else Pool(1)
        res = []
        total=len(dialogues)
        
        self.log('info', '[masker operating on {} cores]'.format(p._processes))
        for d in tqdm.tqdm(p.imap(self.mask_dialogue, dialogues, chunksize=chunksize), total=total):
                res.append(d)
                
        p.close()
        p.join()

        return res

    def mask_dialogue(self, dialogue):
        return [self.swap_pad_and_zero(u) for u in dialogue]
    
    def swap_pad_and_zero(self, utterance):
        for i, w in enumerate(utterance):
            if w == 0:
                utterance[i] = self.pad_u_index
            elif w == self.pad_u_index:
                utterance[i] = 0

        return utterance
    
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