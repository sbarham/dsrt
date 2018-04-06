import logging
import tqdm
from multiprocessing import Pool

from dsrt.config.defaults import DataConfig

class Padder:
    def __init__(self, properties, parallel=True, config=DataConfig()):
        self.properties = properties
        self.config = config
        self.parallel = parallel
        
        self.max_ulen = self.properties['max-utterance-length']
        self.max_dlen = self.properties['max-dialogue-length']
        
        self.init_logger()
        
    def init_logger(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(self.config['logging-level'])

    def transform(self, dialogues):
        self.log('info', 'Padding the dialogues (using max utterance length={} tokens) ...'.format(self.max_ulen))

        self.empty_turn = [self.config['pad-d']] * (self.properties['max-utterance-length'] + 1)

        chunksize=self.config['chunksize']
        p = Pool() if self.parallel else Pool(1)
        res = []
        total = len(dialogues)

        self.log('info', '[padder running on {} cores]'.format(p._processes))

        for d in tqdm.tqdm(p.imap(self.pad_dialogue, dialogues, chunksize=chunksize), total=total):
            res.append(d)

        p.close()
        p.join()
        
        return res

    def pad_dialogues(self, dialogues):
        """
        Pad the entire dataset.
        This involves adding padding at the end of each sentence, and in the case of
        a hierarchical model, it also involves adding padding at the end of each dialogue,
        so that every training sample (dialogue) has the same dimension.
        """
        
        self.log('info', 'Padding the dialogues ...')
        
        return [self.pad_dialogue(d) for d in dialogues]

    def pad_dialogue(self, dialogue):
        for i, u in enumerate(dialogue):
            dif = self.max_ulen - len(u) + 1
            dialogue[i] += [self.config['pad-u']] * dif
                
        # only pad the dialogue if we're training a hierarchical model
        if self.config['hierarchical']:
            dif = self.max_dlen - len(dialogue)
            dialogues += [self.empty_turn] * dif

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
