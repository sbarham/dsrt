from collections import defaultdict
import logging
import os
from dsrt.definitions import LIB_DIR
from dsrt.config.defaults import BaseConfig

class DataConfig(dict):
    def __init__(self):
        # add the base configuration
        self.update(BaseConfig())
        
        # general dataset parameters
        self['corpora-dir'] = os.path.join(LIB_DIR, 'corpora')
        self['datasets-dir'] = os.path.join(LIB_DIR, 'datasets')
        self['corpus-name'] = 'test'
        self['dataset-name'] = 'test'
        self['vocab-size'] = 10000
        self['random-state'] = 100
        self['shuffle'] = True
        self['one-hot-encode'] = False
        self['hierarchical'] = False # this influences how the dialogues are preprocessed
<<<<<<< HEAD
        
        # preprocessing paramaters
        self['chunksize'] = 100
=======
>>>>>>> 862dff8777db5d01aa74b25c93cb60f1514e208d

        # reserved keywords
        self['start'] = '<start>'
        self['stop'] = '<stop>'
        self['pad-d'] = '<pad_d>'
        self['pad-u'] = '<pad_u>'
        self['unk'] = '<unk>'

        # logging level -- may be set to one of:
        # - CRITICAL     [50]
        # - ERROR        [40]
        # - WARNING      [30]
        # - INFO         [20]
        # - DEBUG        [10]
        # - NOTSET       [0]
        # Either str or int is acceptable
        self['logging-level'] = logging.INFO
        self.init_levelmap()

        # allows us to restrict the size of the dataset used -- this might help in
        # memory restricted environments, or to speed up testing:
        self['restrict-sample-size'] = True

        self['sample-size'] = 1000 # restrict the number of dialogues to an arbitrary number


        # use the max lengths already present in the corpus in case the user doesn't provide values
        # (this is usually preferred)
        self['filter-dialogues-with-long-utterances'] = True
        self['filter-long-dialogues'] = False
        # with a default setting of 20x100 dialogues, a one-hot encoded
        # dialogue occupies about 160MB of space in memory
        self['max-utterance-length'] = 40
        self['max-dialogue-length'] = None #10

        self['train-test-split'] = .8

    def init_levelmap(self):
        self.levelmap = dict(
            #lambda: 20,
            {
                'CRITICAL': 50,
                'critical': 50,
                'ERROR': 50,
                'error': 50,
                'WARNING': 50,
                'warning': 50,
                'INFO': 50,
                'info': 50,
                'DEBUG': 50,
                'debug': 50
            })
