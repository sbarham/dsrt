from collections import defaultdict
import logging

from dsrt.definitions import ROOT_DIR
from dsrt.config.defaults import BaseConfig

class ConversationConfig(dict):
    def __init__(self):
        # add the abse configuration
        self.update(BaseConfig())
        
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
        self['use-corpus-max-utterance-length'] = False
        self['use-corpus-max-dialogue-length'] = False
        self['max-utterance-length'] = 40
        self['max-dialogue-length'] = 6
        
        self['recurrent-unit-type'] = 'gru'
        
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

