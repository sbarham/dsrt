from collections import defaultdict
from dsrt.definitions import ROOT_DIR
import logging

class BaseConfig(dict):
    def __init__(self):
        self['model-name'] = 'test'
        
        # these are more or less global parameters
        self['path-to-corpus'] = ROOT_DIR + '/archive/corpora/cornell/dialogues.txt'
        self['vocab-size'] = 10000
        self['random-state'] = 100
        self['shuffle'] = True
        self['one-hot-encode'] = False
        
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
        # with a default setting of 20x100 dialogues, a one-hot encoded
        # dialogue occupies about 160MB of space in memory
        self['max-utterance-length'] = 40
        self['max-dialogue-length'] = 6
        
        # general network parameters
        self['recurrent-unit-type'] = 'gru'
        self['hierarchical'] = False

        # init parameters relevant to the four components of the network
        self.init_embedding_config()
        self.init_utterance_encoder_config()
        self.init_context_encoder_config()
        self.init_utterance_decoder_config()
        
        # init training and validation parameters
        self.init_training_and_validation_parms()
        
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

    def init_embedding_config(self):
        self['embedding-dim'] = 512
        
    def init_utterance_encoder_config(self):
        self['encoding-layer-width'] = 512
        self['encoding-layer-depth'] = 3
        self['encoding-layer-bidirectional'] = True

    def init_context_encoder_config(self):
        self['context-layer-width'] = 512
        self['context-layer-depth'] = 2
        self['context-layer-bidirectional'] = True

    def init_utterance_decoder_config(self):
        self['decoding-layer-width'] = 512
        self['decoding-layer-depth'] = 3
        self['decoding-layer-bidirectional'] = True
        
    def init_training_and_validation_parms(self):
        self['optimizer'] = 'rmsprop'
        self['loss'] = 'categorical_crossentropy'
        self['batch-size'] = 64
        self['train-test-split'] = .8
        self['validation-split'] = .2
        self['num-epochs'] = 500
        self['reporting-frequency'] = 100
