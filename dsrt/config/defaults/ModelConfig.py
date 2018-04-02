from collections import defaultdict
import logging

from dsrt.definitions import ROOT_DIR
from dsrt.config.defaults import BaseConfig

class ModelConfig(BaseConfig):
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
        self['embedding-dim'] = 64
        
    def init_utterance_encoder_config(self):
        self['encoding-layer-width'] = 128
        self['encoding-layer-depth'] = 3
        self['encoding-layer-bidirectional'] = True

    def init_context_encoder_config(self):
        self['context-layer-width'] = 128
        self['context-layer-depth'] = 2
        self['context-layer-bidirectional'] = True

    def init_utterance_decoder_config(self):
        self['decoding-layer-width'] = 128
        self['decoding-layer-depth'] = 3
        self['decoding-layer-bidirectional'] = True
        
    def init_training_and_validation_parms(self):
        self['optimizer'] = 'rmsprop'
        self['loss'] = 'sparse_categorical_crossentropy'
        self['batch-size'] = 64
        self['train-test-split'] = .8
        self['validation-split'] = .2
        self['num-epochs'] = 500
        self['reporting-frequency'] = 100
