# NumPy
import numpy as np
from numpy import argmax

# Python stdlib
import logging

# our own imports
from dsrt.config import DataConfig

class SampleSet:
    def __init__(self, dialogues=None, vectorizer=None, properties=None, config=DataConfig(),
                 preprocessed=False, name=None, f=None):
        self.dialogues = dialogues
        self.length = len(dialogues)
        self.config = config
        self.vectorizer = vectorizer
        self.properties = properties
        
        # initialize the logger
        self.init_logger()
        
        # if we are reading in a preprocessed sampleset, exit initialization now to do that:
        if preprocessed:
            self.load_sampleset(f, name)
            return
        
        # make the encoder/decoder split:
        self.encoder_decoder_split()
        
    def init_logger(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(self.config['logging-level'])
    
    def encoder_decoder_split(self):
        """
        For now, this assumes a flat (non-hierarchical) model, and therefore
        assumes that dialogues are simply adjacency pairs.
        """
        # get start, stop, and pad symbols
        start = self.vectorizer.word_to_index(self.config['start'])
        stop = self.vectorizer.word_to_index(self.config['stop'])
        pad = 0
        
        # initialize encoder/decoder samples
        self.encoder_x = np.copy(self.dialogues[:, 0])
        self.decoder_x = np.zeros(self.encoder_x.shape)
        self.decoder_y = np.copy(self.dialogues[:, 1])
        
        # prepare decoder_x -- (prefix the <start> symbol to every second-pair part)
        self.decoder_x[:, 0] = start
        for i in range(self.decoder_y.shape[0]):
            for j in range(self.decoder_y.shape[1]):
                if self.decoder_y[i, j] == pad:
                    self.decoder_y[i, j] = stop
                    break
        
                self.decoder_x[i, j + 1] = self.decoder_y[i, j]
        
        # prepare decoder_y -- the sparse_categorical_crossentropy loss function expects 3D tensors,
        # where each word sequence is like [[72], [5], [44], [0] ...] -- so we add an extra dim
        old_shape = self.decoder_y.shape
        new_shape = (old_shape[0], old_shape[1], 1)
        self.decoder_y = self.decoder_y.reshape(new_shape)
        
    def save_sampleset(self, f, name):
        '''Serialize the sampleset to file using the HDF5 format. Name is usually in {train, test}.'''
        f.create_dataset(name + '_encoder_x', data=self.encoder_x)
        f.create_dataset(name + '_decoder_x', data=self.decoder_x)
        f.create_dataset(name + '_decoder_y', data=self.decoder_y)
        
    def load_sampleset(self, f, name):
        '''Read the sampleset from using the HDF5 format. Name is usually in {train, test}.'''
        self.encoder_x = np.array(f[name + '_encoder_x'])
        self.decoder_x = np.array(f[name + '_decoder_x'])
        self.decoder_y = np.array(f[name + '_decoder_y'])
    
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
        # self.logger.log(self.config.levelmap[priority], msg)
        self.logger.log(logging.CRITICAL, msg)