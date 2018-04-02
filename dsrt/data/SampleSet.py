# NumPy
import numpy as np
from numpy import argmax

# Python stdlib
import logging

# our own imports
from dsrt.config.defaults import DataConfig

class SampleSet:
    def __init__(self, dialogues=None, vectorizer=None, properties=None, config=DataConfig(),
                 preprocessed=False, name=None, f=None):
        # initialize the logger
        self.init_logger()
        
        self.config = config
        self.vectorizer = vectorizer
        self.properties = properties
        
        # read in dialogues
        self.dialogues = np.swapaxes(dialogues, 0, 1)
        self.length = len(dialogues)
        
        if preprocessed:
        	self.load_sampleset(f, name)
        	return
        
        # make encoder/decoder split
        self.encoder_x = self.dialogues[0]
        self.decoder_x = self.dialogues[1]
        self.decoder_y = self.dialogues[2]
        
    def init_logger(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(self.config['logging-level'])
            
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