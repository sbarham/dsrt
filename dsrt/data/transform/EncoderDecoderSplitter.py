import logging

import numpy as np

# our imports
from dsrt.config.defaults import DataConfig

class EncoderDecoderSplitter:
    def __init__(self, properties, vectorizer, config=DataConfig()):
        self.properties = properties
        self.vectorizer = vectorizer
        self.config = config
        self.init_logger()

    def init_logger(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(self.config['logging-level'])

    def transform(self, dialogues):
        return self.encoder_decoder_split(dialogues)

    def encoder_decoder_split(self, dialogues):
        """
	    For now, this assumes a flat (non-hierarchical) model, and therefore
	    assumes that dialogues are simply adjacency pairs.
	    """
        self.log('info', 'Making encoder decoder split ...')

	    # get start, stop, and pad symbols
        start = self.properties.start
        stop = self.properties.stop
        pad = self.properties.pad_u

	    # initialize encoder/decoder samples
        encoder_x = np.copy(dialogues[:, 0])
        decoder_x = np.zeros(encoder_x.shape)
        decoder_y = np.copy(dialogues[:, 1])

	    # prepare decoder_x -- (prefix the <start> symbol to every second-pair part)
        decoder_x[:, 0] = start
        for i in range(decoder_y.shape[0]):
            for j in range(decoder_y.shape[1] - 1):
                if decoder_y[i, j] == pad:
                    decoder_y[i, j] = stop
                    break
                decoder_x[i, j + 1] = decoder_y[i, j]

        # prepare decoder_y -- the sparse_categorical_crossentropy loss function expects 3D tensors,
        # where each word sequence is like [[72], [5], [44], [0] ...] -- so we add an extra dim
        old_shape = decoder_y.shape
        new_shape = (old_shape[0], old_shape[1], 1)
        decoder_y = decoder_y.reshape(new_shape)

        return [encoder_x, decoder_x, decoder_y]

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
