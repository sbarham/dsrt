"""
Represents an encoder-decoder dialogue model
"""

# Keras packages
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, GRU, Dense, Bidirectional

# nltk
from nltk import word_tokenize

# numpy
import numpy as np

# Python stdlib
import math
import re

# Our packages
from dsrt.config.defaults import ModelConfig
from dsrt.definitions import ROOT_DIR


class EncoderDecoder:
    def __init__(self, encoder, decoder, config=ModelConfig()):
        self.config = config
        self.encoder = encoder
        self.decoder = decoder
        
        # build the trainin and inference models; save them
        self.build_training_model()
        self.build_inference_model()
        
    def build_training_model(self):
        self.encoder_input = self.encoder.encoder_input
        self.encoder_embed = self.encoder.encoder_embed
        self.decoder_input = self.decoder.decoder_input
        self.decoder_embed = self.decoder.decoder_embed
        self.decoder_output = self.decoder.decoder_output
        
        if self.config['hierarchical']:
            # do something
            pass
        else:
            self.training_model = Model([self.encoder_input, self.decoder_input], self.decoder_output)
    
    def build_inference_model(self):
        # grab some important hyperparameters
        hidden_dim = self.config['encoding-layer-width']
        recurrent_unit = self.config['recurrent-unit-type']
        
        # build the encoder model
        self.encoder_model = Model(self.encoder.encoder_input, self.encoder.encoder_hidden_state)
    
        decoder_hidden_state_input = None
        decoder_hidden_state_output = None
        decoder_output = None
        # build the decoder model
        if recurrent_unit == 'lstm':
            decoder_hidden_state_input_h = Input(shape=(hidden_dim,))
            decoder_hidden_state_input_c = Input(shape=(hidden_dim,))
            decoder_hidden_state_input = [decoder_hidden_state_input_h, decoder_hidden_state_input_c]
            # take in the regular inputs, condition on the hidden state
            _, decoder_state_h, decoder_state_c = self.decoder.decoder_rnn(self.decoder_embed,
                                                                           initial_state=decoder_hidden_state_input)
            decoder_hidden_state_output = [decoder_state_h, decoder_state_c]
        elif recurrent_unit == 'gru':
            decoder_hidden_state_input = [Input(shape=(hidden_dim,))]
            # take in the regular inputs, condition on the hidden state
            decoder_output, hidden_state = self.decoder.decoder_rnn(self.decoder_embed,
                                                                    initial_state=decoder_hidden_state_input)
            decoder_hidden_state_output = [hidden_state]
        else:
            raise Exception('Invalid recurrent unit type: {}'.format(recurrent_unit))
            
        decoder_output = self.decoder.decoder_dense(decoder_output)
        self.decoder_model = Model([self.decoder_input] + decoder_hidden_state_input,
                                   [decoder_output] + decoder_hidden_state_output)
    
    def fit(self, data):
        # grab some hyperparameters from our config
        optimizer = self.config['optimizer']
        loss = self.config['loss']
        batch_size = self.config['batch-size']
        num_epochs = self.config['num-epochs']
        validation_split = self.config['validation-split']
        
        # grab the training and validation data
        encoder_x = data.train.encoder_x
        decoder_x = data.train.decoder_x
        decoder_y = data.train.decoder_y #_ohe
        
        self.training_model.compile(optimizer=optimizer, loss=loss)
        self.training_model.fit([encoder_x, decoder_x], decoder_y,
                                batch_size=batch_size,
                                epochs=num_epochs,
                                validation_split=validation_split)
        
        # remember the vectorizer used in training
        self.vectorizer = data.vectorizer
    
    def save_models(self, model_name):
        prefix = ROOT_DIR + '/archive/models/' + model_name
        self.training_model.save(prefix + '_train')
        self.encoder_model.save(prefix + '_inference_encoder')
        self.decoder_model.save(prefix + '_inference_decoder')
        if not self.vectorizer is None:
            self.vectorizer.save(prefix + '_vectorizer')