"""
Represents an encoder-decoder dialogue model
"""

# Keras packages
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, GRU, Dense, Bidirectional
from keras.utils.training_utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint

# we need Tensforflow to save a central copy of the model on our CPU, from which
# we spawn multiple copies on any available (indicated) GPUs
import tensorflow as tf

# nltk
from nltk import word_tokenize

# numpy
import numpy as np

# Python stdlib
import os
import math
import re
import matplotlib.pyplot as plt

# Our packages
from dsrt.definitions import LIB_DIR, CHECKPOINT_DIR
from dsrt.config.defaults import ModelConfig
from dsrt.conversation import Conversation


class EncoderDecoder:
    def __init__(self, encoder=None, decoder=None, config=ModelConfig(), num_gpus=1):
        self.config = config
        self.encoder = encoder
        self.decoder = decoder
        self.num_gpus = num_gpus
        
        # build the trainin and inference models; save them
        self.build_callbacks()
        self.build_training_model()
        self.build_inference_model()
        
    def build_callbacks(self):
        '''Eventually, this should be configured, rather than hardcoded'''
        # checkpoint
        filepath = os.path.join(CHECKPOINT_DIR, 'weights.best.hdf5')
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
                                
                                
        self.callbacks = [checkpoint]
        
    def build_training_model(self):
        self.encoder_input = self.encoder.encoder_input
        self.encoder_embed = self.encoder.encoder_embed
        self.decoder_input = self.decoder.decoder_input
        self.decoder_embed = self.decoder.decoder_embed
        self.decoder_output = self.decoder.decoder_output
        
        if self.config['hierarchical']:
            pass # do something
        else:
            if self.num_gpus <= 1:
                print("[INFO] training with 1 GPU...")
                self.training_model = Model([self.encoder_input, self.decoder_input], self.decoder_output)
            else:
                print("[INFO] training with {} GPUs...".format(self.num_gpus))
                # we'll store a copy of the model on *every* GPU and then combine
                # the results from the gradient updates on the CPU
                with tf.device("/cpu:0"):
                    model = Model([self.encoder_input, self.decoder_input], self.decoder_output)
                
                # make the model parallel
                self.training_model = multi_gpu_model(model, gpus=self.num_gpus)
    
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
        # ensure the checkpoint dir exists
        if not os.path.exists(CHECKPOINT_DIR):
            os.makedirs(CHECKPOINT_DIR)
                                
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

        self.training_model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        # self.training_model.fit([encoder_x, decoder_x], decoder_y,
        #                         batch_size=batch_size,
        #                         epochs=num_epochs,
        #                         validation_split=validation_split,
        #                         callbacks=self.callbacks)
        history = self.training_model.fit([encoder_x, decoder_x], decoder_y,
                                batch_size=batch_size,
                                epochs=num_epochs,
                                validation_split=validation_split,
                                callbacks=self.callbacks)
        #print(history.history.keys())
        fig1 = plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        fig1.savefig('val_trn_loss.png')
        
        fig2 = plt.figure()
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        fig2.savefig('val_trn_accuracy.png')
        
        # remember the vectorizer used in training
        self.vectorizer = data.vectorizer

        # ask the user if they would like to converse with the new model for
        # spot-checking
        if input("Would you like to converse with the newly trained model? ").lower().startswith('y'):
            Conversation(self.encoder_model, self.decoder_model, self.vectorizer).converse()
    
    def save_models(self, model_path):
        self.training_model.save(os.path.join(model_path, 'train'))
        self.encoder_model.save(os.path.join(model_path, 'inference_encoder'))
        self.decoder_model.save(os.path.join(model_path, 'inference_decoder'))
        
        if not self.vectorizer is None:
            self.vectorizer.save_vectorizer(model_path)
