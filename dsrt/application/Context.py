"""
Maintains a dialogue research application context, which includes (minimally)
(1) configuration objects (which < dict()) for dataset management, model creation and management, and
    conversation management
(2) a dataset, usually in the form of a Corpus object, which represents both a corpus of dialogues
    anda sequence of transformations to apply to the raw dialogue/utterance data in order to prepare
    them for processing by our neural models

Using this application context, is able to construct any kind of neural dialog model desired.
"""

import keras
import os
import shutil
from pickle import load
from dsrt.data.transform import Vectorizer
from dsrt.models import Encoder, Decoder, EncoderDecoder
from dsrt.config.defaults import DataConfig, ModelConfig, ConversationConfig
from dsrt.conversation import Conversation
from dsrt.definitions import LIB_DIR

class Context:
    def __init__(self, data_config=DataConfig(), model_config=ModelConfig,
                 conversation_config=ConversationConfig(), dataset=None,
                 num_gpus=1):
        self.data_config = data_config
        self.model_config = model_config
        self.conversation_config = conversation_config

        self.dataset = dataset
        self.num_gpus = num_gpus
        self.model = None

    def train(self):
        self.model.fit(self.dataset)

    def save_model(self, model_name):
        self.model_name = model_name
        self.model_path = os.path.join(LIB_DIR, 'models', model_name)
        
        if os.path.exists(self.model_path):
            choice = input("Model '{}' already exists; overwrite it? y(es) | n(o): ".format(model_name))
            while True:
                if choice.lower().startswith('y'):
                    shutil.rmtree(self.model_path)
                    os.makedirs(self.model_path)
                    break
                elif choice.lower().startswith('n'):
                    print("Acknowledged; aborting command ...")
                    exit(1)
                else:
                    choice = input("Invalid input. Choose (y)es | (n)o: ")
        else:
            os.makedirs(self.model_path)
            
        self.model.save_models(self.model_path)

    def load_model(self, model_name):
        # load the encoder and decoder inference models
        prefix = os.path.join(LIB_DIR, 'models', model_name)
        
        self.training_model = keras.models.load_model(os.path.join(prefix, 'train'))
        self.inference_encoder = keras.models.load_model(os.path.join(prefix, 'inference_encoder'))
        self.inference_decoder = keras.models.load_model(os.path.join(prefix, 'inference_decoder'))
        self.vectorizer = Vectorizer.load_vectorizer(prefix)

    def build_model(self):
        '''Find out the type of model configured and dispatch the request to the appropriate method'''
        if self.model_config['model-type']:
            self.model = self.build_fred()
        elif self.model_config['model-type']:
            self.model = self.buidl_hred()
        else:
            raise Error("Unrecognized model type '{}'".format(self.model_config['model-type']))

    def build_fred(self):
        '''Build a flat recurrent encoder-decoder dialogue model'''
        
        encoder = Encoder(data=self.dataset, config=self.model_config)
        decoder = Decoder(data=self.dataset, config=self.model_config, encoder=encoder)

        return EncoderDecoder(config=self.model_config, encoder=encoder, decoder=decoder, num_gpus=self.num_gpus)

    def build_hred(self):
        '''Build a hierarchical recurrent encoder-decoder dialogue model'''

        print("The HRED has not been implemented yet; returning a RED")

        return self.build_red()

    def get_conversation(self, model_name):
        if model_name is None:
            model_name = self.conversation_config['model-name']
            
        self.load_model(model_name)

        return Conversation(encoder=self.inference_encoder, decoder=self.inference_decoder, vectorizer=self.vectorizer)
