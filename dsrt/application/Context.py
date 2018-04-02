"""
Maintains a dialogue research application context, which includes (minimally)
(1) configuration objects (which < dict()) for dataset management, model creation and management, and
    conversation management
(2) a dataset, usually in the form of a Corpus object, which represents both a corpus of dialogues
    anda sequence of transformations to apply to the raw dialogue/utterance data in order to prepare
    them for processing by our neural models

Using this application context, is able to construct any kind of neural dialog model desired.
"""

# import keras
from pickle import load
from dsrt.data.transform import Vectorizer
from dsrt.models import Encoder, Decoder, EncoderDecoder
from dsrt.config.defaults import DataConfig, ModelConfig, ConversationConfig
from dsrt.conversation import Conversation
from dsrt.definitions import ROOT_DIR

class Context:
    def __init__(self, data_config=DataConfig(), model_config=ModelConfig,
                 conversation_config=ConversationConfig(), data=None):
        self.data_config = data_config
        self.model_config = model_config
        self.conversation_config = conversation_config

        self.data = data
        self.model = None

    def train(self, model):
        model.fit(self.data)

    def save_model(self, model_name):
        self.model.save_models(model_name)

    def save_dataset(self, dataset_name):
        self.data.save_dataset(dataset_name)

    def load_model(self, model_name):
        # load the encoder and decoder inference models
        prefix = ROOT_DIR + '/archive/models/' + model_name
        self.training_model = keras.models.load_model(prefix + '_train')
        self.inference_encoder = keras.models.load_model(prefix + '_inference_encoder')
        self.inference_decoder = keras.models.load_model(prefix + '_inference_decoder')
        vectorizer = Vectorizer.load(prefix + '_vectorizer')

    def load_dataset(self, dataset_name):
        print("This method has not been implemented yet")

        return None

    def build_model(self):
        '''Find out the type of model configured and dispatch the request to the appropriate method'''
        if self.model_config['model-type']:
            return self.build_red()
        elif self.model_config['model-type']:
            return self.buidl_hred()
        else:
            raise Error("Unrecognized model type '{}'".format(self.model_config['model-type']))

    def build_fred(self):
        '''Build a flat recurrent encoder-decoder dialogue model'''

        encoder = Encoder(data=self.data, config=self.model_config)
        decoder = Decoder(data=self.data, config=self.model_config, encoder=encoder)

        return EncoderDecoder(config=self.model_config, encoder=encoder, decoder=decoder)

    def build_hred(self):
        '''Build a hierarchical recurrent encoder-decoder dialogue model'''

        print("The HRED has not been implemented yet; returning a RED")

        return self.build_red()

    def get_conversation(self):
        self.load_model(self.conversation_config['model-name'])

        return Conversation(encoder=self.encoder, decoder=self.decoder, vectorizer=self.vectorizer)
