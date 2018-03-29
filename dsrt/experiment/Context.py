"""
Maintains a dialogue research application context, which includes (minimally) 
(1) configuration objects (which < dict()) for dataset management, model creation and management, and
    conversation management
(2) a dataset, usually in the form of a Corpus object, which represents both a corpus of dialogues
    anda sequence of transformations to apply to the raw dialogue/utterance data in order to prepare
    them for processing by our neural models
    
Using this application context, is able to construct any kind of neural dialog model desired.
"""

from keras.models import load_model
from pickle import load
from dsrt.data.transform import Vectorizer
from dsrt.models import Encoder, Decoder, EncoderDecoder
from dsrt.config import DataConfig, ModelConfig, ConversationConfig
from dsrt.conversation import Conversation
from dsrt.definitions import ROOT_DIR

class Context:
    def __init__(self, data_config=DataConfig(), model_config=ModelConfig,
                 conversation_config=ConversationConfig(), data=None):
        self.data_config = data_config
        self.model_config = model_config
        self.conversation_config = conversation_config
        self.data = data
    
    def train(self, model):
        model.fit(self.data)
        
    def save_model(self, model, model_name):
        model.save_models(model_name)
        
    def get_flat_encoder_decoder(self):
        encoder = Encoder(data=self.data, config=self.model_config)
        decoder = Decoder(data=self.data, config=self.model_config, encoder=encoder)
        
        return EncoderDecoder(config=self.model_config, encoder=encoder, decoder=decoder)
    
    def get_hierarchical_encoder_decoder(self):
        print("This method has not yet been implemented. Returning a flat dialogue model instead.")
        
        return self.get_flat_encoder_decoder(self.model_config)
    
    def get_conversation(self, model_name=None):
        # load the default (i.e., demonstration) model-name if none is provided
        if model_name is None:
            model_name = self.model_config['model-name']
        
        # load the encoder and decoder inference models
        prefix = ROOT_DIR + '/archive/models/' + model_name
        encoder = load_model(prefix + '_inference_encoder')
        decoder = load_model(prefix + '_inference_decoder')
        vectorizer = Vectorizer.load(prefix + '_vectorizer')
            
        return Conversation(encoder=encoder, decoder=decoder, vectorizer=vectorizer)
