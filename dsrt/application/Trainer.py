from dsrt.data import Corpus
from dsrt.config.defaults import ModelConfig
from dsrt.application.utils import dataset_exists

class Trainer:
    def __init__(self, dataset_name=None, config=ModelConfig()):
        self.config = config
        self.dataset_name = dataset_name
        if dataset_name is None:
        	self.dataset_name = config['dataset-name']

    def run(self):
    	# Load the dataset (perhaps make a set Dataset distinct from Corpus?)
    	dataset_path = dataset_exists(dataset_name)
		corpus = Corpus(preprocessed=True, dataset_path=dataset_path)

        # Build our application context
        context = Context(model_config=self.model_config, data=self.data)

        # Build the model
        model_name = self.model_config['model-name']
        model = context.get_flat_encoder_decoder()
        context.train(model)
        context.save_model(model, model_name)
