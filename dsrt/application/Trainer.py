from dsrt.data import DataSet
from dsrt.config.defaults import ModelConfig
from dsrt.application.utils import dataset_exists

class Trainer:
    def __init__(self, dataset_name=None, saved_model_name=None, new_model_name=None,
                config=ModelConfig()):
        self.config = config
        self.dataset_name = dataset_name
        self.saved_model_name = saved_model_name
        if dataset_name is None:
        	self.dataset_name = config['dataset-name']
        if new_model_name is None:
            self.new_model_name = config['model-name']

    def run(self):
    	# Load the dataset (perhaps make a set Dataset distinct from Corpus?)
    	dataset_path = dataset_exists(dataset_name)
		dataset = DataSet().load_dataset(dataset_path)

        # Build our application context
        context = Context(model_config=self.model_config, data=self.data)

        # Build or load the model
        if self.saved_model_name is None:
            context.build_model(self.config)
        else:
            context.load_model(old_model_name)

        # train and save the model
        context.train(model)
        context.save_model(model, self.model_name)
