from dsrt.data import DataSet
from dsrt.config.defaults import ModelConfig
from dsrt.application import Context
from dsrt.application.utils import dataset_exists

class Trainer:
    def __init__(self, dataset_name=None, saved_model_name=None, new_model_name=None, num_gpus=1, config=ModelConfig()):
        self.config = config
        self.num_gpus = num_gpus
        self.dataset_name = dataset_name
        self.saved_model_name = saved_model_name
        
        if dataset_name is None:
            self.dataset_name = config['dataset-name']
        if new_model_name is None:
            self.new_model_name = config['model-name']

    def run(self):
        # Load the dataset (perhaps make a set Dataset distinct from Corpus?)
        dataset_path = dataset_exists(self.dataset_name)
        dataset = DataSet()
        dataset.load_dataset(dataset_path)

        # Build our application context
 
        context = Context(model_config=self.config, dataset=dataset, num_gpus=self.num_gpus)

        # Build or load the model
        if self.saved_model_name is None:
            context.build_model()
        else:
            context.load_model(self.saved_model_name)

        # train and save the model
        context.train()
        context.save_model(self.new_model_name)
