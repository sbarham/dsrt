# from dsrt.application import Context
from dsrt.config.defaults import ModelConfig

class Trainer:
    def __init__(self, config=ModelConfig()):
        self.config = config

        return

    def run(self):
        # do stuff
        pass

    def stuff(self):
        # TODO:
        # This function and its helpers will eventually need its own module, because we'll need a better
        # solution than copying the user preferences to each config in turn
        self.load_config(args.configuration)

        # Load the corpus
        self.load_corpus(args.corpus_path, self.data_config)
        # self.data.save_dataset()

        # Build our application context
        context = Context(model_config=self.model_config, data=self.data)

        # Build the model
        model_name = self.model_config['model-name']
        model = context.get_flat_encoder_decoder()
        context.train(model)
        context.save_model(model, model_name)
