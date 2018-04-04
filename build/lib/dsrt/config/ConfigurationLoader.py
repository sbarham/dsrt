# Python stdlib
import os

# our imports
from dsrt.config import Configuration
from dsrt.config.defaults import DataConfig, ModelConfig, ConversationConfig
from dsrt.config.validation import DataConfigValidator, ModelConfigValidator, ConversationConfigValidator

class ConfigurationLoader:
    def __init__(self, path):
        # initialize defaults
        self.data_config = DataConfig()
        self.model_config = ModelConfig()
        self.conversation_config = ConversationConfig()

        self.configuration = Configuration(self.data_config, self.model_config, self.conversation_config)

        self.path = path

    def load(self):
        self.load_config(self.path)
        return self.configuration

    def load_config(self, path):
        '''Load a configuration file; eventually, support dicts, .yaml, .csv, etc.'''

        # if no path was provided, resort to defaults
        if path == None:
            print("Path to config was null; using defaults.")
            return

        if not os.path.exists(path):
            print("[No user config file found at default location; using defaults.]\n")
            return


        user_config = None

        # read in the user's configuration file (for now, we hope it's yaml)
        with open(path) as f:
            user_config = f.read()

        # load the user's configuration file into a Config object
        extension = os.path.splitext(path)
        if extension == 'yaml':
            user_config = yaml.load(user_config)
        else:
            raise Error('Configuration file type "{}" not supported'.format(extension))

        # copy the user's preferences into the default configurations
        self.merge_config(user_config)

        # build the udpated configuration
        self.configuration = Configuration(self.data_config, self.model_config, self.conversation_config)

    def merge_config(self, user_config):
        '''
        Take a dictionary of user preferences and use them to update the default
        data, model, and conversation configurations.
        '''

        # provisioanlly update the default configurations with the user preferences
        temp_data_config = copy.deepcopy(self.data_config).update(user_config)
        temp_model_config = copy.deepcopy(self.model_config).update(user_config)
        temp_conversation_config = copy.deepcopy(self.conversation_config).update(user_config)

        # if the new configurations validate, apply them
        if validate_data_config(temp_data_config):
            self.data_config = temp_data_config
        if validate_model_config(temp_model_config):
            self.model_config = temp_model_config
        if validate_conversation_config(temp_conversation_config):
            self.conversation_config = temp_conversation_config

    def validate_data_config(self, config):
        return DataConfigValidator(config).validate()

    def validate_model_config(self, config):
        return ModelConfigValidator(config).validate()

    def validate_conversation_config(self, config):
        return ConversationConfigValidator(config).validate()
