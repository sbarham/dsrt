"""
Typically, this script will be invoked with options and arguments.
When it is not, it becomes an interactive command-line tool.
"""

import argparse
from dsrt.config import DataConfig, ModelConfig, ConversationConfig
from dsrt.data import Corpus
from dsrt.experiment import Context
import os
import sys

class App:
    description = '''Description:
dsrt is Neural DSRT's command-line application. This is an intuitive interface for building, configuring, and running experiments with neural dialogue models.'''
    usage = '''dsrt [subcommand]
    
Some common subcommands include:
    train
    converse
    wizard
Try these commands with -h (or --help) for more information.'''
    
    def __init__(self):
        self.parser = argparse.ArgumentParser(description=self.description, usage=self.usage)
        self.data_config = DataConfig()
        self.model_config = ModelConfig()
        self.conversation_config = ConversationConfig()
        
        # initialize primary command's arguments
        self.init_global_args(self.parser)
        
        # dispatch the subcommand (or, if it's bad, issue a usage message)
        self.dispatch_subcommand(self.parser)
    
    
    ##############################
    #   Application Subcommands  #
    ##############################
    
    def dispatch_subcommand(self, parser):
        args = parser.parse_args(sys.argv[1:2])
        
        if not hasattr(self, args.subcommand):
            print('Unrecognized command.')
            parser.print_help()
            exit(1)
            
        # use dispatch pattern to invoke method with same name
        getattr(self, args.subcommand)()
        
        return

    def train(self):
        '''The 'train' subcommand'''
        # This logic will eventually need to be moved out to its own submodule, train ...
        
        # Initialize the train subcommand's argparser
        parser = argparse.ArgumentParser(description='Train a dialogue model on a corpus of dialogues')
        self.init_train_args(parser)
        
        # Parse the args we got
        args = parser.parse_args(sys.argv[2:])
        
        # TODO:
        # This function and its helpers will eventually need its own module, because we'll need a better
        # solution than copying the user preferences to each config in turn
        self.load_config(args.configuration)
        
        # Load the corpus
        self.load_corpus(args.corpus_path, self.data_config)
        # self.data.save_dataset()
        
        # Build our application context
        context = Context(self.data_config, self.model_config, self.conversation_config, self.data)
        
        # Build the model
        model_name = self.model_config['model-name']
        model = context.get_flat_encoder_decoder()
        context.train(model)
        context.save_model(model, model_name)
    
    def converse(self):
        '''The 'converse' subcommand'''
        # This logic may eventually need to be moved out to its own submodule, converse ...
        
        # Initialize the train subcommand's argparser
        parser = argparse.ArgumentParser(description='Initiate a conversation with a trained dialogue model')
        self.init_converse_args(parser)
        
        # Parse the args we got
        args = parser.parse_args(sys.argv[2:])
        
        # TODO:
        # This function and its helpers will eventually need its own module, because we'll need a better
        # solution than copying the user preferences to each config in turn
        # self.load_config(args.configuration)
                           
        # Build our application context
        context = Context(self.data_config, self.model_config, self.conversation_config)
        
        # Get the name of the model to use and start a conversation with it
        model_name = args.model # self.model_config['model-name']
        context.get_conversation(model_name).start()
        
        return
    
    def interactive(self):
        '''The 'interactive' default subcommand, which launches an interactive CLI application'''
        
        print("This submodule has not yet been written.")
        exit(0)
        
        return
    
    
    #############################
    #   Configuration Loading   #
    #############################
    
    def load_config(self, path):
        '''Load a configuration file; eventually, support dicts, .yaml, .csv, etc.'''
        
        # if no path was provided, resort to defaults
        if path == None:
            print("Path to config was null; using defaults.")
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
        '''
        !! THIS SHOULD BE ITS OWN CLASS !!
        This is a stub for now; it should validate the settings and preferences in the config,
        raising a helpful exception if the settings are inconsistent
        '''
        return True
    
    def validate_model_config(self, config):
        '''
        !! THIS SHOULD BE ITS OWN CLASS !!
        This is a stub for now; it should validate the settings and preferences in the config,
        raising a helpful exception if the settings are inconsistent
        '''
        return True
    
    def validate_conversation_config(self, config):
        '''
        !! THIS SHOULD BE ITS OWN CLASS !!
        This is a stub for now; it should validate the settings and preferences in the config,
        raising a helpful exception if the settings are inconsistent
        '''
        return True
    
    #############################
    #      Dataset Loading      #
    #############################
    
    def load_corpus(self, path, config):
        '''Load a dialogue corpus; eventually, support pickles and potentially other formats'''
        
        # use the default dataset if no path is provided
        # TODO -- change this to use a pre-saved dataset
        if path == '':
            path = self.default_path_to_corpus
        
        self.data = Corpus(path=path, config=self.data_config)
    
    
    ###############################
    #   Argument Initialization   #
    ###############################

    def init_global_args(self, parser):
        parser.add_argument('subcommand', help='the subcommand to be run')

    def init_train_args(self, parser):
        '''Only invoked conditionally if subcommand is 'train' '''
        parser.add_argument('-C', '--configuration', help='the path to the configuration file to use')
        # eventually use names rather than paths, and store/save datasets in the archive according
        # to a consistent scheme
        parser.add_argument('-D', '--corpus-path', help='the path to the dialogue corpus to train on')
        parser.add_argument('-d', '--corpus-name', help='the name of the saved dataset to train on')

    def init_converse_args(self, parser):
        '''Only invoked conditionally if subcommand is 'converse' '''
        parser.add_argument('-m', '--model', help='the name of the (pretrained) dialogue model to use')

        
def run():
    App()
                        
if __name__ == '__main__':
    # build the application
    App()
