# Python stdlib
import argparse
import os
import sys

# our imports
import dsrt.application.utils
from dsrt.definitions import DEFAULT_USER_CONFIG_PATH
from dsrt.config import ConfigurationLoader
from dsrt.config.defaults import DataConfig, ModelConfig, ConversationConfig
from dsrt.application import Preprocessor, Trainer, Conversant


class Application:
    description = '''Description:
dsrt is Neural DSRT's command-line application. This is an intuitive interface for building, configuring, and running experiments with neural dialogue models.'''
    usage = '''dsrt [subcommand]

Some common subcommands include:
    train
    converse
    wizard
Try these commands with -h (or --help) for more information.'''

    def __init__(self):
        # build the primary command parser
        self.parser = argparse.ArgumentParser(description=self.description, usage=self.usage)

        # load the default configurations
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
        '''
        Note that the way we're doing this is elegant, but dangerous.
        It doesn't require, but does encourage, us to define no Application#
        methods beyond those corresponding to the supported subcommands
        '''
        # read in the subcommand
        args = parser.parse_args(sys.argv[1:2])

        # check that the subcommand is good
        if not hasattr(self, args.subcommand):
            print('Unrecognized command.')
            parser.print_help()
            exit(1)

        self.configs = ConfigurationLoader(DEFAULT_USER_CONFIG_PATH).load()

        # use dispatch pattern to invoke method with same name
        getattr(self, args.subcommand)()

    def addcorpus(self):
        '''Command to add a corpus to the dsrt library'''

        # Initialize the addcorpus subcommand's argparser
        parser = argparse.ArgumentParser(description='Preprocess a raw dialogue corpus into a dsrt dataset')
        self.init_addcorpus_args(parser)

        # parse the args we got
        args = parser.parse_args(sys.argv[2:])

        utils.import_corpus(**vars(args))


    def prepare(self):
        '''The data preprocessing subcommand, 'prepare' '''

        # Initialize the prepare subcommand's argparser
        parser = argparse.ArgumentParser(description='Preprocess a raw dialogue corpus into a dsrt dataset')
        self.init_preprocessor_args(parser)

        # Parse the args we got
        args = parser.parse_args(sys.argv[2:])
        args.config = ConfigurationLoader(args.config).load().data_config

        Preprocessor(**vars(args)).run()

    def train(self):
        '''The 'train' subcommand'''

        # Initialize the train subcommand's argparser
        parser = argparse.ArgumentParser(description='Train a dialogue model on a dialogue corpus or a dsrt dataset')
        self.init_train_args(parser)

        # Parse the args we got
        args = parser.parse_args(sys.argv[2:])
        args.config = ConfigurationLoader(args.config).load().model_config

        Trainer(**vars(args)).run()

    def converse(self):
        '''The 'converse' subcommand'''

        # Initialize the converse subcommand's argparser
        parser = argparse.ArgumentParser(description='Initiate a conversation with a trained dialogue model')
        self.init_converse_args(parser)

        # Parse the args we got
        args = parser.parse_args(sys.argv[2:])
        args.config = ConfigurationLoader(args.config).load().conversation_config

        Conversant(**vars(arg)).run()

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

    def init_addcorpus_args(self, parser):
        parser.add_argument('-f', '--corpus-path', dest='corpus_path',
                            help='the path to the corpus you wish to add')
        parser.add_argument('-n', '--name', dest='corpus_name',
                            help='the name you wish to give the corpus in dsrt\'s library')

    def init_preprocessor_args(self, parser):
        '''Only invoked conditionally if subcommand is 'prepare' '''
        parser.add_argument('-f', '--configuration', dest='config', default=DEFAULT_USER_CONFIG_PATH,
                            help='the path to the configuration file to use -- ./config.yaml by default')
        parser.add_argument('-c', '--corpus-name', help='the name of the corpus to process')

    def init_train_args(self, parser):
        '''Only invoked conditionally if subcommand is 'train' '''
        parser.add_argument('-f', '--configuration', dest='config', default=DEFAULT_USER_CONFIG_PATH,
                            help='the path to the configuration file to use -- ./config.yaml by default')
        parser.add_argument('-c', '--corpus-name', help='the name of the corpus or saved dataset to train on')

    def init_converse_args(self, parser):
        '''Only invoked conditionally if subcommand is 'converse' '''
        parser.add_argument('-f', '--configuration', dest='config', default=DEFAULT_USER_CONFIG_PATH,
                            help='the path to the configuration file to use -- ./config.yaml by default')
        parser.add_argument('-m', '--model', help='the name of the (pretrained) dialogue model to use')
