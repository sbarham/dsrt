"""
Ideas:
    (x) move data transformers to a sub-package
    (2) build corpus such that it conditionally builds and passes dialogues
        through the transformers according to properties in DataConfig
"""

# Scikit-Learn imports
from sklearn.model_selection import train_test_split

# H5Py
import h5py

# NumPy
import numpy as np
from numpy import argmax

# Python stdlib
import logging
import re
import copy
import os
from math import ceil

# our own imports
from dsrt.config.defaults import DataConfig
from dsrt.data import SampleSet, Properties
from dsrt.data.transform import Tokenizer, Filter, Padder, AdjacencyPairer, Vectorizer, EncoderDecoderSplitter
from dsrt.definitions import ROOT_DIR

class Corpus:
    def __init__(self, corpus_path=None, config=DataConfig(), preprocessed=False, dataset_path=None):
        # load configuration and init path to corpus
        self.corpus_path = corpus_path
        self.config = config

        # init logger
        self.init_logger()

        self.log('info', 'Logger initialized')
        self.log('info', 'Configuration loaded')

        if preprocessed:
            # if we're loading a preprocessed dataset, exit initialization here and load it
            self.log('warn', 'Preparing to load the dialogue dataset ...')
            self.dataset_name = dataset_path if not dataset_path is None else self.config['dataset-name']
            self.load_dataset(dataset_path)
            return

        # otherwise, continue with the preprocessing pipeline ...
        self.log('warn', 'Preparing to process the dialogue corpus ...')
        self.corpus_loaded = False

        # load and tokenize the dataset
        self.tokenizer = Tokenizer(self.config)
        self.dialogues = self.load_corpus(self.corpus_path) # <-- tokenization happens here
        self.word_list, self.word_set = self.load_vocab()
        self.vocab_size = len(self.word_set)

        # gather dataset properties
        self.vectorizer = Vectorizer(self.word_list, self.config)
        self.properties = Properties(self.dialogues, self.config)
        self.add_reserved_words(self.properties, self.vectorizer)

        # create the data transformers
        self.filter = Filter(self.properties, self.config)
        self.padder = Padder(self.properties, self.config)
        self.pairer = AdjacencyPairer(self.properties, self.config)
        self.enc_dec_splitter = EncoderDecoderSplitter(self.properties, self.vectorizer, self.config)

        transformers = [self.filter, self.padder, self.pairer, self.vectorizer]

        # transform the data
        self.dialogues = self.transform(self.dialogues, transformers)
        self.train, self.test = self.train_test_split()

        # report success!
        self.log('info', 'Corpus succesfully loaded!\n')

    def init_logger(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(self.config['logging-level'])

    def load_corpus(self, path):
        self.log('info', 'Loading the dataset ...')

        dialogues = []

        if not self.corpus_loaded:
            with open(path, 'r') as f:
                dialogues = list(f)

        # if desired, retain only a subset of the dialogues:
        # if self.config['restrict-sample-size']:
        #    dialogues = np.random.choice(dialogues, self.config['sample-size'])
        # ^ this is doing WEIRD things -- perhaps it needs to be placed elsewhere,
        # where it can actually operate on a numpy array

        return self.tokenizer.transform(dialogues)

    def reload_corpus(self):
        """
        Reload the dialogue corpus, in case some changes have been made to it since we last
        loaded.
        """
        self.corpus_loaded = False
        self.load_corpus()

        return

    def load_vocab(self):
        self.log('info', 'Initializing vocabulary ...')

        reserved_words = set([
            self.config['pad-u'],
            self.config['pad-d'],
            self.config['start'],
            self.config['stop'],
            self.config['unk'],
        ])

        corpus_words = set([w for d in self.dialogues for u in d for w in u])

        vocab_set = set.union(reserved_words, corpus_words)
        vocab_list = list(vocab_set)

        vocab_size = len(vocab_list)

        return vocab_list, vocab_set

    def add_reserved_words(self, properties, vectorizer):
        properties.pad_u = vectorizer.word_to_index(self.config['pad-u'])
        properties.pad_d = vectorizer.word_to_index(self.config['pad-d'])
        properties.start = vectorizer.word_to_index(self.config['start'])
        properties.stop = vectorizer.word_to_index(self.config['stop'])
        properties.unk = vectorizer.word_to_index(self.config['unk'])

    def transform(self, dialogues, transformers):
        for i, t in enumerate(transformers):
            dialogues = t.transform(dialogues)

        return dialogues

    def train_test_split(self):
        '''
        Get a new train-test split of the Corpus' dialogues (each train and
        test samplest is represented by a SampleSet obejct
        '''
        self.log('info', 'Splitting the corpus into train/test subsets ...')

        # grab some hyperparameters from our config
        split = self.config['train-test-split']
        rand_state = self.config['random-state']

        # split the corpus into train and test samples
        # train, test = train_test_split(self.dialogues, train_size=split, random_state=rand_state)
        shuffled = np.random.permutation(self.dialogues)
        split_index = ceil(split * len(self.dialogues))
        train, test = np.split(shuffled, [split_index])
        train = np.copy(train)
        test = np.copy(test)

        train = SampleSet(train, properties=self.properties, enc_dec_splitter=self.enc_dec_splitter)
        test = SampleSet(test, properties=self.properties, enc_dec_splitter=self.enc_dec_splitter)

        return train, test


    ###################
    #    UTILITIES    #
    ###################

    def save_dataset(self, dataset_name, dataset_path):
        # record dataset hyperparameters
        # max_samples = self.config['sample-size']
        #
        # all_dialogues = self.config['use-corpus-max-dialogue-length']
        # all_utterances = self.config['use-corpus-max-utterance-length']
        #
        # max_dialogue_length = 'all' if all_dialogues else str(self.config['max-dialogue-length'])
        # max_utterance_length = 'all' if all_utterances else str(self.config['max-utterance-length'])
        #
        # # create dataset name
        # dataset_name = dataset_name + '_' + max_samples + '_' + max_dialogue_length + '_' + max_utterance_length
        # dataset_path = ROOT_DIR + '/archive/data/' + dataset_name + '/'

        # save dataset
        with h5py.File(os.path.join(dataset_path, 'data'), 'w') as f:
            f.create_dataset('vectorized_corpus', data=self.dialogues)

            self.train.save_sampleset(f=f, name='train')
            self.test.save_sampleset(f=f, name='test')

            # save the changes to disk
            f.flush()

        # save the vectorizer, which we simply pickle
        self.vectorizer.save_vectorizer(dataset_path)

    def load_dataset(self, dataset_path):
        with h5py.File(dataset_path, 'w') as f:
            self.dialogues = f['corpus']

            self.train = Sampleset(preprocessed=True, f=f, name='train')
            self.test = Sampleset(preprocessed=True, f=f, name='test')

        # load the vectorizer, which we simply pickled
        self.vectorizer = Vectorizer.load_vectorizer(dataset_path)

    def log(self, priority, msg):
        """
        Just a wrapper, for convenience.
        NB1: priority may be set to one of:
        - CRITICAL     [50]
        - ERROR        [40]
        - WARNING      [30]
        - INFO         [20]
        - DEBUG        [10]
        - NOTSET       [0]
        Anything else defaults to [20]
        NB2: the levelmap is a defaultdict stored in Config; it maps priority
             strings onto integers
        """
        self.logger.log(logging.CRITICAL, msg)

    def pretty_print_dialogue(self, dia):
        for utt in dia:
            if utt[0] == self.pad_d:
                break
            print(self.stringify_utterance(utt))

    def stringify_utterance(self, utt):
        return ' '.join([w for w in utt if not w == self.pad_u])
