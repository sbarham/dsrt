from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import numpy as np
from numpy import argmax

import logging
import os
import pickle
import copy
import tqdm
from multiprocessing import Pool

from dsrt.config.defaults import DataConfig

class Vectorizer:
    def __init__(self, vocab_list, properties=None, parallel=True, config=DataConfig()):
        self.config = config
        self.vocab_list = vocab_list
        self.properties = properties
        self.parallel = parallel

        # initialize the logger
        self.init_logger()

        # reserved vocabulary items
        self.pad_u = config['pad-u']
        self.pad_d = config['pad-d']
        self.start = config['start']
        self.stop = config['stop']
        self.unk = config['unk']

        # initialize the integer- and OHE-encoders
        self.init_encoders()

    def init_logger(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(self.config['logging-level'])

    def init_encoders(self, config=DataConfig()):
        """
        Initialize the integer encoder and the one-hot encoder, fitting them to the vocabulary
        of the corpus.
        """
        self.log('info', 'Initializing the encoders ...')

        # create the integer encoder and fit it to our corpus' vocab
        self.ie = LabelEncoder()
        self.ie_vocab = self.ie.fit_transform(self.vocab_list)

    def transform(self, dialogues, ohe=False, from_corpus=True):
        self.log('info', 'Vectorizing dialogues (this may take a while) ...')

        chunksize=self.config['chunksize']
        p = Pool() if self.parallel else Pool(1)
        res = []
        total=len(dialogues)

        self.log('info', '[vectorizer operating on {} cores]'.format(p._processes))

        if from_corpus:
            for d in tqdm.tqdm(p.imap(self.vectorize_dialogue_unsafe, dialogues, chunksize=chunksize), total=total):
                res.append(d)
        else:
            for d in tqdm.tqdm(p.imap(self.vectorize_dialogue_safe, dialogues, chunksize=chunksize), total=total):
                res.append(d)

        p.close()
        p.join()
        
        return np.array(res)


    #################################
    #     INTEGER VECTORIZATION     #
    #################################

    def vectorize_dialogues(self, dialogues):
        """
        Take in a list of dialogues and vectorize them all
        """
        return [self.vectorize_dialogue(d) for d in dialogues]

    def vectorize_dialogue(self, dialogue, safe=True):
        """
        Take in a dialogue (a sequence of tokenized utterances) and transform it into a
        sequence of sequences of indices
        """
        if safe:
            return vectorize_dialogue_safe(dialogue)
        else:
            return vectorize_dialogue_unsafe(dialogue)
        
    def vectorize_dialogue_safe(self, dialogue):
        return [self.vectorize_utterance_safe(u) for u in dialogue]
    
    def vectorize_dialogue_unsafe(self, dialogue):
        return [self.vectorize_utterance_unsafe(u) for u in dialogue]


    def vectorize_utterance(self, utterance, safe=True):
        """
        Take in a tokenized utterance and transform it into a sequence of indices
        """
        if safe:
            return self.vectorize_utterance_safe(utterance)
        else:
            return self.vectorize_utterance_unsafe(utterance)
        
    def vectorize_utterance_safe(self, utterance):
        for i, word in enumerate(utterance):
            if not word in self.vocab_list:
                utterance[i] = '<unk>'

        return self.ie.transform(utterance)
        
    def vectorize_utterance_unsafe(self, utterance):
        return self.ie.transform(utterance)

    def devectorize_dialogues(self, dialogues):
        """
        Take in a sequence of integer-encoded dialogues and transform them into tokenized dialogues
        """
        return [self.devectorize_dialogue(d) for d in dialogues]

    def devectorize_dialogue(self, dialogue):
        """
        Take in a dialogue of integer-encoded utterances and transform them into a tokenized dialogue
        """
        return [self.devectorize_utterance(u) for u in dialogue]

    def devectorize_utterance(self, utterance):
        """
        Take in a sequence of indices and transform it back into a tokenized utterance
        """
        if not utterance:
            return []
        
        return self.ie.inverse_transform(utterance).tolist()

    def word_to_index(self, word):
        return self.ie.transform([word])[0]

    def index_to_word(self, index):
        return self.ie.inverse_transform([index])[0]
    

    ####################
    #     UTILITIES    #
    ####################

    def save_vectorizer(self, path):
        with open(os.path.join(path, 'vectorizer'), 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_vectorizer(path):
        with open(os.path.join(path, 'vectorizer'), 'rb') as f:
            return pickle.load(f)

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
