import os

from dsrt.definitions import LIB_DIR
from dsrt.config.defaults import DataConfig
from dsrt.data import Corpus

class Preprocessor:
    def __init__(self, corpus_name=None, dataset_name=None, config=DataConfig()):
        self.config = config

        if corpus_name is None:
            corpus_name = config['corpus-name']
        if dataset_name is None:
            dataset_name = config['dataset-name']

        self.corpus_path = os.path.join(LIB_DIR, 'corpora', corpus_name)
        self.dataset_name = dataset_name
        self.dataset_path = os.path.join(LIB_DIR, 'datasets', dataset_name)

        if not os.path.exists(self.dataset_path):
            choice = input("Dataset '{}' already exists; overwrite it? y(es) | n(o): ".format(dataset_name))
            while True:
                if choice.lower().startswith('y'):
                    os.makedirs(self.dataset_path)
                    break
                elif choice.lower().startswith('n'):
                    print("Acknowledged; aborting command ...")
                    exit(1)
                else:
                    choice = input("Invalid input. Choose (y)es | (n)o: ")

    def run(self):
        print("Processing corpus ...")

        self.corpus = Corpus(self.corpus_path, self.config)

        print("Processing successful; saving corpus ...")

        self.corpus.save_dataset(self.dataset_name, self.dataset_path)

        print("Done; exiting successfully.")
