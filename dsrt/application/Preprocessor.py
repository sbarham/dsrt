import os
import shutil
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

        if os.path.exists(self.dataset_path):
            choice = input("Dataset '{}' already exists; overwrite it? y(es) | n(o): ".format(dataset_name))
            while True:
                if choice.lower().startswith('y'):
                    shutil.rmtree(self.dataset_path)
                    os.makedirs(self.dataset_path)
                    break
                elif choice.lower().startswith('n'):
                    print("Acknowledged; aborting command ...")
                    exit(1)
                else:
                    choice = input("Invalid input. Choose (y)es | (n)o: ")
        else:
            os.makedirs(self.dataset_path)

    def run(self):
        print("Loading corpus ...")

        corpus = Corpus(self.corpus_path, self.config)
        
        print("Processing corpus ...")
        
        corpus.prepare_dataset()

        print("Processing successful; saving corpus ...")

        corpus.dataset.save_dataset(self.dataset_name, self.dataset_path)

        print("Done; exiting successfully.")
