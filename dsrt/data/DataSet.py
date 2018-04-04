import h5py
import os
from dsrt.data import SampleSet, Properties
from dsrt.data.transform import Vectorizer

class DataSet:
    def __init__(self, dialogues=None, properties=None, train=None, test=None, vectorizer=None):
        self.dialogues = dialogues
        self.properties = properties
        self.train = train
        self.test = test
        self.vectorizer = vectorizer

    def save_dataset(self, dataset_name, dataset_path):
        # save dataset
        with h5py.File(os.path.join(dataset_path, 'data'), 'w') as f:
            f.create_dataset('vectorized_corpus', data=self.dialogues)

            self.train.save_sampleset(f=f, name='train')
            self.test.save_sampleset(f=f, name='test')

            # save the changes to disk
            f.flush()

        # save the vectorizer and properties, which we simply pickle
        self.vectorizer.save_vectorizer(dataset_path)
        self.properties.save_properties(dataset_path)

    def load_dataset(self, dataset_path):
        with h5py.File(os.path.join(dataset_path, 'data'), 'r') as f:            
            self.dialogues = f['vectorized_corpus']

            self.train = SampleSet()
            self.train.load_sampleset(f=f, name='train')
            
            self.test = SampleSet()
            self.test.load_sampleset(f=f, name='test')

        # load the vectorizer, which we simply pickled
        self.vectorizer = Vectorizer.load_vectorizer(dataset_path)
        self.properties = Properties.load_properties(dataset_path)
