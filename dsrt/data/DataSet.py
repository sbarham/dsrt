

class DataSet:
    def __init__(self, data=None, train=None, test=None, vectorizer=None):
        self.data = data
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

        # save the vectorizer, which we simply pickle
        self.vectorizer.save_vectorizer(dataset_path)

    def load_dataset(self, dataset_path):
        with h5py.File(dataset_path, 'w') as f:
            self.dialogues = f['corpus']

            self.train = Sampleset(preprocessed=True, f=f, name='train')
            self.test = Sampleset(preprocessed=True, f=f, name='test')

        # load the vectorizer, which we simply pickled
        self.vectorizer = Vectorizer.load_vectorizer(dataset_path)
