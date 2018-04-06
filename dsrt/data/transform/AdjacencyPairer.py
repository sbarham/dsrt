import tqdm
from multiprocessing import Pool
import logging
from dsrt.config.defaults import DataConfig

class AdjacencyPairer:
    def __init__(self, properties, parallel=True, config=DataConfig()):
        self.properties = properties
        self.parallel = parallel
        self.config = config
        
        # initialize the logger
        self.init_logger()
    
    def init_logger(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(self.config['logging-level'])
    
    def transform(self, dialogues):
        self.log('info', 'Flattening dialogues into adjacency pairs ...')

        chunksize=self.config['chunksize']
        p = Pool() if self.parallel else Pool(1)
        res = []
        total = len(dialogues)

        for d in tqdm.tqdm(p.imap(self.dialogue_to_adjacency_pairs, dialogues, chunksize=chunksize), total=total):
            res += d

        p.close()
        p.join()
        
        return res

    def dialogues_to_adjacency_pairs(self, dialogues):
        return [ap for d in dialogues for ap in self.dialogue_to_adjacency_pairs(d)]

    def dialogue_to_adjacency_pairs(self, dialogue):
        adjacency_pairs = []
        for i in range(len(dialogue)):
            if i + 1 < len(dialogue):
                adjacency_pairs += [[dialogue[i], dialogue[i + 1]]]
        
        return adjacency_pairs
    
    ####################
    #     UTILITIES    #
    ####################
    
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
