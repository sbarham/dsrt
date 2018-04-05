import logging
from dsrt.config.defaults import DataConfig

class AdjacencyPairer:
    def __init__(self, properties, config=DataConfig()):        
        self.properties = properties
        self.config = config
        
        # initialize the logger
        self.init_logger()
    
    def init_logger(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(self.config['logging-level'])
    
    def transform(self, dialogues):
        self.log('info', 'Flattening dialogues into adjacency pairs ...')
        return self.dialogues_to_adjacency_pairs(dialogues)

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