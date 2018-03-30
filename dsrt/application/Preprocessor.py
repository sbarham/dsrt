from dsrt.config import DataConfig

class Preprocessor:
    def __init__(self, config=DataConfig()):
        self.config = config
        
        return