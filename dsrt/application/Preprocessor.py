from dsrt.config.defaults import DataConfig

class Preprocessor:
    def __init__(self, config=DataConfig()):
        self.config = config
        
        return