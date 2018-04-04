from dsrt.config.defaults import DataConfig, ModelConfig, ConversationConfig

class Configuration:
    def __init__(self, data_config=DataConfig(), model_config=ModelConfig(), conversation_config=ConversationConfig()):
        self.data_config = data_config
        self.model_config = model_config
        self.conversation_config = conversation_config