from dsrt.config.defaults import ConversationConfig
from dsrt.application import Context

class Conversant:
    def __init__(self, model_name=None, config=ConversationConfig):
        self.config = config
        self.model_name = model_name
        if self.model_name is None:
            self.model_name = config['model-name']
        
        self.context = Context(conversation_config=config)
        self.conversation = self.context.get_conversation(self.model_name)

    def run(self):
        self.conversation.start()
