# from dsrt.application import Context

class Conversant:
    def __init__(self, conversation_config):
        self.context = Context(conversation_config=conversation_config)
        self.conversation = context.get_conversation()

    def run(self):
        self.conversation.start()
