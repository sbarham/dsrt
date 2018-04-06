import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
LIB_DIR = os.path.join(ROOT_DIR, 'library')
CHECKPOINT_DIR = os.path.join(LIB_DIR, 'tmp', 'checkpoints')
DEFAULT_USER_CONFIG_PATH = './config.yaml'

CLI_DIVIDER = '--------------------------------'
