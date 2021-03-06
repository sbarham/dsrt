from collections import defaultdict
from dsrt.definitions import LIB_DIR
import os
import logging

class BaseConfig(dict):
    def __init__(self):
        # these are more or less global parameters
        self['library-dir'] = LIB_DIR
        self['corpora-dir'] = os.path.join(LIB_DIR, 'corpora')
        self['datasets-dir'] = os.path.join(LIB_DIR, 'datasets')
        self['models-dir'] = os.path.join(LIB_DIR, 'models')
        self['corpus-name'] = 'test'
        self['dataset-name'] = 'test'
        self['model-name'] = 'test'

        # reserved keywords
        self['start'] = '<start>'
        self['stop'] = '<stop>'
        self['pad-d'] = '<pad_d>'
        self['pad-u'] = '<pad_u>'
        self['unk'] = '<unk>'

        # logging level -- may be set to one of:
        # - CRITICAL     [50]
        # - ERROR        [40]
        # - WARNING      [30]
        # - INFO         [20]
        # - DEBUG        [10]
        # - NOTSET       [0]
        # Either str or int is acceptable
        self['logging-level'] = logging.INFO
        self.init_levelmap()

    def init_levelmap(self):
        self.levelmap = dict(
            #lambda: 20,
            {
                'CRITICAL': 50,
                'critical': 50,
                'ERROR': 50,
                'error': 50,
                'WARNING': 50,
                'warning': 50,
                'INFO': 50,
                'info': 50,
                'DEBUG': 50,
                'debug': 50
            })
