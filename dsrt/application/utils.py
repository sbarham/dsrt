'''
This script contains miscellaneous utilities needed by the command-line
application.
'''

from dsrt.definitions import LIB_DIR
import os
from shutil import copyfile

def import_corpus(corpus_path, corpus_name):
    if not os.path.exists(corpus_path):
        raise Error("Unable to locate corpus at '{}'".format(corpus_path))

    copyfile(corpus_path, os.path.join(LIB_DIR, 'corpora', corpus_name))
