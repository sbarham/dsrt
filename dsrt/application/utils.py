'''
This script contains miscellaneous utilities needed by the command-line
application.
'''

from dsrt.definitions import LIB_DIR
import os
from shutil import copyfile

def import_corpus(src, new_name):
    if not os.path.exists(src):
        raise Error("Unable to locate corpus at '{}'".format(src))

    dst = os.path.join(LIB_DIR, 'corpora', new_name)

    if os.path.exists(dst):
        choice = input("Corpus '{}' already exists; overwrite it? y(es) | n(o): ")
        while True:
            if choice.lower().startswith('y'):
                break
            elif choice.lower().startswith('n'):
                print("Acknowledged; aborting command ...")
            else:
                choice = input("Invalid input. Choose (y)es | (n)o: ")

    open(dst, 'w+').close()
    copyfile(src, dst)
