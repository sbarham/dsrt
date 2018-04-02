'''
This script contains miscellaneous utilities needed by the command-line
application.
'''

from dsrt.definitions import LIB_DIR
import os
from shutil import copyfile
from os.path import isfile, join


########################
#   Corpus Utilities   #
########################

def list_corpus():
    corpus_dir = os.path.join(LIB_DIR, 'corpora')
    corpora = [f for f in os.listdir(corpus_dir) if os.path.isfile(os.path.join(corpus_dir, f))]

    if not corpora:
        print("No available corpora; try importing one with 'dsrt corpus add'")
        return

    print("Available corpora:")
    for f in corpora:
        print("\t* " + f)

def list_dataset():
    dataset_dir = os.path.join(LIB_DIR, 'datasets')
    datasets = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]

    if not datasets:
        print("No available datasets; try creating one by processing a corpus with 'dsrt dataset prepare'")
        return

    print("Available datasets:")
    for d in datasets:
        print("\t* " + d)


def import_corpus(src, new_name):
    if not os.path.exists(src):
        raise Error("Unable to locate corpus at '{}'".format(src))

    dst = os.path.join(LIB_DIR, 'corpora', new_name)

    if os.path.exists(dst):
        choice = input("Corpus '{}' already exists; overwrite it? y(es) | n(o): ".format(new_name))
        while True:
            if choice.lower().startswith('y'):
                break
            elif choice.lower().startswith('n'):
                print("Acknowledged; aborting command ...")
                exit(1)
            else:
                choice = input("Invalid input. Choose (y)es | (n)o: ")

    open(dst, 'a+').close()
    copyfile(src, dst)

    print("Sucessfully added corpus; exiting ...")
