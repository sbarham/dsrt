"""
Typically, this script will be invoked with options and arguments.
When it is not, it becomes an interactive command-line tool.
"""

from dsrt.definitions import CLI_DIVIDER
from dsrt.application import Application

def run():
    Application()

    print('\n' + CLI_DIVIDER + '\n')

if __name__ == '__main__':
    # run the application
    run()
