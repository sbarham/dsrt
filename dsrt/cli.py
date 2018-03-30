"""
Typically, this script will be invoked with options and arguments.
When it is not, it becomes an interactive command-line tool.
"""

from dsrt.application import Application
        
def run():
    Application()
                        
if __name__ == '__main__':
    # run the application
    run()
    
