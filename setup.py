from setuptools import setup

setup(
    name="dsrt",
    version="0.1",
    packages=['dsrt'],
    install_requires=[
        'dsrt',
        'dill'
    ],

    entry_points={
        'console_scripts': [
            'dsrt = dsrt.cli:run'
        ]
    },
    
    author="Samuel Barham, Nirat Saini",
    author_email="sbarham@terpmail.umd.edu, nirat@cs.umd.edu",
)
