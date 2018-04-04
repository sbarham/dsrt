from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    long_description = f.read()
with open('CLASSIFIERS', 'r') as f:
    classifiers = f.readline().splitlines()

setup(
    name="dsrt",
    version="0.0.1",
    packages=find_packages(),
    license='MIT',

    description='High-level library for building and testing neural dialogue systems',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/sbarham/dsrt',
    classifiers=classifiers,

    python_requires='>=3',
    
    install_requires=[
        'dsrt',
        'h5py'
    ],

    entry_points={
        'console_scripts': [
            'dsrt = dsrt.cli:run'
        ]
    },
    
    author="Samuel Barham, Nirat Saini",
    author_email="sbarham@terpmail.umd.edu, nirat@cs.umd.edu",

    keywords='AI artificial intelligence neural dialogue systems chat chatbot chatterbot nlp nautral language generation human computer interaction conversation',

    project_urls={
        'Documentation': 'https://github.com/sbarham/dsrt',
        'Source': 'https://github.com/sbarham/dsrt',
        'Tracker': 'https://github.com/sbarham/dsrt/issues'
    }
)
