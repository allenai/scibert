from setuptools import setup, find_packages

setup(
    name='scibert',
    version='0.1.0',
    url='https://github.com/allenai/scibert.git',
    author='Iz Beltagy',
    description='A BERT model for scientific text.',
    packages=find_packages(),
    install_requires=[
        'allennlp @ git+https://github.com/ibeltagy/allennlp@fp16_and_others',
        'jsonlines',
        'lxml'
    ],
)
