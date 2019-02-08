"""
Script to run the allennlp model
"""
import logging
import sys
from pathlib import Path
import os

# from allennlp.commands import main
from allennlp.commands import main


sys.path.append(str(Path().absolute()))
sys.path.append(str(Path().absolute().parent))

import sci_bert.models.ner_model
import sci_bert.data_readers.conll_data_reader

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)



if __name__ == "__main__":
    # Make sure a new predictor is imported in processes/__init__.py
    main(prog="python -m allennlp.run")

