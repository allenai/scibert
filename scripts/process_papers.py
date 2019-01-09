import json
import plac
from distil.utils import s2_utils
from distil import paper_faq_wrapper
from distil import answer_finder_baseline
from distil.allennlp import interface
import datetime
import json
import multiprocessing as mp
import functools
import sys
import os
import functools
import spacy
import re

os.environ['OPENBLAS_NUM_THREADS'] = '1'  # disable numpy parallelization


@plac.annotations(
    infilename=("input file (elastic search buildpaper file)", "positional", None, str),
    outfilename=("output file", "positional",  None, str),
    max_paper_count=("maximum paper count", "option",  "max_paper_count", int),
    with_body=("include body or a`bstract only", 'flag', 'with_body'))
def main(infilename, outfilename, max_paper_count=10, with_body=False):
    with open(infilename) as infile:
        with open(outfilename, 'w') as outfile:
            paper_count = 0
            for line in infile:
                paper_record = json.loads(line)
                print(paper_count, end='')
                process_paper_record(paper_record, outfile, with_body)
                paper_count += 1
                if paper_count >= max_paper_count:
                    break


@functools.lru_cache()
def _get_spacy_nlp():
    nlp = spacy.load('en_scispacy_core_web_sm', disable=['ner', 'tagger'])
    # nlp = spacy.load('en_core_web_sm', disable=['ner', 'tagger'])
    return nlp


def get_spacy_sentences(doc_text):
    """
    Split given document into its sentences
    :param doc_text: Text to tokenize
    :return: list of spacy sentences
    """
    doc = _get_spacy_nlp()(doc_text)
    return list(doc.sents)


def _paper_record_to_sentences(paper_record, with_body):
        abstract = paper_record.get('paperAbstract') or paper_record.get('abstract') or ''

        if with_body:
            text = paper_record.get('bodyText') or ''
        else:
            text = ''

        abstract = abstract.strip()
        if abstract:
            if abstract[-1] != '.':
                abstract += '. '
            text = abstract + text

        return get_spacy_sentences(text)


MIN_TOKEN_COUNT = 5
MIN_WORD_TOKENS_RATIO = 0.40  # percentage of words in the tokens
MIN_LETTER_CHAR_RATIO = 0.60  # percentage of letters in the string


def is_sentence(spacy_sentence):
    """
    Checks if the string is an English sentence.
    :param sentence: spacy string
    :return: True / False
    """


    tokens = [t for t in spacy_sentence.doc]

    # Minimum number of words per sentence
    if len(tokens) < MIN_TOKEN_COUNT:
        return False

    # Most tokens should be words
    if sum([t.is_alpha for t in tokens]) / len(tokens) < MIN_WORD_TOKENS_RATIO:
        return False

    text = spacy_sentence.text

    # Most characters should be letters, not numbers and not special characters
    if sum([c.isalpha() for c in text]) / len(text) < MIN_LETTER_CHAR_RATIO:
        return False

    return True


def process_paper_record(paper_record, outfile, with_body):
    sentences = _paper_record_to_sentences(paper_record, with_body)
    print(" paper {} with {} sents".format(paper_record['id'], len(sentences)))
    for s in sentences:
        s_text = re.sub('\s+', ' ', s.text).strip()
        if s_text != "":
            outfile.write("{}\n".format(s_text))
    if len(sentences) > 0:
        outfile.write("\n")
plac.call(main, sys.argv[1:])
