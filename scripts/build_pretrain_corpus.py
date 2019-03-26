"""
Convert papers records into plain text files with one sentence per line.
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # disable numpy parallelization

import json
import plac
import datetime
import json
import multiprocessing as mp
import functools
import sys
import functools
import spacy
import re
import s2base
import gzip
import datetime



@plac.annotations(
    s3_in_dir=("s3 input directory", "positional", None, str),
    out_dir=("output directory", "positional",  None, str),
    pool_size=("number of workers", "option", "pool_size", int),
    num_parts=("number of partitions of the input file", "option",  "num_parts", int),
    start_part=("start partition", "option",  "start_part", int),
    end_part=("end partition", "option",  "end_part", int),
    max_paper_count=("maximum paper count per file", "option",  "max_paper_count", int),
    with_body=("include body or `abstract only", 'flag', 'with_body'))
def main(s3_in_dir, out_dir, pool_size=2, num_parts=6000, start_part=0, end_part=1, max_paper_count=10, with_body=False):
    assert start_part <= end_part
    assert end_part <= num_parts

    jobs = [{'part_id': p, 's3_in_dir': s3_in_dir, 'out_dir': out_dir,
             'max_paper_count': max_paper_count, 'with_body': with_body
            } for p in range(start_part, end_part + 1)]

    if pool_size == 1:
        results = [process_paper_file(job['part_id'], job['s3_in_dir'], job['out_dir'], job['max_paper_count'], job['with_body'])for job in jobs]
    else:
        with mp.Pool(processes=pool_size) as pool:
            pool.map(process, jobs)

    print('Done processing')


def process(job):
    process_paper_file(job['part_id'], job['s3_in_dir'], job['out_dir'], job['max_paper_count'], job['with_body'])


def process_paper_file(part_id, s3_in_dir, out_dir, max_paper_count, with_body):
    print("{} Processing job {}".format(datetime.datetime.now(), part_id))
    s3_in_filename = "{0}/part-{1:05d}.gz".format(s3_in_dir, part_id)
    # import ipdb; ipdb.set_trace()
    local_in_filename = s2base.file_util.cache_file(s3_in_filename)
    out_filename = "{}/{}.out".format(out_dir, part_id)
    with gzip.open(local_in_filename) as in_file:
        with open(out_filename, 'w') as out_file:
            paper_count = 0
            for line in in_file:
                paper_record = json.loads(line.decode('utf-8'))
                # print(paper_count, end='')
                process_paper_record(paper_record, out_file, with_body)
                paper_count += 1
                if paper_count % 5 == 0:
                    print("{} Job {} at paper {}".format(datetime.datetime.now(), part_id, paper_count))
                if paper_count >= max_paper_count:
                    break
            out_file.write("DONE")
    print("{} DONE Processing job {} with {} papers".format(datetime.datetime.now(), part_id, paper_count))

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


def process_paper_record(paper_record, out_file, with_body):
    sentences = _paper_record_to_sentences(paper_record, with_body)
    # print(" paper {} with {} sents".format(paper_record['id'], len(sentences)))
    for s in sentences:
        s_text = re.sub('\s+', ' ', s.text).strip()
        if s_text != "":
            out_file.write("{}\n".format(s_text))
    if len(sentences) > 0:
        out_file.write("\n")
plac.call(main, sys.argv[1:])
