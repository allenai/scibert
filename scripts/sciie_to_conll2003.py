""" Convert SciIE data to CoNLL-2003 format
SciIE: http://nlp.cs.washington.edu/sciIE/
CoNLL-2003 format: https://www.clips.uantwerpen.be/conll2003/ner/000README

A Caveat: SciIE has some nested entities (rarely occur)
CoNLL format does not support this
We assume the larger entity to be the correct one
"""
from typing import List

import click
import jsonlines
import pathlib


def _convert(data_in: List[dict]):
    """

    Args:
        data_in: a list of dictionaries
            each dict has several fields including `sentences` and `ner`
            `ner` data are list of tuples corresponding to sentences
            with following format:  #[[0, 0, 'Material'],  [10, 11, 'Method'], ...]

    Returns:
         A list of strings according to CoNll format
         e.g.: CNN NN I-NP I-Method
    """
    conll = []
    for d in data_in:
        conll.append('-DOCSTART- NN O O')
        conll.append('')
        offset = 0
        for sent_idx, sent in enumerate(d['sentences']):
            ners = d['ner'][sent_idx]  #[[0, 0, 'Material'],  [10, 10, 'OtherScientificTerm'], ...]
            sent_tags = {}
            last_index = -10
            prev_entity = ''
            for ner_idx, ner in enumerate(ners):
                beg = ner[0] - offset
                end = ner[1] - offset
                if beg == last_index and prev_entity and prev_entity == ner[2]:
                    sent_tags[beg] = f'B-{ner[2]}'
                    prev_entity = ''
                else:
                    sent_tags[beg] = f'I-{ner[2]}'
                    last_index = end + 1
                    for i in range(beg + 1, end + 1):
                        sent_tags[i] = f'I-{ner[2]}'
                        prev_entity = ner[2]
                        last_index = i
            for i in range(len(sent)):
                tag = sent_tags[i] if i in sent_tags else 'O'
                conll.append(f'{sent[i]} NN O {tag}')
            conll.append('')
            offset += len(sent)
    return conll


@click.command()
@click.argument('inpath')
@click.argument('outpath')
def convert(inpath, outpath):
    with jsonlines.open(inpath) as f_in:
        data = [e for e in f_in]

    conll = _convert(data)

    pathlib.Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, 'w') as f_out:
        for line in conll:
            f_out.write(f'{line}\n')


convert()
