"""
Export the relations part of the SciIE dataset
SciIE: http://nlp.cs.washington.edu/sciIE/
"""

import click
import jsonlines
import pathlib

@click.command()
@click.argument('inpath')
@click.argument('outpath')
@click.argument('with_entity_markers')
def main(inpath, outpath, with_entity_markers):
    """
    Args:
        inpath: input file from sciie
        outpath: output file with relations information
        with_entity_markers: True/False, if True, highlight entities in the string
    """

    pathlib.Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(outpath, 'w') as fout:
        with jsonlines.open(inpath) as fin:
            for entry in fin:
                sent_start_index = 0
                for original_sent, rels in zip(entry['sentences'], entry['relations']):
                    for rel in rels:
                        sent = list(original_sent)
                        e1_from, e1_to, e2_from, e2_to, rel_type = rel
                        e1_from -= sent_start_index
                        e1_to -= sent_start_index
                        e2_from -= sent_start_index
                        e2_to -= sent_start_index
                        if with_entity_markers == 'True':
                            if e2_to > e1_to:
                                sent.insert(e2_to + 1, '>>')
                                sent.insert(e2_from, '<<')
                                sent.insert(e1_to + 1, ']]')
                                sent.insert(e1_from, '[[')
                            else:
                                sent.insert(e1_to + 1, ']]')
                                sent.insert(e1_from, '[[')
                                sent.insert(e2_to + 1, '>>')
                                sent.insert(e2_from, '<<')

                        d = {'text': ' '.join(sent), 'label': rel_type, 'metadata': [e1_from, e1_to, e2_from, e2_to]}
                        fout.write(d)
                    sent_start_index += len(original_sent)

main()
