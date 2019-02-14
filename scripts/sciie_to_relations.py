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
def main(inpath, outpath):
    """
    Args:
        inpath: input file from sciie
        outpath: output file with relations information
    """

    pathlib.Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(outpath, 'w') as fout:
        with jsonlines.open(inpath) as fin:
            for entry in fin:
                for sent, rels in zip(entry['sentences'], entry['relations']):
                    for rel in rels:
                        e1_from, e1_to, e2_from, e2_to, rel_type = rel
                        d = {'text': " ".join(sent), 'label': rel_type, 'metadata': [e1_from, e1_to, e2_from, e2_to]}
                        fout.write(d)

main()
