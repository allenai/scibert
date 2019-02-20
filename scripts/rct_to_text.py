"""
Script to convert Pubmed RCT dataset to textual format for sent classification
"""

import jsonlines
import click
import pathlib

@click.command()
@click.argument('inpath')
@click.argument('outpath')
def convert(inpath, outpath):

    pathlib.Path(outpath).parent.mkdir(parents=True, exist_ok=True)

    with open(inpath) as f_in:
        with jsonlines.open(outpath, 'w') as f_out:
            for line in f_in:
                abstract_id = ''
                line = line.strip()
                if not line:
                    continue
                if line.startswith('###'):
                    abstract_id = line
                    continue
                label, sent = line.split('\t')
                f_out.write({'label': label, 'text': sent, 'metadata':abstract_id})


convert()
