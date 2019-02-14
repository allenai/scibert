"""
This script converts Jurgens citation data format to the generic text classification format
the generic format is according to the following:
jsonlines with the following format:{'text': ... , 'label': ...., 'metadata: {}}
"""
import pathlib

import click
import json


@click.command()
@click.argument('inpath')
@click.argument('outpath')
def convert(inpath, outpath):

    pathlib.Path(outpath).parent.mkdir(parents=True, exist_ok=True)

    data = []
    with open(outpath, 'w') as f_out:
        with open(inpath) as f_in:
            for line in f_in:
                obj = json.loads(line)
                new_obj = {'text': obj['text'],
                           'label': obj['intent'],
                           'metadata': {}}
                f_out.write(f'{json.dumps(new_obj)}\n')
convert()
