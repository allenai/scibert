"""
Export the relations part of the ChemProt dataset
"""

import click
import jsonlines
from lxml import etree
import pathlib

@click.command()
@click.argument('inpath')
@click.argument('outpath')
@click.argument('with_entity_markers')
def main(inpath, outpath, with_entity_markers):
    """
    Args:
        inpath: input file from chemprot in xml format
        outpath: output file with relations information
        with_entity_markers: True/False, if True, highlight entities in the string
    """
    pathlib.Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(outpath, 'w') as fout:
        doc = etree.parse(inpath)
        root = doc.getroot()
        for document in root:
            for sentence in document:
                text = sentence.attrib['text']
                entities = {}
                for child in sentence:
                    if child.tag == 'entity':
                        entity = child.attrib
                        char_offset_from, char_offset_to = entity['charOffset'].split('-')
                        entities[entity['id']] = {'from': int(char_offset_from), 'to': int(char_offset_to)}
                    elif child.tag == 'interaction':
                        relation = child.attrib
                        e1 = entities.get(relation['e1'])
                        e2 = entities.get(relation['e2'])
                        if e1 is None or e2 is None:
                            print('Missing entities')
                            continue
                        rel_type = relation['relType']
                        if rel_type not in ['AGONIST-ACTIVATOR', 'DOWNREGULATOR', 'SUBSTRATE_PRODUCT-OF',
                                            'AGONIST', 'INHIBITOR', 'PRODUCT-OF', 'ANTAGONIST', 'ACTIVATOR',
                                            'INDIRECT-UPREGULATOR', 'SUBSTRATE', 'INDIRECT-DOWNREGULATOR',
                                            'AGONIST-INHIBITOR', 'UPREGULATOR', ]:
                            continue
                        if e1['from'] < e2['from']:
                            text_with_markers = "".join([
                                    text[:e1['from']],
                                    '<< ', text[e1['from']:e1['to']], ' >>',
                                    text[e1['to']:e2['from']],
                                    '[[ ',text[e2['from']:e2['to']], ' ]]',
                                    text[e2['to']:]
                            ])
                        else:
                            text_with_markers = "".join([
                                    text[:e2['from']],
                                    '<< ', text[e2['from']:e2['to']], ' >>',
                                    text[e2['to']:e1['from']],
                                    '[[ ',text[e1['from']:e1['to']], ' ]]',
                                    text[e1['to']:]
                            ])

                        d = {'text': text_with_markers, 'label': rel_type, 'metadata': []}
                        fout.write(d)

main()
