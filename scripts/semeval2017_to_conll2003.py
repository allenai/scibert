"""

Input:
    Text:
        Based on the theoretical analysis, the value of the measuring resistor...

    Annotations:
        T1	Process 39 70	value of the measuring resistor
        T2	Process 72 74	Rm
        *	Synonym-of T2 T1
        T3	Material 427 466	NI-USB 6009 analog-to-digital converter
        T4	Material 322 336	4.7kΩ resistor
        T5	Material 375 402	saturated calomel electrode
        T6	Process 472 500	electrochemical noise signal
        T7	Material 520 547	in-house developed software
        T8	Material 778 785	dataset
        T10	Task 674 708	obtain a single value of potential
        T11	Task 979 1051	accurate recording of the potential noise in the frequencies of interest
        T12	Process 97 114	corrosion process
        T13	Process 136 161	value of noise resistance
        T14	Task 175 199	validate this conclusion
        T15	Material 257 294	pair of nominally identical specimens
        T16	Task 13 33	theoretical analysis

Output:  Entity extraction (CONLL2003 format)

"""

from typing import *

import os
import spacy
nlp = spacy.load('en_core_web_md')

from sci_bert.common.span import Span, TokenSpan, MentionSpan, label_sent_token_spans

# each instance is a single NER[split][instance_id] = {'spans': List, 'labels': List}
NER = {'train': {}, 'dev': {}, 'test': {}}
for split in ['train', 'dev', 'test']:
    print(f'Processing {split}')
    ann_dir = f'semeval2017/{split}/'

    # loop over each instance ID in this split
    instance_ids = sorted({os.path.splitext(ann_file)[0] for ann_file in os.listdir(ann_dir)})
    for id in instance_ids:
        print(f'Processing {id}')
        ann_file = os.path.join(ann_dir, f'{id}.ann')
        txt_file = ann_file.replace('.ann', '.txt')

        # load text & tokenize w/ Spacy
        with open(txt_file, 'r') as f_txt:
            text = f_txt.read().strip()
            text = ''.join([char if char.strip() != '' else ' ' for char in text])  # normalize whitespaces, such as '\xa0' --> ' '
            spacy_text = nlp(text)

        # split sentences & tokenize
        sent_token_spans = TokenSpan.find_sent_token_spans(text=text, sent_tokens=[[token.text for token in sent if token.text.strip() != ''] for sent in spacy_text.sents])

        # load annotations
        mention_spans = set()
        with open(ann_file, 'r') as f_ann:
            for line in f_ann:
                tup = line.strip('\n').split('\t')

                # load entity mention
                if tup[0].startswith('T'):
                    entity_id = tup[0]
                    # note: occasionally data looks really stupid like `Task 400 436;437 453` in `S0167931713005042.ann`
                    try:
                        entity_type, start, stop = tup[1].split(' ')
                    except ValueError as e:
                        print(f'Failed unpacking line {tup} in {id}. Skipping...')
                        continue
                    start = int(start)
                    stop = int(stop)
                    entity_text = tup[2]
                    # note: occasionally, spans include whitespace like `line = "T6	Task 220 268	modelling of red blood cells in Poiseuille flow "`
                    # where `stop=268` actually includes the `\xa0` token at end of `flow`
                    # >> correct these situations
                    if len(entity_text.strip()) != stop - start:
                        entity_text = entity_text.lstrip()
                        start += stop - start - len(entity_text)
                        entity_text = entity_text.rstrip()
                        stop -= stop - start - len(entity_text)
                        print(f'Corrected {tup} in {id} -> ({start}, {stop}) due to whitespace in mention text')
                    assert len(entity_text) == stop - start
                    mention_span = MentionSpan(start=start, stop=stop, text=entity_text, entity_types=[entity_type], entity_id=entity_id)
                    mention_spans.add(mention_span)

        # overlapping mentions are handled by picking longest mention in the group
        # this also handles same-mention multiple-types (arbitrarily picks one of them)
        clean_mention_spans = []
        clusters = Span.sort_cluster_spans(mention_spans)
        for cluster in clusters:
            if len(cluster) == 1:
                clean_mention_spans.extend(cluster)
            else:
                longest_span = sorted(cluster, key=lambda s: len(s))[-1]
                clean_mention_spans.append(longest_span)

        # conll2003 -> BIO
        sent_token_labels = label_sent_token_spans(sent_token_spans=sent_token_spans,
                                                   mention_spans=clean_mention_spans)
        # save
        NER[split][id] = {
            'spans': sent_token_spans,
            'labels': sent_token_labels
        }

for split in ['train', 'dev', 'test']:
    num_papers = len(NER[split])
    print(f'Finished processing {num_papers} papers from {split}')

# write NER
os.makedirs('data/ner/semeval2017/', exist_ok=True)
for split in ['train', 'dev', 'test']:
    with open(f'data/ner/semeval2017/{split}.txt', 'w') as f_out:
        for id, instance in NER[split].items():
            f_out.write(f'-DOCSTART- ({id})\n\n')
            for token_spans, token_labels in zip(instance['spans'], instance['labels']):
                for token_span, token_label in zip(token_spans, token_labels):
                    f_out.write('\t'.join([token_span.text, str(token_span.start), str(token_span.stop), token_label]))
                    f_out.write('\n')  # new token
                f_out.write('\n')  # new sent
            f_out.write('\n')  # new paper

