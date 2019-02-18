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

Output 1:  Entity extraction (CONLL2003 format)


Output 2:  Relation classification
    JSONlines where each JSON is:
    {
        "text": "Based on the theoretical analysis , the [[ value of the measuring resistor ]] ...",
        "label": "Synonym-of",
        "metadata": [7, 11, 12, 12]
    }
    ** only contains sentences with relations; entities are gold wrt their Arg1/Arg2 position; no cross-sentence relations **

Output 2:  Relation extraction
    ** same format as above, but includes sentences without relations & all entity pairs are attempted as Arg1/Arg2 positions **



Pseudocode for 1:
    - sentence-ify & tokenize text & assign char-level span indices to each token
    - for each sentence:
        for each token:
            if within entity, do stuff


"""

from typing import *

import os
import spacy
nlp = spacy.load('en_core_web_md')

from sci_bert.common.span import Span, TokenSpan, MentionSpan, label_sent_token_spans

MIN_NUM_TOKENS_PER_SENT = 5

os.makedirs('data/ner/semeval2017/', exist_ok=True)
for split in ['train', 'dev', 'test']:
    with open(f'data/ner/semeval2017/{split}.txt', 'w') as f_out:
        ann_dir = f'semeval2017/{split}/'
        instance_ids = sorted({os.path.splitext(ann_file)[0] for ann_file in os.listdir(ann_dir)})
        for id in instance_ids:
            ann_file = os.path.join(ann_dir, f'{id}.ann')
            txt_file = ann_file.replace('.ann', '.txt')

            # load text & tokenize w/ Spacy
            with open(txt_file, 'r') as f_txt:
                text = f_txt.read().strip()
                text = ''.join([char if char.strip() != '' else ' ' for char in text])  # normalize whitespaces, such as '\xa0' --> ' '
                spacy_text = nlp(text)

            # no sentence splitting (one long sentence) & tokenize
            # sent_token_spans = TokenSpan.find_sent_token_spans(text=text, sent_tokens=[[token.text for sent in spacy_text.sents for token in sent if token.text.strip() != '']])

            # split sentences & tokenize
            sent_token_spans = TokenSpan.find_sent_token_spans(text=text, sent_tokens=[[token.text for token in sent if token.text.strip() != ''] for sent in spacy_text.sents])

            # load annotated entity mentions
            mention_spans = set()
            with open(ann_file, 'r') as f_ann:
                for line in f_ann:
                    tup = line.strip().split('\t')
                    if tup[0].startswith('T'):
                        entity_id = tup[0]
                        # note: occasionally data looks really stupid like `Task 400 436;437 453` in `S0167931713005042.ann`
                        try:
                            entity_type, start, stop = tup[1].split(' ')
                        except ValueError as e:
                            print(f'Tried unpacking line {tup} in {id}')
                            continue
                        start = int(start)
                        stop = int(stop)
                        entity_text = tup[2]
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

            # write
            print(f'Writing {id}')
            f_out.write(f'-DOCSTART- ({id})\n\n')
            for token_spans, token_labels in zip(sent_token_spans, sent_token_labels):
                for token_span, token_label in zip(token_spans, token_labels):
                    f_out.write('\t'.join([token_span.text, 'NN', 'O', token_label]))
                    f_out.write('\n')  # new token
                f_out.write('\n')  # new sent
            f_out.write('\n')  # new paper

            break
    break




# TODO: stop here; below incomplete relations script





# group TokenSpans into sentences
sents: List[List[TokenSpan]] = []
sent: List[TokenSpan] = []
for i, token_span in enumerate(token_spans):
    if i in index_sent_start_tokens:
        sents.append(sent)
        sent: List[TokenSpan] = []
    sent.append(token_span)

# load annotations
ID_TO_ENTITY = {}
E1E2_TO_RELATION = {}
with open(ann_file, 'r') as f_ann:
    for line in f_ann:
        tup = line.strip().split('\t')

        # check if it's an entity
        if tup[0].startswith('T'):
            entity_id = tup[0]
            entity_type, start, stop = tup[1].split(' ')
            start = int(start)
            stop = int(stop)
            entity_text = tup[2]
            ID_TO_ENTITY[entity_id] = {
                'type': entity_type,
                'span': (start, stop),
                'text': entity_text
            }

        # or if it's an asymmetric relation
        elif tup[0].startswith('R'):
            relation_id = tup[0]
            relation_type, arg1, arg2 = tup[1].split(' ')
            tag1, e1 = arg1.split(':')
            tag2, e2 = arg2.split(':')
            assert tag1 == 'Arg1' and tag2 == 'Arg2'
            E1E2_TO_RELATION[(e1, e2)] =

        # or if it's a symmetric relation
        elif tup[0].startswith('*'):

        # not sure
        else:
            raise ValueError('Cant parse this line {}'.format(line))

        E1E2_TO_RELATION[]



with open(ann_file, 'r') as f_ann:
    pass