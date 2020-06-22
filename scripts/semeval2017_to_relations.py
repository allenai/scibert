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

Output 1:  Relation classification
    JSONlines where each JSON is:
    {
        "text": "Based on the theoretical analysis , the [[ value of the measuring resistor ]] ...",
        "label": "Synonym-of",
        "metadata": [7, 11, 12, 12]
    }
    ** only contains sentences with relations; entities are gold wrt their Arg1/Arg2 position; no cross-sentence relations **

Output 2:  Relation extraction
    ** same format as above, but includes sentences without relations & all entity pairs are attempted as Arg1/Arg2 positions **

"""

from typing import *

import os

import json

import spacy
nlp = spacy.load('en_core_web_md')

from collections import defaultdict

from sci_bert.common.span import Span, TokenSpan, MentionSpan, _match_mention_spans_to_sentences
from sci_bert.common.relation import RelationMention


def load_annotations(ann_file: str) -> Tuple[DefaultDict, Set]:
    mention_spans = defaultdict(MentionSpan)
    relation_mentions = set()
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
                # note: occasionally, spans include whitespace like in S0370269304008998 with mention `'transition form factors defined through the matrix elements of the operator\xa0O7, φB'`
                # this makes it hard to check that `text[start:stop]` == mention_text
                # >> correct these situations
                entity_text = ''.join([char if char.strip() != '' else ' ' for char in entity_text])  # normalize whitespaces, such as '\xa0' --> ' '

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
                mention_span = MentionSpan(start=start, stop=stop,
                                           text=entity_text,
                                           entity_types=[entity_type],
                                           entity_id=entity_id)
                mention_spans[entity_id] = mention_span

            # load asymmetric relation
            elif tup[0].startswith('R'):
                relation_id = tup[0]
                relation_type, arg1, arg2 = tup[1].split(' ')
                tag1, e1 = arg1.split(':')
                tag2, e2 = arg2.split(':')
                assert tag1 == 'Arg1' and tag2 == 'Arg2'
                relation_mention = RelationMention(e1=mention_spans[e1],
                                                   e2=mention_spans[e2],
                                                   labels=[relation_type],
                                                   is_symmetric=False)
                relation_mentions.add(relation_mention)

            # load symmetric relation
            elif tup[0].startswith('*'):
                relation_id = tup[0]
                # handle odd cases like in `S0021999114008523` with 3 entities: `*	Synonym-of T11 T12 T10`
                try:
                    relation_type, e1, e2 = tup[1].split(' ')
                except ValueError as e:
                    print(f'Failed unpacking line {tup} in {id}. Skipping...')
                    continue
                relation_mention = RelationMention(e1=mention_spans[e1],
                                                   e2=mention_spans[e2],
                                                   labels=[relation_type],
                                                   is_symmetric=True)
                relation_mentions.add(relation_mention)
    return mention_spans, relation_mentions


def tag_mentions_in_sentence(text: str,
                             sent_start: int,
                             sent_stop: int,
                             e1: MentionSpan,
                             e2: MentionSpan) -> str:
    sent_text = text[sent_start:sent_stop]
    if e1 < e2:
        e1_start = e1.start - sent_start
        e1_stop = e1.stop - sent_start
        assert sent_text[e1_start:e1_stop] == relation.e1.text
        sent_text = sent_text[:e1_start] + '<E1>' + sent_text[e1_start:e1_stop] + '</E1>' + sent_text[e1_stop:]
        e2_start = relation.e2.start - sent_start + 9
        e2_stop = relation.e2.stop - sent_start + 9
        assert sent_text[e2_start:e2_stop] == relation.e2.text
        sent_text = sent_text[:e2_start] + '<E2>' + sent_text[e2_start:e2_stop] + '</E2>' + sent_text[e2_stop:]
        return sent_text

    elif e2 < e1:
        e2_start = e2.start - sent_start
        e2_stop = e2.stop - sent_start
        assert sent_text[e2_start:e2_stop] == relation.e2.text
        sent_text = sent_text[:e2_start] + '<E2>' + sent_text[e2_start:e2_stop] + '</E2>' + sent_text[e2_stop:]
        e1_start = relation.e1.start - sent_start + 9
        e1_stop = relation.e1.stop - sent_start + 9
        assert sent_text[e1_start:e1_stop] == relation.e1.text
        sent_text = sent_text[:e1_start] + '<E1>' + sent_text[e1_start:e1_stop] + '</E1>' + sent_text[e1_stop:]
        return sent_text

    else:
        raise Exception('GAHH!! Nested mentions!!')


RE = {'train': {}, 'dev': {}, 'test': {}}
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
        mention_spans, relation_mentions = load_annotations(ann_file)

        # map entity mentions to sentences
        entity_to_sentence = {}
        sent_mention_spans = _match_mention_spans_to_sentences(sent_token_spans, sorted(mention_spans.values(), key=lambda s: (s.start, s.stop)))
        for i, mention_spans in enumerate(sent_mention_spans):
            for mention_span in mention_spans:
                entity_to_sentence[mention_span.entity_id] = i

        # remove cross sentence relations
        sent_id_to_relations = defaultdict(set)
        for relation_mention in relation_mentions:
            sent_id_e1 = entity_to_sentence.get(relation_mention.e1.entity_id)
            sent_id_e2 = entity_to_sentence.get(relation_mention.e2.entity_id)
            if sent_id_e1 is None:
                print(f'Mention {relation_mention.e1} was removed so skipping relation...')
                continue
            if sent_id_e2 is None:
                print(f'Mention {relation_mention.e2} was removed so skipping relation...')
                continue
            if sent_id_e1 == sent_id_e2:
                sent_id_to_relations[sent_id_e1].add(relation_mention)

        # for each sentence containing relation(s), save the sentence after
        # tagging the entities with << >> and [[ ]]
        for sent_id, relations in sent_id_to_relations.items():
            sent_start = sent_token_spans[sent_id][0].start
            sent_stop = sent_token_spans[sent_id][-1].stop
            for relation in relations:
                e1 = relation.e1
                e2 = relation.e2
                sent_text = tag_mentions_in_sentence(text, sent_start, sent_stop, e1, e2)
                RE[split][id] = {
                    'text': sent_text,
                    'label': relation.labels[0],
                    'metadata': {
                        'id': id,
                        'spans': [e1.start, e1.stop, e2.start, e2.stop]
                    }
                }

for split in ['train', 'dev', 'test']:
    num_papers = len(RE[split])
    print(f'Finished processing {num_papers} papers from {split}')

# write NER
os.makedirs('data/rel/semeval2017/', exist_ok=True)
for split in ['train', 'dev', 'test']:
    with open(f'data/rel/semeval2017/{split}.txt', 'w') as f_out:
        for id, instance in RE[split].items():
            json.dump(instance, f_out)
            f_out.write('\n')