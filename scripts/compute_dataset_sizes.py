

from collections import Counter

N = Counter()

# count tokens for ner
for dataset in ['bc5cdr', 'JNLPBA', 'NCBI-disease', 'sciie']:
    for split in ['train', 'dev', 'test']:
        with open(f'data/ner/{dataset}/{split}.txt') as f_in:
            for line in f_in:
                if line.strip() == '':
                    N[dataset] += 1

# count tokens for PICO
for split in ['train', 'dev', 'test']:
    with open(f'data/pico/ebmnlp/{split}.txt') as f_in:
        for line in f_in:
            for line in f_in:
                if line.strip() == '':
                    N['ebmnlp'] += 1

# count num sentences for parsing
for split in ['train', 'dev', 'test']:
    with open(f'data/parsing/genia/{split}.txt') as f_in:
        for line in f_in:
            if line.strip() == '':
                N['genia'] += 1

# count num JSONs for CLS
for dataset in ['chemprot', 'citation_intent', 'pico', 'rct-20k', 'sci-cite', 'sciie-relation-extraction']:
    for split in ['train', 'dev', 'test']:
        with open(f'data/text_classification/{dataset}/{split}.txt') as f_in:
            for line in f_in:
                N[dataset] += 1

import json
print(json.dumps(N, indent=4))