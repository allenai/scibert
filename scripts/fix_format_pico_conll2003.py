"""

The processed PICO files for EBMNLP (see amandalynne's project)
are formatted with 2 columns:  Token & Tag

** CONLL2003 reader requires 4 columns.

Also, the PICO data is separate files for each paper instead of 1 long file.

** Concats those and adds --DOCSTART-- to mark the paaper IDS

Finally, ebmnlp is annotated with I/O tags

** Converts to BIO

Also, PICO data is not sentence split, but BERT truncates if sequence is
longer than 250 tokens.  This causes bugs.

** Don't do anything fancy, just split on every '.' that isn't part of a PICO element.

"""

import os

old_dirname = 'ebmnlp_amandalynne/'
new_dirname = 'data/ner/ebmnlp/'
for split in ['train', 'dev', 'test']:
    with open(os.path.join(new_dirname, f'{split}.txt'), 'w') as f_new:
        filenames = os.listdir(os.path.join(old_dirname, split))
        for filename in filenames:
            paper_id = os.path.splitext(filename)[0]
            with open(os.path.join(old_dirname, split, filename), 'r') as f_old:
                f_new.write(f'--DOCSTART-- ({paper_id})')
                f_new.write('\n\n')
                prev_label = 'O'
                for line in f_old:
                    token, label = line.strip().split(' ')
                    if prev_label == 'O' and label.startswith('I'):
                        label = 'B-' + label[2:]
                    f_new.write('\t'.join([token, 'NN', 'O', label]))
                    f_new.write('\n')
                    # add extra whitespace to split sentences (hacky sort of)
                    if token == '.' and label == 'O':
                        f_new.write('\n')
                    prev_label = label
                f_new.write('\n\n')
