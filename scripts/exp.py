"""

Alternative to `exp.sh` script because it doesn't seem to run properly on UNIX machines

"""

import subprocess

for dataset in [
                # 'NCBI-disease',
                # 'bc5cdr',
                # 'JNLPBA',
                # 'sciie',
                # 'chemprot',
                # 'citation_intent',
                # 'mag',
                # 'rct-20k',
                # 'sciie-relation-extraction',
                # 'sci-cite',
                # 'ebmnlp',
                'genia',
               ]:
    for seed in [
                    # 15370,
                    # 15570,
                    # 15680,
                    # 15780,
                    # 15210,
                    # 16210,
                    # 16310,
                    # 16410,
                    # 18210,
                    # 18310,
                    18410,
                    18510,
                    18610,
                ]:

        pytorch_seed = seed // 10
        numpy_seed = pytorch_seed // 10

        for model in ['bertbase_basevocab_cased',
                      'bertbase_basevocab_uncased',
                      'biobert_pmc_basevocab_cased',
                      'biobert_pubmed_pmc_basevocab_cased',
                      'biobert_pubmed_basevocab_cased',
                      's2bert_basevocab_cased_512',
                      's2bert_basevocab_uncased_512',
                      's2bert_s2vocab_cased_512',
                      's2bert_s2vocab_uncased_512',
                     ]:

            if dataset in ['NCBI-disease', 'bc5cdr', 'JNLPBA', 'sciie']:
                task = 'ner'
            elif dataset in ['chemprot', 'citation_intent', 'mag', 'rct-20k', 'sciie-relation-extraction', 'sci-cite']:
                task = 'text_classification'
            elif dataset in ['ebmnlp']:
                task = 'pico'
            elif dataset in ['genia']:
                task = 'parsing'
            else:
                assert False

            # determine casing based on model
            if 'uncased' in model:
                is_lowercase = 'true'
                vocab_file = 'uncased'
            else:
                is_lowercase = 'false'
                vocab_file = 'cased'

            # determine vocab file based on model
            if 'basevocab' in model:
                vocab_file = 'basevocab_' + vocab_file
            else:
                vocab_file = 's2vocab_' + vocab_file

            # config file
            config_file = f'allennlp_config/{task}.json'

            # bert files
            bert_vocab = f'/bert_vocab/{vocab_file}.vocab'
            bert_weights = f'/bert_weights/{model}.tar.gz'

            # data files
            train_path = f'data/{task}/{dataset}/train.txt'
            dev_path = f'data/{task}/{dataset}/dev.txt'
            test_path = f'data/{task}/{dataset}/test.txt'

            # data files
            train_path = f'data/{task}/{dataset}/train.txt'
            dev_path = f'data/{task}/{dataset}/dev.txt'
            test_path = f'data/{task}/{dataset}/test.txt'
            print(task, dataset, seed, bert_weights, bert_vocab, train_path)
            cmd = ' '.join(['python', 'scripts/run_with_beaker.py',
                   f'{config_file}',
                   '--source ds_dpsaxi4ltpw9:/bert_vocab/',
                   '--source ds_jda1d19zqy6z:/bert_weights/',
                   f'--env BERT_VOCAB={bert_vocab}',
                   f'--env BERT_WEIGHTS={bert_weights}',
                   f'--env TRAIN_PATH={train_path}',
                   f'--env DEV_PATH={dev_path}',
                   f'--env TEST_PATH={test_path}',
                   f'--env is_lowercase={is_lowercase}',
                   f'--env SEED={seed}',
                   f'--env PYTORCH_SEED={pytorch_seed}',
                   f'--env NUMPY_SEED={numpy_seed}'])
            completed = subprocess.run(cmd, shell=True)
            print(f'returncode: {completed.returncode}')
