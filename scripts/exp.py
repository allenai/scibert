"""

Alternative to `exp.sh` script because it doesn't seem to run properly on UNIX machines

"""

import subprocess


if __name__ == '__main__':

    dataset_sizes = {
                    'NCBI-disease': 5424,
                    'bc5cdr': 4942,
                    'JNLPBA': 18607,
                    'sciie': 2211,
                    'chemprot': 4169,
                    'citation_intent': 1688,
                    'mag': 84000,
                    'rct-20k': 180040,
                    'sciie-relation-extraction': 3219,
                    'sci-cite': 7320,
                    'ebmnlp': 38124,
                    'genia': 14326,
                    }

    for dataset in [
                    'NCBI-disease',
                    'bc5cdr',
                    'JNLPBA',
                    'sciie',
                    'chemprot',
                    'citation_intent',
                    # 'mag',
                    # 'rct-20k',
                    'sciie-relation-extraction',
                    'sci-cite',
                    # 'ebmnlp',
                    # 'genia',
                ]:
        for seed in [
                    15370,
                    15570,
                    15680,
                    # 15780,
                    # 15210,
                    # 16210,
                    # 16310,
                    # 16410,
                    # 18210,
                    # 18310,
                    # 18410,
                    # 18510,
                    # 18610
                    ]:

            pytorch_seed = seed // 10
            numpy_seed = pytorch_seed // 10

            for model in [
                        None
                        # 'bertbase_basevocab_uncased',
                        # 'bertbase_basevocab_cased',
                        # 'biobert_pmc_basevocab_cased',
                        # 'biobert_pubmed_pmc_basevocab_cased',
                        # 'biobert_pubmed_basevocab_cased',
                        # 'scibert_basevocab_uncased',
                        # 'scibert_basevocab_cased',
                        # 'scibert_scivocab_uncased',
                        # 'scibert_scivocab_cased',
                        ]:

                for with_finetuning in [
                                        '_finetune',
                                        # ''
                                        ]:

                    for grad_accum_batch_size in [
                                                  32
                                                 ]:

                        for num_epochs in [
                                            # 75      # no-finetuning
                                            1,
                                            2,
                                            3,
                                            # 4,
                                            # 5
                                            ]:

                            for learning_rate in [
                                                    # 0.001,  # no-finetuning
                                                    # 5e-6,
                                                    # 1e-5,
                                                    2e-5,
                                                    3e-5,
                                                    5e-5
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

                                dataset_size = dataset_sizes[dataset]
                                # determine casing from model name
                                # if 'uncased' in model:
                                #     is_lowercase = 'true'
                                # else:
                                #     is_lowercase = 'false'
                                is_lowercase = 'false'

                                # config file
                                config_file = f'allennlp_config/{task}{with_finetuning}.json'

                                # bert files
                                model_id = '7'
                                bert_vocab = f'/bigscibert/vocab.txt'
                                bert_weights = f'/bigscibert/{model_id}.tar.gz'

                                # data files
                                train_path = f'data/{task}/{dataset}/train.txt'
                                dev_path = f'data/{task}/{dataset}/dev.txt'
                                test_path = f'data/{task}/{dataset}/test.txt'

                                cmd = ' '.join(['python', 'scripts/run_with_beaker.py',
                                    f'{config_file}',
                                    f'--name {model_id}_{dataset}_{seed}_{num_epochs}_{learning_rate}',
                                    '--source ds_ennmwk8zygn8:/bigscibert/',
                                    '--include-package scibert',
                                    '--env CUDA_DEVICE=0',
                                    f'--env DATASET_SIZE={dataset_size}',
                                    f'--env BERT_VOCAB={bert_vocab}',
                                    f'--env BERT_WEIGHTS={bert_weights}',
                                    f'--env TRAIN_PATH={train_path}',
                                    f'--env DEV_PATH={dev_path}',
                                    f'--env TEST_PATH={test_path}',
                                    f'--env IS_LOWERCASE={is_lowercase}',
                                    f'--env SEED={seed}',
                                    f'--env PYTORCH_SEED={pytorch_seed}',
                                    f'--env NUMPY_SEED={numpy_seed}',
                                    f'--env GRAD_ACCUM_BATCH_SIZE={grad_accum_batch_size}',
                                    f'--env NUM_EPOCHS={num_epochs}',
                                    f'--env LEARNING_RATE={learning_rate}'
                                ])
                                print('\n')
                                print(cmd)

                                completed = subprocess.run(cmd, shell=True)
                                print(f'returncode: {completed.returncode}')
