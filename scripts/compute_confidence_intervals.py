"""

Computes bootstrap confidence intervals

"""

from collections import defaultdict

DATASET_TO_RESULTS = defaultdict(lambda: defaultdict(list))
with open('misc/results.tsv', 'r') as f_in:
    for line in f_in:
        try:
            score, model, dataset = line.strip().split('\t')
            dataset = dataset.replace('data/', '').replace('/dev.txt', '')
            model = model.replace('/bert_weights/', '').replace('_512', '').replace('.tar.gz', '')

            # skip biobert
            if 'biobert' in model:
                continue

            # skip semeval
            if 'semeval' in dataset:
                continue

            score = float(score)
            DATASET_TO_RESULTS[dataset][model].append(score)
        except ValueError:
            continue

import numpy as np

def ci(x):
    np.random.seed(100)
    x = [np.mean(np.random.choice(x, size=5, replace=True)) for _ in range(10000)]
    return np.percentile(x, q=[2.5, 97.5]) * 100

def diff_ci(baseline_scores, new_scores):
    np.random.seed(100)
    baseline_scores = np.array([np.mean(np.random.choice(baseline_scores, size=5, replace=True)) for _ in range(10000)])
    new_scores = np.array([np.mean(np.random.choice(new_scores, size=5, replace=True)) for _ in range(10000)])
    z = new_scores - baseline_scores
    return np.percentile(z, q=[2.5, 97.5]) * 100

def paired_ci(baseline_scores, new_scores):
    assert len(baseline_scores) == len(new_scores)
    np.random.seed(100)
    diffs = np.array(new_scores) - np.array(baseline_scores)
    new_diffs = np.array([np.mean(np.random.choice(diffs, size=5, replace=True)) for _ in range(10000)])
    return np.percentile(new_diffs, q=[2.5, 97.5]) * 100


def compute_table_1(is_paired_ci: bool = False):
    if is_paired_ci:
        get_ci = paired_ci
    else:
        get_ci = diff_ci

    for dataset, model_to_results in DATASET_TO_RESULTS.items():

        # filter models by casing
        MODEL_TO_RESULTS = {}
        for model, results in model_to_results.items():
            is_cased = True if 'uncased' not in model else False
            # cased models for Seq tagging
            if ('ner' in dataset or 'pico' in dataset or 'parsing' in dataset) and is_cased is False:
                continue
            # uncased models for Text class
            if 'text_classification' in dataset and is_cased is True:
                continue
            MODEL_TO_RESULTS[model] = results

        # print(f'*** Dataset: {dataset} ***')
        row = []
        # get bert base results
        for model, results in MODEL_TO_RESULTS.items():
            if 'bertbase' in model:
                bertbase_results = results
                # print(f'BERT-Base: {np.round(np.mean(bertbase_results)} * 100, 2)}')
                # print(f'BERT-Base: {ci(bertbase_results)}')
                row.append(f'{np.round(np.mean(bertbase_results) * 100, 2)}')
                break
        assert bertbase_results

        # get scibert basevocab results
        for model, results in MODEL_TO_RESULTS.items():
            if 's2bert_basevocab' in model:
                scibert_basevocab_results = results
                # print(f'SciBERT-BaseVocab: {np.round(np.mean(scibert_basevocab_results) * 100, 2)}')
                # print(f'SciBERT-BaseVocab: {ci(scibert_basevocab_results)}')
                row.append(f'{np.round(np.mean(scibert_basevocab_results) * 100, 2)}')
                break
        assert scibert_basevocab_results

        for model, results in MODEL_TO_RESULTS.items():
            if 's2bert_s2vocab' in model:
                scivocab_results = results
                # print(f'SciBERT-SciVocab: {np.round(np.mean(scivocab_results) * 100, 2)}')
                # print(f'SciBERT-SciVocab: {ci(scivocab_results)}')
                row.append(f'{np.round(np.mean(scivocab_results) * 100, 2)}')
                break
        assert scivocab_results

        print(' & '.join([dataset] + row))

        # compute diff CIs
        # print(f'SciBERT (BaseVocab) - BERT-Base: {get_ci(bertbase_results, scibert_basevocab_results)}')
        # print(f'SciBERT (SciVocab) - BERT-Base: {get_ci(bertbase_results, scivocab_results)}')
        # print(f'SciBERT:  SciVocab - BaseVocab: {get_ci(scibert_basevocab_results, scivocab_results)}')

        # print('--------')


def compute_full_table():
    for dataset, MODEL_TO_RESULTS in DATASET_TO_RESULTS.items():
        print(f'*** Dataset: {dataset} ***')
        # get bert base cased results
        for model, results in MODEL_TO_RESULTS.items():
            if 'bertbase' in model and 'uncased' not in model:
                bertbase_cased_results = results
                break
        assert bertbase_cased_results

        # get bert base uncased results
        for model, results in MODEL_TO_RESULTS.items():
            if 'bertbase' in model and 'uncased' in model:
                bertbase_uncased_results = results
                break
        assert bertbase_uncased_results

        # get scibert basevocab results (cased)
        for model, results in MODEL_TO_RESULTS.items():
            if 's2bert_basevocab' in model and 'uncased' not in model:
                scibert_basevocab_cased_results = results
                break
        assert scibert_basevocab_cased_results

        # get scibert basevocab results (uncased)
        for model, results in MODEL_TO_RESULTS.items():
            if 's2bert_basevocab' in model and 'uncased' in model:
                scibert_basevocab_uncased_results = results
                break
        assert scibert_basevocab_uncased_results

        # get scibert scivocab results (cased)
        for model, results in MODEL_TO_RESULTS.items():
            if 's2bert_s2vocab' in model and 'uncased' not in model:
                scivocab_cased_results = results
                break
        assert scivocab_cased_results

        # get scibert scivocab results (uncased)
        for model, results in MODEL_TO_RESULTS.items():
            if 's2bert_s2vocab' in model and 'uncased' in model:
                scivocab_uncased_results = results
                break
        assert scivocab_uncased_results

        # compute diff CIs
        print(f'SciBERT (BaseVocab) - BERT-Base  [CASED]: {diff_ci(bertbase_cased_results, scibert_basevocab_cased_results)}')
        print(f'SciBERT (SciVocab) - BERT-Base  [CASED]: {diff_ci(bertbase_cased_results, scivocab_cased_results)}')
        print(f'SciBERT:  SciVocab - BaseVocab  [CASED]: {diff_ci(scibert_basevocab_cased_results, scivocab_cased_results)}')

        print(f'SciBERT (BaseVocab) - BERT-Base  [UNCASED]: {diff_ci(bertbase_uncased_results, scibert_basevocab_uncased_results)}')
        print(f'SciBERT (SciVocab) - BERT-Base  [UNCASED]: {diff_ci(bertbase_uncased_results, scivocab_uncased_results)}')
        print(f'SciBERT:  SciVocab - BaseVocab  [UNCASED]: {diff_ci(scibert_basevocab_uncased_results, scivocab_uncased_results)}')

        print('--------')


