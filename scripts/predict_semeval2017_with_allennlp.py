"""

Load an Allennlp model and make predictions.  Then format predictions into
Semeval2017 format to use their evaluation script.

"""
import argparse

import os
import subprocess
import json

from collections import defaultdict
import time

from allennlp.common import Params
from allennlp.data import DatasetReader
from allennlp.models import Model


def fetch_beaker_group_experiments_results(results_dir: str, group_id: str):
    group_json = json.loads(subprocess.check_output(['beaker', 'group', 'inspect', '--contents', f'{group_id}'], stderr=subprocess.STDOUT).decode('UTF-8'))
    experiment_ids = group_json[0]['experiments']
    for experiment_id in experiment_ids:
        experiment_dir = os.path.join(results_dir, experiment_id)
        experiment_json = json.loads(subprocess.check_output(['beaker', 'experiment', 'inspect', f'{experiment_id}'], stderr=subprocess.STDOUT).decode('UTF-8'))
        result_id = experiment_json[0]['nodes'][0]['result_id']
        os.makedirs(experiment_dir, exist_ok=True)
        if len(os.listdir(experiment_dir)) == 0:
            subprocess.check_output(['beaker', 'dataset', 'fetch', '--output', f'{experiment_dir}', f'{result_id}'], stderr=subprocess.STDOUT).decode('UTF-8')


def fetch_beaker_experiment_results(results_dir: str, experiment_id: str):
    experiment_dir = os.path.join(results_dir, experiment_id)
    experiment_json = json.loads(subprocess.check_output(['beaker', 'experiment', 'inspect', f'{experiment_id}'], stderr=subprocess.STDOUT).decode('UTF-8'))
    result_id = experiment_json[0]['nodes'][0]['result_id']
    os.makedirs(experiment_dir, exist_ok=True)
    if len(os.listdir(experiment_dir)) == 0:
        subprocess.check_output(['beaker', 'dataset', 'fetch', '--output', f'{experiment_dir}', f'{result_id}'], stderr=subprocess.STDOUT).decode('UTF-8')
    return experiment_dir


def load_bert_reader_model_from_beaker_experiment_dir(experiment_dir: str,
                                                      bert_vocab_dir: str,
                                                      bert_weights_dir: str,
                                                      cuda_device: int = -1):
    # check values of existing config
    config_file = os.path.join(experiment_dir, 'config.json')
    with open(config_file, 'r') as f:
        config_json = json.load(f)
        bert_vocab_name = os.path.basename(config_json['dataset_reader']['token_indexers']['bert']['pretrained_model'])
        bert_weights_name = os.path.basename(config_json['model']['text_field_embedder']['token_embedders']['bert']['pretrained_model'])

    overrides = {
        'dataset_reader.token_indexers.bert.pretrained_model': os.path.join(bert_vocab_dir, bert_vocab_name),
        'model.text_field_embedder.token_embedders.bert.pretrained_model': os.path.join(bert_weights_dir, bert_weights_name)
    }

    # config w/ overrriden new paths
    config = Params.from_file(config_file, params_overrides=json.dumps(overrides))

    # instantiate dataset reader
    reader = DatasetReader.from_params(config["dataset_reader"])

    # instantiate model w/ pretrained weights
    model = Model.load(config.duplicate(),
                       weights_file=os.path.join(experiment_dir, 'best.th'),
                       serialization_dir=experiment_dir,
                       cuda_device=cuda_device)

    return reader, model


def load_spans_for_tokens(conll_file: str):
    all_tokens_spans = []
    with open(conll_file) as f_test:
        token_spans = []
        for line in f_test:
            if 'DOCSTART' in line:
                id = line.strip().split('-DOCSTART- ')[-1][1:-1]
            elif line.strip() == '':
                if len(token_spans) > 0:
                    all_tokens_spans.append(token_spans)
                    token_spans = []
                else:
                    continue
            else:
                token, span, _, label = line.strip().split('\t')
                token_spans.append({
                    'id': id,
                    'token': token,
                    'span': span
                })
    return all_tokens_spans


def extract_spans_with_allennlp(all_tokens_spans, instances, model):
    s = time.time()
    all_extractions = defaultdict(list)
    for instance, token_spans in zip(instances, all_tokens_spans):
        id = token_spans[0]['id']
        print(f'Starting instance {id}')

        # load all the variables and check both lists align
        assert len(instance['metadata'].metadata['words']) == len(token_spans)
        out = model.forward_on_instance(instance)
        pred_tags = out['tags']
        tokens = out['words']
        spans = [tuple(int(i) for i in d['span'].split(',')) for d in token_spans]
        assert [d['token'] for d in token_spans] == tokens
        assert len(pred_tags) == len(tokens) == len(spans)

        # build extractions by converting BIO back to text w/ appropriate whitespace padding between merged tokens
        extractions = []
        for token, pred_tag, span in zip(tokens, pred_tags, spans):
            if pred_tag == 'O':
                continue

            bio, label = pred_tag.split('-')
            if bio == 'B' or bio == 'U':
                extractions.append([{
                    'entity': token,
                    'span': span,
                    'label': label
                }])

            elif bio == 'I' or bio == 'L':
                extractions[-1].append({
                    'entity': token,
                    'span': span,
                    'label': label
                })
        all_extractions[id].extend(extractions)

    e = time.time()
    print(f'Time took for prediction: {e - s}')
    return all_extractions


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--beaker_dir', type=str,
                        help='Location of beaker experiments dump')
    parser.add_argument('--experiment_id', type=str,
                        help='Beaker experiment ID')
    parser.add_argument('--bert_vocab_dir', type=str,
                        help='Location of BERT vocab files')
    parser.add_argument('--bert_weights_dir', type=str,
                        help='Location of BERT weights files')
    parser.add_argument('--conll_file', type=str,
                        help='CONLL2003 file to predict')
    parser.add_argument('--output_dir', type=str,
                        help='Output directory to dump predicted .ann files')
    parser.add_argument('--gpu', action='store_true',
                        help='Flag to use GPU')
    args = parser.parse_args()

    # fetch_beaker_group_experiments_results(results_dir='beaker/', group_id='gr_nfp4yg5x4rsx')
    experiment_dir = fetch_beaker_experiment_results(results_dir=args.beaker_dir,
                                                     experiment_id=args.experiment_id)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    reader, model = load_bert_reader_model_from_beaker_experiment_dir(experiment_dir=experiment_dir,
                                                                      bert_vocab_dir=args.bert_vocab_dir,
                                                                      bert_weights_dir=args.bert_weights_dir,
                                                                      cuda_device=0 if args.gpu else -1)
    # collect token spans from the dataset
    all_tokens_spans = load_spans_for_tokens(args.conll_file)

    # read instances for prediction
    instances = reader.read(args.conll_file)

    # predict while aligning read spans with each token
    all_extractions = extract_spans_with_allennlp(all_tokens_spans, instances, model)

    # each `extraction` is actually a List of tokens & metadata about that token
    # compile them into strings & metadata about that string
    for id, extractions in all_extractions.items():
        compiled_extractions = []
        for extraction in extractions:
            compiled_text = extraction[0]['entity']
            min_start, max_stop = extraction[0]['span']
            label = extraction[0]['label']
            for d in extraction[1:]:
                assert d['label'] == label
                current_start, current_stop = d['span']
                whitespace_between_tokens = ' ' * (current_start - max_stop)
                compiled_text += whitespace_between_tokens + d['entity']
                max_stop = current_stop
            compiled_extraction = {
                'entity': compiled_text,
                'span': (min_start, max_stop),
                'label': label
            }
            compiled_extractions.append(compiled_extraction)
        all_extractions[id] = compiled_extractions

    # write in `.ann` format to match with semeval
    os.makedirs(args.output_dir, exist_ok=True)
    for id, extractions in all_extractions.items():
        with open(os.path.join(args.output_dir, f'{id}.ann'), 'w') as f_out:
            for i, extraction in enumerate(extractions):
                text = extraction['entity']
                start, stop = extraction['span']
                label = extraction['label']
                f_out.write('\t'.join([f'T{i+1}', f'{label} {start} {stop}', f'{text}']))
                f_out.write('\n')
