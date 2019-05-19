import os
import csv
import numpy as np
from collections import defaultdict, Counter



def conservative_argmax(hyperparam_metrics):
    # map hyperparams to row/cols
    lr_to_row = {
        '5e-06': 0,
        '1e-05': 1,
        '2e-05': 2,
        '5e-05': 3
    }
    row_to_lr = {row: lr for lr, row in lr_to_row.items()}
    num_epochs_to_col = {
        '2': 0,
        '3': 1,
        '4': 2,
        '5': 3
    }
    col_to_num_epochs = {col: num_epochs for num_epochs, col in num_epochs_to_col.items()}
    # build up matrix
    M = np.zeros((len(lr_to_row), len(num_epochs_to_col)))
    S = np.zeros((len(lr_to_row), len(num_epochs_to_col)))
    for hyperparam, avg_metric, std_metric in hyperparam_metrics:
        lr, num_epochs = hyperparam.replace('(', '').replace(')', '').split(', ')
        M[lr_to_row[lr], num_epochs_to_col[num_epochs]] = avg_metric
        S[lr_to_row[lr], num_epochs_to_col[num_epochs]] = std_metric
    # best in each row
    index_best_col_per_row = []
    for index_row in range(len(M)):
        index_best_col = np.argmax(M[index_row])
        value_to_beat = M[index_row, index_best_col] - S[index_row, index_best_col]
        # pick best one w/ conservative bias
        for index_col in range(M.shape[1]):
            if M[index_row, index_col] >= value_to_beat:
                index_best_col_per_row.append(index_col)
                break
    # best across rows
    index_best_row = np.argmax([M[index_row, index_best_col] for index_row, index_best_col in zip(range(len(M)), index_best_col_per_row)])

    # finally
    best_lr = row_to_lr[index_best_row]
    best_num_epochs = col_to_num_epochs[index_best_col_per_row[index_best_row]]
    return f'({best_lr}, {best_num_epochs})'


# load in raw data and organize
DATASET_TO_MODEL_TO_HYPERPARAM_TO_DEV_METRICS = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
DATASET_TO_MODEL_TO_HYPERPARAM_TO_TEST_METRICS = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
RESULTS_DIR = 'results/'
for results_file in ['ner_lr2e5_epochs245.csv',
                     'ner_lr2e5_epochs3.csv',
                     'ner_lr5e61e55e5_epochs2345.csv',
                     'pico_all_lr_epochs2345_uncased.csv',
                     'cls_lr2e5_epochs3.csv',
                     'cls_lr2e5_epochs3_mag_only.csv',
                     'cls_lr2e5_epochs245.csv',
                     'cls_lr5e61e55e5_epochs2345.csv']:
    with open(os.path.join(RESULTS_DIR, results_file)) as f_in:
        reader = csv.DictReader(f_in)
        for row in reader:
            # skip model we dont care abt
            model = row['env_BERT_WEIGHTS']
            if 'scibert_basevocab' in model:
                continue

            # get dataset & corresp metrics
            dataset = row['env_TRAIN_PATH']
            if '/ner/' in dataset:
                dev_metric = float(row['metric_best_validation_f1-measure-overall'])
                test_metric = float(row['metric_test_f1-measure-overall'])
            elif '/text_classification/' in dataset:
                if 'chemprot' in dataset:
                    dev_metric = float(row['metric_best_validation_accuracy'])
                    test_metric = float(row['metric_test_accuracy'])
                else:
                    dev_metric = float(row['metric_best_validation_average_F1'])
                    test_metric = float(row['metric_test_average_F1'])
            elif '/pico/' in dataset:
                dev_metric = (
                    float(row['metric_best_validation_F1_I-PAR']) +
                    float(row['metric_best_validation_F1_I-INT']) +
                    float(row['metric_best_validation_F1_I-OUT'])
                ) / 3
                test_metric = (
                             float(row['metric_test_F1_I-PAR']) +
                             float(row['metric_test_F1_I-INT']) +
                             float(row['metric_test_F1_I-OUT'])
                         ) / 3

            # get hyperparams
            num_epochs = row['env_NUM_EPOCHS']
            lr = row.get('env_LEARNING_RATE', '2e-05')
            hyperparam = f'({lr}, {num_epochs})'

            DATASET_TO_MODEL_TO_HYPERPARAM_TO_DEV_METRICS[dataset][model][hyperparam].append(dev_metric)
            DATASET_TO_MODEL_TO_HYPERPARAM_TO_TEST_METRICS[dataset][model][hyperparam].append(test_metric)

# compute averages across seeds
DATASET_MODEL_TO_HYPERPARAM_METRICS = defaultdict(list)
for dataset, model_to_hyperparam_to_metrics in DATASET_TO_MODEL_TO_HYPERPARAM_TO_DEV_METRICS.items():
    for model, hyperparam_to_metrics in model_to_hyperparam_to_metrics.items():
        for hyperparam, metrics in hyperparam_to_metrics.items():
            avg_metric = np.mean(metrics) * 100
            std_metric = np.std(metrics) * 100
            DATASET_MODEL_TO_HYPERPARAM_METRICS[(dataset, model)].append((hyperparam, avg_metric, std_metric))

# pick best hyperparams for each (dataset, model)
DATASET_MODEL_TO_BEST_HYPERPARAM = {}
for (dataset, model), hyperparam_metrics in DATASET_MODEL_TO_HYPERPARAM_METRICS.items():

    # TODO:
    # pick the raw argmax
    # index_best_hyperparam = np.argmax(avg_metric for hyperparam, avg_metric, std_metric in hyperparam_metrics)
    # best_hyperparam, _, _ = hyperparam_metrics[index_best_hyperparam]

    # pick the argmax, but favor simplest within epsilon
    best_hyperparam = conservative_argmax(hyperparam_metrics)

    DATASET_MODEL_TO_BEST_HYPERPARAM[(dataset, model)] = best_hyperparam

# lookup (dataset, model) test performance under best hyperparam
BEST_HYPERPARAM_COUNTS = Counter()
for (dataset, model), best_hyperparam in DATASET_MODEL_TO_BEST_HYPERPARAM.items():
    avg_metric = np.mean(DATASET_TO_MODEL_TO_HYPERPARAM_TO_TEST_METRICS[dataset][model][best_hyperparam]) * 100
    std_metric = np.std(DATASET_TO_MODEL_TO_HYPERPARAM_TO_TEST_METRICS[dataset][model][best_hyperparam]) * 100
    dataset = dataset.replace('data/', '').replace('/train.txt', '')
    model = model.replace('/scibert/', '').replace('/weights.tar.gz', '')
    print('\t'.join([dataset, model, str(best_hyperparam), f'{avg_metric:.2f}', f'{std_metric:.2f}']))
    BEST_HYPERPARAM_COUNTS[best_hyperparam] += 1
for tup in BEST_HYPERPARAM_COUNTS.most_common():
    print(tup)


