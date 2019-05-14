# Run allennlp training locally

#
# edit these variables before running script
DATASET='bc5cdr'
TASK='ner'
with_finetuning='_finetune'  # or '' for not fine tuning
dataset_size=4942

export BERT_VOCAB=/net/nfs.corp/s2-research/scibert/scibert_scivocab_cased/vocab.txt
export BERT_WEIGHTS=/net/nfs.corp/s2-research/scibert/scibert_scivocab_cased/weights.tar.gz

export DATASET_SIZE=$dataset_size

CONFIG_FILE=allennlp_config/"$TASK""$with_finetuning".json

SEED=13270
PYTORCH_SEED=`expr $SEED / 10`
NUMPY_SEED=`expr $PYTORCH_SEED / 10`
export SEED=$SEED
export PYTORCH_SEED=$PYTORCH_SEED
export NUMPY_SEED=$NUMPY_SEED

export IS_LOWERCASE=false
export TRAIN_PATH=data/$TASK/$DATASET/train.txt
export DEV_PATH=data/$TASK/$DATASET/dev.txt
export TEST_PATH=data/$TASK/$DATASET/test.txt

export CUDA_DEVICE=0

python -m allennlp.run train $CONFIG_FILE  --include-package scibert -s "$@"