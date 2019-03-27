# Run allennlp training locally


#
# edit these variables before running script
DATASET='chemprot'
TASK='text_classification'
export BERT_VOCAB=scibert_scivocab_uncased/vocab.txt
export BERT_WEIGHTS=scibert_scivocab_uncased/weights.tar.gz

#
#
CONFIG_FILE=allennlp_config/"$TASK".json

SEED=13270
PYTORCH_SEED=`expr $SEED / 10`
NUMPY_SEED=`expr $PYTORCH_SEED / 10`
export SEED=$SEED
export PYTORCH_SEED=$PYTORCH_SEED
export NUMPY_SEED=$NUMPY_SEED

export IS_LOWERCASE=true
export TRAIN_PATH=data/$TASK/$DATASET/train.txt
export DEV_PATH=data/$TASK/$DATASET/dev.txt
export TEST_PATH=data/$TASK/$DATASET/test.txt

export CUDA_VISIBLE_DEVICES=0

python -m allennlp.run train $CONFIG_FILE  --include-package scibert -s "$@"
