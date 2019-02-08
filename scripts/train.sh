# Run allennlp training locally
config_file="allennlp_config/ner_bert.jsonnet"

SEED=13270
PYTORCH_SEED=`expr $SEED / 10`
NUMPY_SEED=`expr $PYTORCH_SEED / 10`

dataset='bc5cdr'
export SEED=$SEED
export PYTORCH_SEED=$PYTORCH_SEED
export NUMPY_SEED=$NUMPY_SEED

export BERT_VOCAB=vocab/s2vocab_cased.vocab
export BERT_WEIGHTS=pytorch_models/s2bert_s2vocab_cased_512_finetune128.tar.gz
export is_lowercase=false
export NER_TRAIN_DATA_PATH=data/ner/$dataset/train.conll2003
export NER_DEV_PATH=data/ner/$dataset/dev.conll2003
export NER_TEST_PATH=data/ner/$dataset/test.conll2003

python -m allennlp.run  train $config_file  -s $1 $2
