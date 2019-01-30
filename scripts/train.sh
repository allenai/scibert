# Run allennlp training locally
config_file="allennlp_config/ner_bert.jsonnet"

SEED=13270
PYTORCH_SEED=`expr $SEED / 10`
NUMPY_SEED=`expr $PYTORCH_SEED / 10`

dataset='chemdner' # 'msh'
export SEED=$SEED
export PYTORCH_SEED=$PYTORCH_SEED
export NUMPY_SEED=$NUMPY_SEED

export BERT_VOCAB=data/bert_s2_uncased.vocab
export BERT_WEIGHTS=data/bert_s2_uncased_weights.tar.gz
export is_lowercase=true
export NER_TRAIN_DATA_PATH=data/ner/$dataset/train.conll2003
export NER_DEV_PATH=data/ner/$dataset/dev.conll2003
export NER_TEST_PATH=data/ner/$dataset/test.conll2003


python -m allennlp.run  train $config_file  -s $1 $2
