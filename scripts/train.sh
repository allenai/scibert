# Run allennlp training locally

dataset='sciie'
task='ner'
config_file=allennlp_config/"$task".jsonnet

SEED=13270
PYTORCH_SEED=`expr $SEED / 10`
NUMPY_SEED=`expr $PYTORCH_SEED / 10`

export SEED=$SEED
export PYTORCH_SEED=$PYTORCH_SEED
export NUMPY_SEED=$NUMPY_SEED

export BERT_VOCAB=vocab/s2vocab_cased.vocab
export BERT_WEIGHTS=pytorch_models/s2bert_s2vocab_cased_512.tar.gz
export is_lowercase=false
export NER_TRAIN_DATA_PATH=data/$task/$dataset/train.txt
export NER_DEV_PATH=data/$task/$dataset/dev.txt
export NER_TEST_PATH=data/$task/$dataset/test.txt

python -m allennlp.run train $config_file -s "$@"
