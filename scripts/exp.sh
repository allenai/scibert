# run allennlp training on beaker

dataset1="ds_5kuso5cektdd:/data/"
dataset2="ds_dpsaxi4ltpw9:/bert_vocab/"
dataset3="ds_q0zib4lz374s:/bert_weights/"
config_file="allennlp_config/ner_bert.jsonnet"

for ner_dataset in bc5cdr  # chemdner msh
do
    for SEED in 13370  # 13570 14680
    do
        for model in s2bert_s2vocab_cased_512_finetune128  # s2bert_s2vocab_uncased_512_finetune128 s2bert_basevocab_cased_512_finetune128 s2bert_basevocab_cased_512 s2bert_basevocab_uncased_128 s2bert_basevocab_uncased_512 bert_base_uncased bert_base_cased
        do

PYTORCH_SEED=`expr $SEED / 10`
NUMPY_SEED=`expr $PYTORCH_SEED / 10`

export SEED=$SEED
export PYTORCH_SEED=$PYTORCH_SEED
export NUMPY_SEED=$NUMPY_SEED


if [[ $model =~ 'uncased' ]];
then
    export is_lowercase=true
    vocab_file="uncased"
else
    export is_lowercase=false
    vocab_file="cased"
fi

if [[ $model =~ 'basevocab' ]];
then
    vocab_file="basevocab_"$vocab_file
else
    vocab_file="s2vocab_"$vocab_file
fi

export BERT_VOCAB=/bert_vocab/"$vocab_file".vocab
export BERT_WEIGHTS=/bert_weights/"$model".tar.gz
export NER_TRAIN_DATA_PATH=/data/ner/$ner_dataset/train.conll2003
export NER_DEV_PATH=/data/ner/$ner_dataset/dev.conll2003
export NER_TEST_PATH=/data/ner/$ner_dataset/test.conll2003


echo "$BERT_VOCAB", "$BERT_WEIGHTS", "$is_lowercase"
# continue  # delete this continue for the experiment to be submitted to beaker
# remember to change the desc below
python scripts/run_with_beaker.py $config_file --source $dataset1 --source $dataset2 --source $dataset3  --desc 's2-bert' \
    --env "BERT_VOCAB=$BERT_VOCAB" --env "BERT_WEIGHTS=$BERT_WEIGHTS" \
    --env "NER_TRAIN_DATA_PATH=$NER_TRAIN_DATA_PATH" --env "NER_DEV_PATH=$NER_DEV_PATH" --env "NER_TEST_PATH=$NER_TEST_PATH" \
    --env "is_lowercase=$is_lowercase" \
    --blueprint bp_1gglr3so9tnr   # this Blueprint has allennlp v0.8
        done
    done
done
