# run allennlp training on beaker

dataset="ds_gluijwejrare:/data/"
config_file="allennlp_config/ner_bert.jsonnet"

for ner_dataset in chemdner # msh # bc5cdr
do
    for SEED in 13370 13570 14680 #  13970 13070 13170 13270 13370 14070 14170 14270 14370 14470 14570 14670 14770 14870 14970 15070 15170 # list more than one seed to run more than one run
    do
        for model in bert_s2_uncased bert_base_uncased bert_base_cased
        do

PYTORCH_SEED=`expr $SEED / 10`
NUMPY_SEED=`expr $PYTORCH_SEED / 10`

export SEED=$SEED
export PYTORCH_SEED=$PYTORCH_SEED
export NUMPY_SEED=$NUMPY_SEED

if [[ $model =~ 'uncased' ]];
then
    export is_lowercase=true
else
    export is_lowercase=false
fi

export BERT_VOCAB=/data/"$model".vocab
export BERT_WEIGHTS=/data/"$model"_weights.tar.gz
export NER_TRAIN_DATA_PATH=/data/ner/$ner_dataset/train.conll2003
export NER_DEV_PATH=/data/ner/$ner_dataset/dev.conll2003
export NER_TEST_PATH=/data/ner/$ner_dataset/test.conll2003


echo "$BERT_VOCAB", "$BERT_WEIGHTS", "$is_lowercase"
# continue  # delete this continue for the experiment to be submitted to beaker
# remember to change the desc below
python scripts/run_with_beaker.py $config_file --source $dataset --desc 's2-bert' \
    --env "BERT_VOCAB=$BERT_VOCAB" --env "BERT_WEIGHTS=$BERT_WEIGHTS" \
    --env "NER_TRAIN_DATA_PATH=$NER_TRAIN_DATA_PATH" --env "NER_DEV_PATH=$NER_DEV_PATH" --env "NER_TEST_PATH=$NER_TEST_PATH" \
    --env "is_lowercase=$is_lowercase" \
    --blueprint bp_1gglr3so9tnr
        done
    done
done
