#@IgnoreInspection BashAddShebang
# run allennlp training on beaker

bertvocab="ds_dpsaxi4ltpw9:/bert_vocab/"
bertweights="ds_jda1d19zqy6z:/bert_weights/"

for task in text_classification
do
    for dataset in chemprot
    do
        for SEED in 13370 # 13570 14680
        do
            for model in  s2bert_basevocab_cased_512 # bertbase_basevocab_cased biobert_pmc_basevocab_cased biobert_pubmed_pmc_basevocab_cased s2bert_basevocab_uncased_512 s2bert_s2vocab_uncased_512 bertbase_basevocab_uncased biobert_pubmed_basevocab_cased s2bert_basevocab_cased_512 s2bert_s2vocab_cased_512
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


config_file=allennlp_config/"$task".jsonnet

export BERT_VOCAB=/bert_vocab/"$vocab_file".vocab
export BERT_WEIGHTS=/bert_weights/"$model".tar.gz
export TRAIN_PATH=data/$task/$dataset/train.txt
export DEV_PATH=data/$task/$dataset/dev.txt
export TEST_PATH=data/$task/$dataset/test.txt


echo "$BERT_VOCAB", "$BERT_WEIGHTS", "$is_lowercase", "$TRAIN_PATH", "$config_file"
# continue  # delete this continue for the experiment to be submitted to beaker
# remember to change the desc below
python scripts/run_with_beaker.py $config_file --source $bertvocab --source $bertweights \
    --desc 's2-bert' \
    --env "BERT_VOCAB=$BERT_VOCAB" --env "BERT_WEIGHTS=$BERT_WEIGHTS" \
    --env "TRAIN_PATH=$TRAIN_PATH" --env "DEV_PATH=$DEV_PATH" --env "TEST_PATH=$TEST_PATH" \
    --env "is_lowercase=$is_lowercase" \
    --env "SEED=$SEED" --env "PYTORCH_SEED=$PYTORCH_SEED" --env "NUMPY_SEED=$NUMPY_SEED"
            done
        done
    done
done
