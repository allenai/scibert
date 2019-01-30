for vocab in s2vocab # basevocab # s2vocab
do
    for case_status in cased uncased # cased
    do
        for max_len in 128 # 512 # 128 512
        do
            for ((i=0;i<=0;i++)); 
            do 


if [[ "$max_len" == 512 ]]; then
    max_pred=75
else
    max_pred=20
fi

if [[ $case_status == 'uncased' ]];
then
    export do_lower_case=True
else
    export do_lower_case=False
fi

command="--output_file=tfRecords/tfRecords_"$vocab"_"$case_status"_"$max_len"/$i.tfrecord --vocab_file=vocab/"$vocab"_"$case_status".vocab --do_lower_case="$do_lower_case" --max_seq_length="$max_len" --max_predictions_per_seq="$max_pred""
echo $command

sem -j 55 --no-notice  "python ../bert/create_pretraining_data.py --input_file=data/out_without_DONE/$i.out $command --masked_lm_prob=0.15 --random_seed=12345 --dupe_factor=5"

            done
        done
    done
done
