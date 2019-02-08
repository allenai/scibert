echo "Input Model: "$1
echo "Model Config: "$2
echo "Output Model: "$3
# pytorch_pretrained_bert convert_tf_checkpoint_to_pytorch bert_models/3B-basevocab_cased_128 bert_config/bertbase_cased.json tmp/pytorch_model.bin
pytorch_pretrained_bert convert_tf_checkpoint_to_pytorch $1 $2 tmp/pytorch_model.bin
cp $2 tmp/bert_config.json
# tar -cvzf pytorch_models/s2bert_basevocab_cased_128.tar.gz tmp/bert_config.json tmp/pytorch_model.bin
cd tmp
tar -cvzf ../$3 bert_config.json pytorch_model.bin
cd ..
