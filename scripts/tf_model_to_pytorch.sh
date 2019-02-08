# Script to convert a Tensorflow BERT model into a pytorch BERT model using the `pytorch_pretrained_bert` library

echo "Input Model: "$1
echo "Model Config: "$2
echo "Output Model: "$3
pytorch_pretrained_bert convert_tf_checkpoint_to_pytorch $1 $2 tmp/pytorch_model.bin
cp $2 tmp/bert_config.json
cd tmp
tar -cvzf ../$3 bert_config.json pytorch_model.bin
cd ..
