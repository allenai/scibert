# <center>SciBERT
<center>A BERT model trained on scientific text. It results into state-of-the-art performance on a wide range of scientific domain nlp tasks.</center>


## Training Corpus
SciBERT is trained on papers from the corpus of [semanticscholar.org]. Corpus size is 1.14M papers, 3.1B tokens. We use the full text of the papers in training, not just abstracts.

## Vocabulary
SciBERT has its own vocabulary that's built to best match the training corpus. We trained a cased and an uncased version. We also include models trained on the original BERT vocabulary for comparison.

## Evaluation
AllenNLP models. Data is in the repo. 

## Downloading Trained Models
We release the tensorflow and the pytorch version of the trained models. The pytorch version is created using the [Hugging Face](https://github.com/huggingface/pytorch-pretrained-BERT) library. 

All other variants (base-vocab or sci-vocab, cased or uncased) are available here:
### Tensorflow Models
* [scibert-scivocab-uncased](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/tensorflow_models/scibert_scivocab_uncased.tar.gz) <<-- Recommended 
* [scibert-scivocab-cased](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/tensorflow_models/scibert_scivocab_cased.tar.gz)
* [scibert-basevocab-uncased](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/tensorflow_models/scibert_basevocab_uncased.tar.gz)
* [scibert-basevocab-cased](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/tensorflow_models/scibert_basevocab_cased.tar.gz)

### Pytorch Models
* [scibert-scivocab-uncased](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/pytorch_models/scibert_scivocab_uncased.tar.gz) <<-- Recommended 
* [scibert-scivocab-cased](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/pytorch_models/scibert_scivocab_cased.tar.gz)
* [scibert-basevocab-uncased](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/pytorch_models/scibert_basevocab_uncased.tar.gz)
* [scibert-basevocab-cased](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/pytorch_models/scibert_basevocab_cased.tar.gz)


## Citing

If you use ScispaCy in your research, please cite .... 

SciBERT is an open-source project developed by [the Allen Institute for Artificial Intelligence (AI2)](http://www.allenai.org).
AI2 is a non-profit institute with the mission to contribute to humanity through high-impact AI research and engineering.




