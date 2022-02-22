[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/scibert-pretrained-contextualized-embeddings/named-entity-recognition-bc5cdr)](https://paperswithcode.com/sota/named-entity-recognition-bc5cdr?p=scibert-pretrained-contextualized-embeddings)  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/scibert-pretrained-contextualized-embeddings/relation-extraction-chemprot)](https://paperswithcode.com/sota/relation-extraction-chemprot?p=scibert-pretrained-contextualized-embeddings)  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/scibert-pretrained-contextualized-embeddings/participant-intervention-comparison-outcome)](https://paperswithcode.com/sota/participant-intervention-comparison-outcome?p=scibert-pretrained-contextualized-embeddings)  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/scibert-pretrained-contextualized-embeddings/named-entity-recognition-ncbi-disease)](https://paperswithcode.com/sota/named-entity-recognition-ncbi-disease?p=scibert-pretrained-contextualized-embeddings)  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/scibert-pretrained-contextualized-embeddings/sentence-classification-paper-field)](https://paperswithcode.com/sota/sentence-classification-paper-field?p=scibert-pretrained-contextualized-embeddings)  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/scibert-pretrained-contextualized-embeddings/citation-intent-classification-scicite)](https://paperswithcode.com/sota/citation-intent-classification-scicite?p=scibert-pretrained-contextualized-embeddings)  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/scibert-pretrained-contextualized-embeddings/sentence-classification-sciencecite)](https://paperswithcode.com/sota/sentence-classification-sciencecite?p=scibert-pretrained-contextualized-embeddings)  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/scibert-pretrained-contextualized-embeddings/relation-extraction-scierc)](https://paperswithcode.com/sota/relation-extraction-scierc?p=scibert-pretrained-contextualized-embeddings)  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/scibert-pretrained-contextualized-embeddings/named-entity-recognition-scierc)](https://paperswithcode.com/sota/named-entity-recognition-scierc?p=scibert-pretrained-contextualized-embeddings)  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/scibert-pretrained-contextualized-embeddings/citation-intent-classification-acl-arc)](https://paperswithcode.com/sota/citation-intent-classification-acl-arc?p=scibert-pretrained-contextualized-embeddings)  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/scibert-pretrained-contextualized-embeddings/sentence-classification-acl-arc)](https://paperswithcode.com/sota/sentence-classification-acl-arc?p=scibert-pretrained-contextualized-embeddings)  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/scibert-pretrained-contextualized-embeddings/dependency-parsing-genia-las)](https://paperswithcode.com/sota/dependency-parsing-genia-las?p=scibert-pretrained-contextualized-embeddings)  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/scibert-pretrained-contextualized-embeddings/dependency-parsing-genia-uas)](https://paperswithcode.com/sota/dependency-parsing-genia-uas?p=scibert-pretrained-contextualized-embeddings)    
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/scibert-pretrained-contextualized-embeddings/named-entity-recognition-jnlpba)](https://paperswithcode.com/sota/named-entity-recognition-jnlpba?p=scibert-pretrained-contextualized-embeddings)   
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/scibert-pretrained-contextualized-embeddings/sentence-classification-pubmed-20k-rct)](https://paperswithcode.com/sota/sentence-classification-pubmed-20k-rct?p=scibert-pretrained-contextualized-embeddings)  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/scibert-pretrained-contextualized-embeddings/sentence-classification-scicite)](https://paperswithcode.com/sota/sentence-classification-scicite?p=scibert-pretrained-contextualized-embeddings)


# <p align=center>`SciBERT`</p>
`SciBERT` is a `BERT` model trained on scientific text.

* `SciBERT` is trained on papers from the corpus of [semanticscholar.org](https://semanticscholar.org). Corpus size is 1.14M papers, 3.1B tokens. We use the full text of the papers in training, not just abstracts.

* `SciBERT` has its own vocabulary (`scivocab`) that's built to best match the training corpus. We trained cased and uncased versions. We also include models trained on the original BERT vocabulary (`basevocab`) for comparison.

* It results in state-of-the-art performance on a wide range of scientific domain nlp tasks. The details of the evaluation are in the [paper](https://arxiv.org/abs/1903.10676). Evaluation code and data are included in this repo. 

### Downloading Trained Models
Update! SciBERT models now installable directly within Huggingface's framework under the `allenai` org:
```
from transformers import *

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
model = AutoModelWithLMHead.from_pretrained('allenai/scibert_scivocab_uncased')

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_cased')
model = AutoModelWithLMHead.from_pretrained('allenai/scibert_scivocab_cased')
```

------

We release the tensorflow and the pytorch version of the trained models. The tensorflow version is compatible with code that works with the model from [Google Research](https://github.com/google-research/bert). The pytorch version is created using the [Hugging Face](https://github.com/huggingface/pytorch-pretrained-BERT) library, and this repo shows how to use it in AllenNLP.  All combinations of `scivocab` and `basevocab`, `cased` and `uncased` models are available below. Our evaluation shows that `scivocab-uncased` usually gives the best results.

#### Tensorflow Models
* __[`scibert-scivocab-uncased`](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/tensorflow_models/scibert_scivocab_uncased.tar.gz) (Recommended)__
* [`scibert-scivocab-cased`](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/tensorflow_models/scibert_scivocab_cased.tar.gz)
* [`scibert-basevocab-uncased`](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/tensorflow_models/scibert_basevocab_uncased.tar.gz)
* [`scibert-basevocab-cased`](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/tensorflow_models/scibert_basevocab_cased.tar.gz)

#### PyTorch AllenNLP Models
* __[`scibert-scivocab-uncased`](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/pytorch_models/scibert_scivocab_uncased.tar) (Recommended)__
* [`scibert-scivocab-cased`](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/pytorch_models/scibert_scivocab_cased.tar)
* [`scibert-basevocab-uncased`](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/pytorch_models/scibert_basevocab_uncased.tar)
* [`scibert-basevocab-cased`](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/pytorch_models/scibert_basevocab_cased.tar)

#### PyTorch HuggingFace Models
* __[`scibert-scivocab-uncased`](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/huggingface_pytorch/scibert_scivocab_uncased.tar) (Recommended)__
* [`scibert-scivocab-cased`](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/huggingface_pytorch/scibert_scivocab_cased.tar)
* [`scibert-basevocab-uncased`](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/huggingface_pytorch/scibert_basevocab_uncased.tar)
* [`scibert-basevocab-cased`](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/huggingface_pytorch/scibert_basevocab_cased.tar)

### Using SciBERT in your own model

SciBERT models include all necessary files to be plugged in your own model and are in same format as BERT.
If you are using Tensorflow, refer to Google's [BERT repo](https://github.com/google-research/bert) and if you use PyTorch, refer to [Hugging Face's repo](https://github.com/huggingface/pytorch-pretrained-BERT) where detailed instructions on using BERT models are provided. 

### Training new models using AllenNLP

To run experiments on different tasks and reproduce our results in the [paper](https://arxiv.org/abs/1903.10676), you need to first setup the Python 3.6 environment:

```pip install -r requirements.txt```

which will install dependencies like [AllenNLP](https://github.com/allenai/allennlp/).

Use the `scibert/scripts/train_allennlp_local.sh` script as an example of how to run an experiment (you'll need to modify paths and variable names like `TASK` and `DATASET`).

We include a broad set of scientific nlp datasets under the `data/` directory across the following tasks. Each task has a sub-directory of available datasets.
```
├── ner
│   ├── JNLPBA
│   ├── NCBI-disease
│   ├── bc5cdr
│   └── sciie
├── parsing
│   └── genia
├── pico
│   └── ebmnlp
└── text_classification
    ├── chemprot
    ├── citation_intent
    ├── mag
    ├── rct-20k
    ├── sci-cite
    └── sciie-relation-extraction
```

For example to run the model on the Named Entity Recognition (`NER`) task and on the `BC5CDR` dataset (BioCreative V CDR), modify the `scibert/train_allennlp_local.sh` script according to:
```
DATASET='bc5cdr'
TASK='ner'
...
```

Decompress the PyTorch model that you downloaded using  
`tar -xvf scibert_scivocab_uncased.tar`  
The results will be in the `scibert_scivocab_uncased` directory containing two files:
A vocabulary file (`vocab.txt`) and a weights file (`weights.tar.gz`).
Copy the files to your desired location and then set correct paths for `BERT_WEIGHTS` and `BERT_VOCAB` in the script:
```
export BERT_VOCAB=path-to/scibert_scivocab_uncased.vocab
export BERT_WEIGHTS=path-to/scibert_scivocab_uncased.tar.gz
```

Finally run the script:

```
./scibert/scripts/train_allennlp_local.sh [serialization-directory]
```

Where `[serialization-directory]` is the path to an output directory where the model files will be stored. 

### Citing

If you use `SciBERT` in your research, please cite [SciBERT: Pretrained Language Model for Scientific Text](https://arxiv.org/abs/1903.10676).
```
@inproceedings{Beltagy2019SciBERT,
  title={SciBERT: Pretrained Language Model for Scientific Text},
  author={Iz Beltagy and Kyle Lo and Arman Cohan},
  year={2019},
  booktitle={EMNLP},
  Eprint={arXiv:1903.10676}
}
```

`SciBERT` is an open-source project developed by [the Allen Institute for Artificial Intelligence (AI2)](http://www.allenai.org).
AI2 is a non-profit institute with the mission to contribute to humanity through high-impact AI research and engineering.




