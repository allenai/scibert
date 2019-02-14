{
  "random_seed": std.parseInt(std.extVar("SEED")),
  "pytorch_seed": std.parseInt(std.extVar("PYTORCH_SEED")),
  "numpy_seed": std.parseInt(std.extVar("NUMPY_SEED")),
  "dataset_reader": {
    "type": "classification_dataset_reader",
  },
  "train_data_path": std.extVar("CITATION_TRAIN_PATH"),
  "validation_data_path": std.extVar("CITATION_DEV_PATH"),
  "test_data_path": std.extVar("CITATION_TEST_PATH"),
  "evaluate_on_test": true,
  "model": {
    "type": "text_classifier",
    "verbose_metrics": false,
    "text_field_embedder": {
      "tokens": {
        "type": "embedding",
        "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz",
        "embedding_dim": 100,
        "trainable": false
      }
    },
      "text_encoder": {
      "type": "lstm",
      "bidirectional": true,
      "input_size": 100,
      "hidden_size": 50,
      "num_layers": 2,
      "dropout": 0.4
    },
    "classifier_feedforward": {
      "input_dim": 100,
      "num_layers": 2,
      "hidden_dims": [50, 6],
      "activations": ["relu", "linear"],
      "dropout": [0.25, 0.0]
    }
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["text", "num_tokens"]],
    "batch_size": 8
  },

  "trainer": {
    "num_epochs": 30,
    "grad_clipping": 5.0,
    "patience": 10,
    "validation_metric": "+average_F1", //"-loss",
    "cuda_device": 0,
    "optimizer": {
        "type": "adam",
        "lr": 0.001
    },
  }
}
