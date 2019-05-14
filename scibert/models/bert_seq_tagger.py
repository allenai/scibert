from typing import Dict, Optional, List, Any

import numpy
from overrides import overrides
import torch
from torch.nn.modules.linear import Linear
import torch.nn.functional as F

from allennlp.data import Vocabulary
from allennlp.modules import TimeDistributed, TextFieldEmbedder, ConditionalRandomField
from allennlp.modules.conditional_random_field import allowed_transitions
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy, SpanBasedF1Measure, F1Measure


@Model.register("bert_seq_tagger")
class BertSeqTagger(Model):
    """
    Implements a basic sequence tagger:
    1) Embed tokens using `text_field_embedder`
    2) Feedforward layer on top of each token
    3) CRF decoding

    Optimized with CrossEntropyLoss.
    Evaluated with CategoricalAccuracy, Span-level, and Token-level F1.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 dropout: float = 0.2,
                 use_crf: bool = True,
                 calculate_span_f1: bool = True,
                 label_encoding: str = 'BIO',
                 label_namespace: str = "labels",
                 verbose_metrics: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        # about labels
        self.label_namespace = label_namespace
        self.num_classes = self.vocab.get_vocab_size(label_namespace)
        labels = self.vocab.get_index_to_token_vocabulary(label_namespace)

        # model components
        self.text_field_embedder = text_field_embedder
        self.dropout = torch.nn.Dropout(dropout)
        self.tag_projection_layer = TimeDistributed(Linear(self.text_field_embedder.get_output_dim(), self.num_classes))
        if use_crf:
            constraints = allowed_transitions(label_encoding, labels)
            self.crf = ConditionalRandomField(self.num_tags, constraints=constraints, include_start_end_transitions=True)
        else:
            self.crf = None

        # metrics
        self._verbose_metrics = verbose_metrics
        self.metrics = { 'accuracy': CategoricalAccuracy() }
        for index, label in self.vocab.get_index_to_token_vocabulary(self.label_namespace).items():
            self.metrics[label] = F1Measure(positive_label=index)
        # span-level F1 is broken unless some form of constrained decoding or use IO tags
        if calculate_span_f1 and use_crf:
            self._f1_metric = SpanBasedF1Measure(vocab, tag_namespace=label_namespace, label_encoding=label_encoding)
        else:
            self._f1_metric = None

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                tags: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[
        str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : Dict[str, torch.LongTensor], required
            The output of ``TextField.as_array()``, which should typically be passed directly to a
            ``TextFieldEmbedder``. This output is a dictionary mapping keys to ``TokenIndexer``
            tensors.  At its most basic, using a ``SingleIdTokenIndexer`` this is: ``{"tokens":
            Tensor(batch_size, num_tokens)}``. This dictionary will have the same keys as were used
            for the ``TokenIndexers`` when you created the ``TextField`` representing your
            sequence.  The dictionary is designed to be passed directly to a ``TextFieldEmbedder``,
            which knows how to combine different word representations into a single vector per
            token in your input.
        tags : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class labels of shape
            ``(batch_size, num_tokens)``.
        metadata : ``List[Dict[str, Any]]``, optional, (default = None)
            metadata containing the original words in the sentence to be tagged under a 'words' key.
        Returns
        -------
        An output dictionary consisting of:
        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            unnormalised log probabilities of the tag classes.
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            a distribution of the tag classes per word.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        embedded_text_input = self.text_field_embedder(tokens)
        batch_size, sequence_length, _ = embedded_text_input.size()
        mask = get_text_field_mask(tokens)
        logits = self.tag_projection_layer(self.dropout(embedded_text_input))

        # output
        output_dict = {}

        if self.crf is not None:
            best_paths = self.crf.viterbi_tags(logits, mask)
            predicted_tags = [x for x, y in best_paths]
            output_dict['tags'] = predicted_tags
            class_probabilities = logits * 0.
            for i, instance_tags in enumerate(predicted_tags):
                for j, tag_id in enumerate(instance_tags):
                    class_probabilities[i, j, tag_id] = 1
        else:
            reshaped_log_probs = logits.view(-1, self.num_classes)
            class_probabilities = F.softmax(reshaped_log_probs, dim=-1).view([batch_size, sequence_length, self.num_classes])

        output_dict["logits"] = logits
        output_dict["class_probabilities"] = class_probabilities

        if tags is not None:

            # loss
            if self.crf is not None:
                log_likelihood = self.crf(logits, tags, mask)
                output_dict["loss"] = -log_likelihood
            else:
                loss = sequence_cross_entropy_with_logits(logits, tags, mask)
                output_dict["loss"] = loss

            # metric
            for metric in self.metrics.values():
                metric(logits, tags, mask.float())
            if self._f1_metric is not None:
                self._f1_metric(logits, tags, mask.float())

        if metadata is not None:
            output_dict["words"] = [x["words"] for x in metadata]

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple position-wise argmax over each token, converts indices to string labels, and
        adds a ``"tags"`` key to the dictionary with the result.
        """
        if output_dict.get('tags'):
            output_dict["tags"] = [
                [self.vocab.get_token_from_index(tag, namespace=self.label_namespace) for tag in instance_tags]
                for instance_tags in output_dict["tags"]
            ]
        else:
            all_predictions = output_dict['class_probabilities']
            all_predictions = all_predictions.cpu().data.numpy()
            if all_predictions.ndim == 3:
                predictions_list = [all_predictions[i] for i in range(all_predictions.shape[0])]
            else:
                predictions_list = [all_predictions]
            all_tags = []
            for predictions in predictions_list:
                argmax_indices = numpy.argmax(predictions, axis=-1)
                tags = [self.vocab.get_token_from_index(x, namespace="labels") for x in argmax_indices]
                all_tags.append(tags)
            output_dict['tags'] = all_tags
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}

        if self._f1_metric is not None:
            f1_dict = self._f1_metric.get_metric(reset=reset)
            if self._verbose_metrics:
                metrics_to_return.update(f1_dict)
            else:
                metrics_to_return.update({x: y for x, y in f1_dict.items() if "overall" in x})
        return metrics_to_return