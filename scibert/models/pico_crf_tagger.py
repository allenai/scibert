from typing import Dict, Optional, List, Any
import logging
import warnings

from overrides import overrides
import torch
from torch.nn.modules.linear import Linear

from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules import ConditionalRandomField, FeedForward
from allennlp.nn import InitializerApplicator, RegularizerApplicator
import allennlp.nn.util as util
from allennlp.training.metrics import F1Measure

logger = logging.getLogger(__name__)

@Model.register("pico_crf_tagger")
class PicoCrfTagger(Model):
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 feedforward: Optional[FeedForward] = None,
                 include_start_end_transitions: bool = True,
                 dropout: Optional[float] = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self.label_namespace = 'labels'
        self.num_tags = self.vocab.get_vocab_size(self.label_namespace)

        # encode text
        self.text_field_embedder = text_field_embedder
        self.encoder = encoder
        self.dropout = torch.nn.Dropout(dropout) if dropout else None
        self.feedforward = feedforward

        # crf
        output_dim = self.encoder.get_output_dim() if feedforward is None else feedforward.get_output_dim()
        self.tag_projection_layer = TimeDistributed(Linear(output_dim, self.num_tags))
        self.crf = ConditionalRandomField(self.num_tags, constraints=None, include_start_end_transitions=include_start_end_transitions)

        initializer(self)

        self.metrics = {}

        # Add F1 score for individual labels to metrics 
        for index, label in self.vocab.get_index_to_token_vocabulary(self.label_namespace).items():
            self.metrics[label] = F1Measure(positive_label=index)

    @overrides
    def forward(self,
                tokens: Dict[str, torch.LongTensor],
                labels: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None,
                **kwargs) -> Dict[str, torch.Tensor]:

        # (batch, tokens, dim)
        embedded_text_input = self.text_field_embedder(tokens)
        mask = util.get_text_field_mask(tokens)
        if self.dropout:
            embedded_text_input = self.dropout(embedded_text_input)

        # (batch, tokens, dim)
        encoded_text = self.encoder(embedded_text_input, mask)
        if self.dropout:
            encoded_text = self.dropout(encoded_text)

        # (batch, tokens, dim)
        if self.feedforward is not None:
            encoded_text = self.feedforward(encoded_text)

        # do crf
        logits = self.tag_projection_layer(encoded_text)

        # decoding:  just get the tags and ignore the score.
        best_paths = self.crf.viterbi_tags(logits, mask)
        predicted_tags = [x for x, y in best_paths]

        output = {"logits": logits, "mask": mask, "labels": predicted_tags}

        if labels is not None:
            # Add negative log-likelihood as loss
            log_likelihood = self.crf(logits, labels, mask)
            output["loss"] = -log_likelihood

            # Represent viterbi tags as "class probabilities" that we can
            # feed into the metrics
            class_probabilities = logits * 0.
            for i, instance_tags in enumerate(predicted_tags):
                for j, tag_id in enumerate(instance_tags):
                    class_probabilities[i, j, tag_id] = 1

            for metric in self.metrics.values():
                metric(class_probabilities, labels, mask.float())

        if metadata is not None:
            output["words"] = [x["words"] for x in metadata]

        return output

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Converts the tag ids to the actual tags.
        ``output_dict["labels"]`` is a list of lists of tag_ids,
        so we use an ugly nested list comprehension.
        """
        output_dict["labels"] = [
                [self.vocab.get_token_from_index(tag, namespace=self.label_namespace)
                 for tag in instance_tags]
                for instance_tags in output_dict["labels"]
        ]

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = {}

        sum_f1 = 0.
        num_labels = 0.

        # for each PICO label
        for index, label in self.vocab.get_index_to_token_vocabulary(self.label_namespace).items():
            p, r, f1 = self.metrics[label].get_metric(reset)
            p_name = "{0}-p".format(label)
            r_name = "{0}-r".format(label)
            f1_name = "{0}-f1".format(label)

            # compute metrics & save & accumulate
            num_labels += 1
            for score, metric in zip([p, r, f1], [p_name, r_name, f1_name]):
                metrics_to_return[metric] = score 
                if metric.endswith('f1'):
                    sum_f1 += score

        # Average across labels 
        metrics_to_return["avg_f1"] = sum_f1 / num_labels

        return metrics_to_return
