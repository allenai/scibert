from typing import Dict, Optional, List, Any
import logging

from overrides import overrides
import torch
from torch.nn.modules.linear import Linear

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules import ConditionalRandomField
from allennlp.nn import InitializerApplicator, RegularizerApplicator
import allennlp.nn.util as util
from allennlp.training.metrics import F1Measure

logger = logging.getLogger(__name__)

@Model.register("pico_crf_tagger")
class PicoCrfTagger(Model):
    """
    Exactly like the CrfTagger in AllenNLP:
    https://github.com/allenai/allennlp/blob/master/allennlp/models/crf_tagger.py

    But differences include:
    - No option for `constrain_crf_decoding` because only supports IO-encoding
      (because that's how EBMNLP dataset is annotated)
    - No option for `calculate_span_f1` because PICO is evaluated at token-level
    - No option for `verbose_metrics`.  Defaults to printing all because in PICO,
      we want to see F1 scores for each class.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
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

        # crf
        output_dim = self.encoder.get_output_dim()
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
                tags: torch.LongTensor = None,
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

        logits = self.tag_projection_layer(encoded_text)
        best_paths = self.crf.viterbi_tags(logits, mask)

        # Just get the tags and ignore the score.
        predicted_tags = [x for x, y in best_paths]

        output = {"logits": logits, "mask": mask, "tags": predicted_tags}

        if tags is not None:
            # Add negative log-likelihood as loss
            log_likelihood = self.crf(logits, tags, mask)
            output["loss"] = -log_likelihood

            # Represent viterbi tags as "class probabilities" that we can
            # feed into the metrics
            class_probabilities = logits * 0.
            for i, instance_tags in enumerate(predicted_tags):
                for j, tag_id in enumerate(instance_tags):
                    class_probabilities[i, j, tag_id] = 1

            for metric in self.metrics.values():
                metric(class_probabilities, tags, mask.float())
            if self.calculate_span_f1:
                self._f1_metric(class_probabilities, tags, mask.float())
        if metadata is not None:
            output["words"] = [x["words"] for x in metadata]
        return output

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Converts the tag ids to the actual tags.
        ``output_dict["tags"]`` is a list of lists of tag_ids,
        so we use an ugly nested list comprehension.
        """
        output_dict["tags"] = [
            [
                self.vocab.get_token_from_index(tag, namespace=self.label_namespace)
                for tag in instance_tags
            ]
            for instance_tags in output_dict["tags"]
        ]

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = {}

        total_f1, total_classes = 0, 0
        for metric_name, metric_obj in self.metrics.items():
            if metric_name.startswith('accuracy'):
                metrics_to_return[metric_name] = metric_obj.get_metric(reset)
            elif metric_name.startswith('F1_'):
                p, r, f1 = metric_obj.get_metric(reset)
                metrics_to_return[metric_name] = f1
                total_f1 += f1
                total_classes += 1
        metrics_to_return['avg_f1'] = total_f1 / total_classes

        return metrics_to_return