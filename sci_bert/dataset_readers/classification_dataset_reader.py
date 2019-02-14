""" Data reader for AllenNLP """


from typing import Dict, List, Any
import logging

import jsonlines
from overrides import overrides
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, MultiLabelField, ListField, ArrayField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("classification_dataset_reader")
class ClassificationDatasetReader(DatasetReader):
    """
    Text classification data reader

    The data is assumed to be in jsonlines format
    each line is a json-dict with the following keys: 'text', 'label', 'metadata'
    'metadata' is optional and only used for passing metadata to the model
    """
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 ) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        with jsonlines.open(file_path) as f_in:
            for json_object in f_in:
                yield self.text_to_instance(
                    text=json_object.get('text'),
                    label=json_object.get('label'),
                    metadata=json_object.get('metadata')
                )

    @overrides
    def text_to_instance(self,
                         text: str,
                         label: str = None,
                         metadata: Any = None) -> Instance:  # type: ignore
        text_tokens = self._tokenizer.tokenize(text)
        fields = {
            'text': TextField(text_tokens, self._token_indexers),
        }
        if label is not None:
            fields['label'] = LabelField(label)

        if metadata:
            fields['metadata'] = MetadataField(metadata)
        return Instance(fields)
