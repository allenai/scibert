from overrides import overrides
import torch

from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder

@Seq2SeqEncoder.register("dummy")
class DummyEncoder(Seq2SeqEncoder):
    # pylint: disable=line-too-long
    """
    A dummy seq2seq encoder that just returns its inputs as is.
    
    Parameters
    ----------
    input_dim : ``int``, required.
        The input dimension of the encoder.
    """
    def __init__(self,
                 input_dim: int,) -> None:
        super(DummyEncoder, self).__init__()
        self._input_dim = input_dim

    @overrides
    def get_input_dim(self) -> int:
        return self._input_dim

    @overrides
    def get_output_dim(self) -> int:
        return self._input_dim

    @overrides
    def is_bidirectional(self):
        return False

    @overrides
    def forward(self, inputs: torch.Tensor, mask: torch.Tensor): # pylint: disable=arguments-differ
        return inputs