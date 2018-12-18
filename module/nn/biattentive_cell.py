from typing import Dict, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.nn import InitializerApplicator
from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.modules import Elmo, FeedForward, Maxout, Seq2SeqEncoder, TextFieldEmbedder
from allennlp.nn import util

ELMO_DIM = 1024


class BiattentiveClassificationNetwork(nn.Module):

    def __init__(self, embedder: TextFieldEmbedder,
                 embedding_dropout_prob: float,
                 embedding_dim: int,
                 use_input_elmo: bool,
                 pre_encode_dim: Union[int, List[int]],
                 pre_encode_layer_dropout_prob: Union[float, List[float]],
                 encoder_dim: int,
                 integrtator_dim: int,
                 integrator_dropout: float,
                 use_integrator_output_elmo: bool,
                 output_dim: Union[int, List[int]],
                 output_pool_size: int,
                 output_dropout_prob: Union[float, List[float]],
                 elmo: Elmo):
        super(BiattentiveClassificationNetwork, self).__init__()
        # pre_encode_feedforward

        self.text_field_embedder = embedder
        self.embedding_dropout = nn.Dropout(embedding_dropout_prob)
        self.use_input_elmo = use_input_elmo
        embedding_dim += ELMO_DIM if self.use_input_elmo else 0
        if isinstance(pre_encode_dim, int):
            pre_encode_layer_num = 1
            pre_encode_dim = [pre_encode_dim]
            pre_encode_layer_dropout_prob = [pre_encode_layer_dropout_prob]
        else:
            pre_encode_layer_num = len(pre_encode_dim)

        self.pre_encode_feedforward = FeedForward(input_dim=embedding_dim, num_layers=pre_encode_layer_num,
                                                  hidden_dims=pre_encode_dim,
                                                  # fixme debug here
                                                  activations=[nn.ReLU()] * pre_encode_layer_num,
                                                  dropout=pre_encode_layer_dropout_prob)
        pytorch_encoder = nn.LSTM(input_size=pre_encode_dim[-1], hidden_size=encoder_dim, num_layers=1,
                                  bidirectional=True, batch_first=True)
        self.encoder = PytorchSeq2SeqWrapper(pytorch_encoder)
        pytorch_integrator = nn.LSTM(input_size=6 * encoder_dim, hidden_size=integrtator_dim, num_layers=1,
                                     bidirectional=True, batch_first=True)
        self.integrator = PytorchSeq2SeqWrapper(pytorch_integrator)
        self.integrator_dropout = nn.Dropout(p=integrator_dropout)
        self.use_integrator_output_elmo = use_integrator_output_elmo

        if self.use_integrator_output_elmo:
            self.combined_integrator_output_dim = (self.integrator.get_output_dim() +
                                                   self.elmo.get_output_dim())
        else:
            self.combined_integrator_output_dim = self.integrator.get_output_dim()

        self.self_attentive_pooling_projection = nn.Linear(self.combined_integrator_output_dim, 1)

        if isinstance(output_dim, int):
            output_layer_num = 1
            output_dim = [output_dim]
            output_dropout_prob = [output_dropout_prob]
        else:
            output_layer_num = len(output_dim)

        self.output_layer = Maxout(input_dim=integrtator_dim * 8, num_layers=output_layer_num,
                                   output_dims=output_dim, pool_sizes=output_pool_size, dropout=output_dropout_prob)

        initializer = InitializerApplicator()
        self.elmo = elmo
        initializer(self)

    def forward(self, tokens: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:
        text_mask = util.get_text_field_mask(tokens).float()

        elmo_tokens = tokens.pop("elmo", None)

        if tokens:
            embedded_text = self.text_field_embedder(tokens)
        else:
            embedded_text = None
        # Add the "elmo" key back to "tokens" if not None, since the tests and the
        # subsequent training epochs rely not being modified during forward()
        if elmo_tokens is not None:
            tokens["elmo"] = elmo_tokens

        # Create ELMo embeddings if applicable
        if self.elmo:
            if elmo_tokens is not None:
                elmo_representations = self.elmo(elmo_tokens)["elmo_representations"]
                # Pop from the end is more performant with list
                if self.use_integrator_output_elmo:
                    integrator_output_elmo = elmo_representations.pop()
                if self.use_input_elmo:
                    input_elmo = elmo_representations.pop()
                assert not elmo_representations
            else:
                raise ConfigurationError(
                    "Model was built to use Elmo, but input text is not tokenized for Elmo.")
        if self.use_input_elmo:
            if embedded_text is not None:
                embedded_text = torch.cat([embedded_text, input_elmo], dim=-1)
            else:
                embedded_text = input_elmo

        dropped_embedded_text = self.embedding_dropout(embedded_text)
        pre_encoded_text = self.pre_encode_feedforward(dropped_embedded_text)
        encoded_tokens = self.encoder(pre_encoded_text, text_mask)

        # Compute biattention. This is a special case since the inputs are the same.
        attention_logits = encoded_tokens.bmm(encoded_tokens.permute(0, 2, 1).contiguous())
        attention_weights = util.masked_softmax(attention_logits, text_mask)
        encoded_text = util.weighted_sum(encoded_tokens, attention_weights)

        # Build the input to the integrator
        integrator_input = torch.cat([encoded_tokens,
                                      encoded_tokens - encoded_text,
                                      encoded_tokens * encoded_text], 2)
        integrated_encodings = self.integrator(integrator_input, text_mask)

        # Concatenate ELMo representations to integrated_encodings if specified
        if self.use_integrator_output_elmo:
            integrated_encodings = torch.cat([integrated_encodings,
                                              integrator_output_elmo], dim=-1)

        # Simple Pooling layers
        max_masked_integrated_encodings = util.replace_masked_values(
            integrated_encodings, text_mask.unsqueeze(2), -1e7)
        max_pool = torch.max(max_masked_integrated_encodings, 1)[0]
        min_masked_integrated_encodings = util.replace_masked_values(
            integrated_encodings, text_mask.unsqueeze(2), +1e7)
        min_pool = torch.min(min_masked_integrated_encodings, 1)[0]
        mean_pool = torch.sum(integrated_encodings, 1) / torch.sum(text_mask, 1, keepdim=True)

        # Self-attentive pooling layer
        # Run through linear projection. Shape: (batch_size, sequence length, 1)
        # Then remove the last dimension to get the proper attention shape (batch_size, sequence length).
        self_attentive_logits = self.self_attentive_pooling_projection(
            integrated_encodings).squeeze(2)
        self_weights = util.masked_softmax(self_attentive_logits, text_mask)
        self_attentive_pool = util.weighted_sum(integrated_encodings, self_weights)

        pooled_representations = torch.cat([max_pool, min_pool, mean_pool, self_attentive_pool], 1)
        pooled_representations_dropped = self.integrator_dropout(pooled_representations)

        logits = self.output_layer(pooled_representations_dropped)
        class_probabilities = F.softmax(logits, dim=-1)

        output_dict = {'logits': logits, 'class_probabilities': class_probabilities}
        return output_dict
