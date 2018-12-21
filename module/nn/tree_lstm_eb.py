# tree lstm for elmo and bert model
from typing import Dict, List, Tuple, cast, Union

import torch.nn as nn
import torch
from .biattentive_cell import BiattentiveClassificationNetwork
from ..module_io.tree import Tree
import numpy as np

from allennlp.data.vocabulary import Vocabulary
from allennlp.data.fields import LabelField, TextField, Field
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.modules import Elmo, TextFieldEmbedder
from allennlp.data.dataset import Batch
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.nn.util import move_to_device
from .crf_be import *

SORTING_KEYS = [("tokens", "num_tokens")]


def sort_by_padding(instances: List[Instance],
                    sorting_keys: List[Tuple[str, str]],  # pylint: disable=invalid-sequence-index
                    vocab: Vocabulary) -> Tuple[List[int], List[Instance]]:
    """
    Sorts the instances by their padding lengths, using the keys in
    ``sorting_keys`` (in the order in which they are provided).  ``sorting_keys`` is a list of
    ``(field_name, padding_key)`` tuples.
    """
    instances_with_lengths = []
    for idx, instance in enumerate(instances):
        # Make sure instance is indexed before calling .get_padding
        instance.index_fields(vocab)
        padding_lengths = cast(Dict[str, Dict[str, float]], instance.get_padding_lengths())
        # if padding_noise > 0.0:
        #     noisy_lengths = {}
        #     for field_name, field_lengths in padding_lengths.items():
        #         noisy_lengths[field_name] = add_noise_to_dict_values(field_lengths, padding_noise)
        #     padding_lengths = noisy_lengths
        instance_with_lengths = ([padding_lengths[field_name][padding_key]
                                  for (field_name, padding_key) in sorting_keys],
                                 instance, idx)
        instances_with_lengths.append(instance_with_lengths)
    instances_with_lengths.sort(key=lambda x: x[0])

    return [instance_with_lengths[-1] for instance_with_lengths in instances_with_lengths], \
           [instance_with_lengths[-2] for instance_with_lengths in instances_with_lengths]


def set_logits_back(tree: Tree, tensor: torch.Tensor, idx: List):
    for child in tree.children:
        set_logits_back(child, tensor, idx)
    tree.crf_cache['be_hidden'] = tensor[idx[0]]
    del idx[0]


class Biattentive(nn.Module):

    def __init__(self, vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 embedding_dropout_prob: float,
                 word_dim: int,
                 use_input_elmo: bool,
                 pre_encode_dim: Union[int, List[int]],
                 pre_encode_layer_dropout_prob: Union[float, List[float]],
                 encode_output_dim: int,
                 integrtator_output_dim: int,
                 integrtator_dropout: float,
                 use_integrator_output_elmo: bool,
                 output_dim: Union[int, List[int]],
                 output_pool_size: int,
                 output_dropout: Union[float, List[float]],
                 elmo: Elmo,
                 # fixme the class of token_indexer
                 token_indexer,
                 device) -> None:
        super(Biattentive, self).__init__()
        self.token_indexers = token_indexer
        self.vocab = vocab
        self.elmo = elmo
        self.biattentive_cell = BiattentiveClassificationNetwork(embedder, embedding_dropout_prob, word_dim,
                                                                 use_input_elmo, pre_encode_dim,
                                                                 pre_encode_layer_dropout_prob, encode_output_dim,
                                                                 integrtator_output_dim, integrtator_dropout,
                                                                 use_integrator_output_elmo, output_dim,
                                                                 output_pool_size, output_dropout, elmo)
        self.nll_loss = nn.CrossEntropyLoss()
        self.device = device

        self.metrics = {"accuracy": CategoricalAccuracy(), }

    def text_to_instance(self, tokens: List[str], sentiment: str = None) -> Instance:  # type: ignore
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.

        Parameters
        ----------
        tokens : ``List[str]``, required.
            The tokens in a given sentence.
        sentiment ``str``, optional, (default = None).
            The sentiment for this sentence.

        Returns
        -------
        An ``Instance`` containing the following fields:
            tokens : ``TextField``
                The tokens in the sentence or phrase.
            label : ``LabelField``
                The sentiment label of the sentence or phrase.
                :param tokens:
                :param sentiment:
        """
        # pylint: disable=arguments-differ
        text_field = TextField([Token(x) for x in tokens], token_indexers=self.token_indexers)
        fields: Dict[str, Field] = {"tokens": text_field}
        if sentiment is not None:
            # 0 and 1 are negative sentiment, 2 is neutral, and 3 and 4 are positive sentiment
            # In 5-class, we use labels as is.
            # 3-class reduces the granularity, and only asks the model to predict
            # negative, neutral, or positive.
            # 2-class further reduces the granularity by only asking the model to
            # predict whether an instance is negative or positive.
            if self._granularity == "3-class":
                if int(sentiment) < 2:
                    sentiment = "0"
                elif int(sentiment) == 2:
                    sentiment = "1"
                else:
                    sentiment = "2"
            elif self._granularity == "2-class":
                if int(sentiment) < 2:
                    sentiment = "0"
                elif int(sentiment) == 2:
                    return None
                else:
                    sentiment = "1"
            fields['label'] = LabelField(sentiment)
        return Instance(fields)

    def collect_phase(self, tree, holder):
        for child in tree.children:
            self.collect_phase(child, holder)
        holder.append(tree.get_str_yield())

    def forward(self, tree: Tree,
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        str_phase_holder = []
        self.collect_phase(tree, str_phase_holder)
        # tokenize and elmo tokenize
        instances = [self.text_to_instance(phase) for phase in str_phase_holder]
        idx, instances = sort_by_padding(instances, [("tokens", "num_tokens")], self.vocab)
        batch = Batch(instances)
        pad_lengths = batch.get_padding_lengths()
        tensor_dict = batch.as_tensor_dict(pad_lengths)
        tensor_dict = move_to_device(tensor_dict, 0)
        output = self.biattentive_cell(**tensor_dict)
        # resort output result
        new_idx = [i for i in range(len(instances))]
        for pos, name in enumerate(idx):
            new_idx[name] = pos
        for name, tensor in output.items():
            output[name] = torch.stack([tensor[i] for i in new_idx])
        if label is not None:
            loss = self.nll_loss(output['logits'], label)
            for metric in self.metrics.values():
                metric(output['logits'], label)
            output["loss"] = loss
        return output

    def loss(self, tree):
        label_holder = []
        tree.collect_golden_labels(label_holder)
        label_holder = torch.tensor(label_holder).to(self.device)
        output = self.forward(tree, label=label_holder)
        return output

    def predict(self, tree):
        output_dict = self.forward(tree)
        predictions = output_dict['class_probabilities'].cpu().data.numpy()
        labels = np.argmax(predictions, axis=-1)
        # labels = [int(self.vocab.get_token_from_index(x, namespace="labels"))
        #           for x in argmax_indices]

        output_dict['label'] = np.array(labels)

        label_holder = []
        tree.collect_golden_labels(label_holder)
        golden_label = np.array(label_holder)
        # fine gain target
        corr = np.equal(labels, golden_label).astype(float)

        # binary target
        binary_mask = np.not_equal(golden_label, 2).astype(int)
        binary_preds = ((predictions[:, 3] + predictions[:, 4]) > (predictions[:, 1] + predictions[:, 2])).astype(int)
        binary_lables = np.greater(labels, 2).astype(int)
        binary_corr = (np.equal(binary_preds, binary_lables) * binary_mask).astype(float)

        output_dict['corr'] = corr
        output_dict['binary_mask'] = binary_mask
        output_dict['binary_corr'] = [binary_corr]
        output_dict['binary_pred'] = [binary_preds]
        return output_dict


class CRFBiattentive(Biattentive):
    def __init__(self, vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 embedding_dropout_prob: float,
                 word_dim: int,
                 use_input_elmo: bool,
                 pre_encode_dim: Union[int, List[int]],
                 pre_encode_layer_dropout_prob: Union[float, List[float]],
                 encode_output_dim: int,
                 integrtator_output_dim: int,
                 integrtator_dropout: float,
                 use_integrator_output_elmo: bool,
                 output_dim: Union[int, List[int]],
                 output_pool_size: int,
                 output_dropout: Union[float, List[float]],
                 elmo: Elmo,
                 # fixme the class of token_indexer
                 token_indexer,
                 device) -> None:
        super(CRFBiattentive, self).__init__(vocab, embedder, embedding_dropout_prob, word_dim,
                                             use_input_elmo, pre_encode_dim, pre_encode_layer_dropout_prob,
                                             encode_output_dim, integrtator_output_dim, integrtator_dropout,
                                             use_integrator_output_elmo, output_dim, output_pool_size,
                                             output_dropout, elmo, token_indexer, device)
        if isinstance(output_dim, List):
            self.crf = TreeCRF(output_dim[-1])
        else:
            self.crf = TreeCRF(output_dim)

    def loss(self, tree):
        label_holder = []
        tree.collect_golden_labels(label_holder)
        idx_list = [i for i in range(len(label_holder))]
        label_holder = torch.tensor(label_holder).to(self.device)
        output = self.forward(tree, label=label_holder)
        set_logits_back(tree, output['logits'], idx_list)
        loss = self.crf.loss(tree)
        output['loss'] = loss

        return output

    def predict(self, tree):
        label_holder = []
        tree.collect_golden_labels(label_holder)
        golden_label = np.array(label_holder)
        idx_list = [i for i in range(len(label_holder))]
        output_dict = self.forward(tree)
        set_logits_back(tree, output_dict['logits'], idx_list)
        labels = np.array(self.crf.predict(tree))

        output_dict['label'] = labels
        # fine gain target
        corr = np.equal(labels, golden_label).astype(float)

        # binary target
        binary_mask = np.not_equal(golden_label, 2).astype(int)
        binary_preds_1 = np.greater(labels, 2)
        binary_preds_2 = np.greater_equal(labels, 2)
        binary_lables = np.greater(golden_label, 2)
        binary_corr_1 = (np.equal(binary_preds_1, binary_lables) * binary_mask).astype(float)
        binary_corr_2 = (np.equal(binary_preds_2, binary_lables) * binary_mask).astype(float)

        output_dict['corr'] = corr
        output_dict['binary_mask'] = binary_mask
        output_dict['binary_corr'] = [binary_corr_1, binary_corr_2]
        output_dict['binary_pred'] = [binary_preds_1, binary_preds_2]
        return output_dict


class BiCRFBiattentive(Biattentive):
    def __init__(self, vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 embedding_dropout_prob: float,
                 word_dim: int,
                 use_input_elmo: bool,
                 pre_encode_dim: Union[int, List[int]],
                 pre_encode_layer_dropout_prob: Union[float, List[float]],
                 encode_output_dim: int,
                 integrtator_output_dim: int,
                 integrtator_dropout: float,
                 use_integrator_output_elmo: bool,
                 output_dim: Union[int, List[int]],
                 output_pool_size: int,
                 output_dropout: Union[float, List[float]],
                 elmo: Elmo,
                 # fixme the class of token_indexer
                 token_indexer,
                 device) -> None:
        super(BiCRFBiattentive, self).__init__(vocab, embedder, embedding_dropout_prob, word_dim,
                                               use_input_elmo, pre_encode_dim, pre_encode_layer_dropout_prob,
                                               encode_output_dim, integrtator_output_dim, integrtator_dropout,
                                               use_integrator_output_elmo, output_dim, output_pool_size,
                                               output_dropout, elmo, token_indexer, device)
        if isinstance(output_dim, List):
            self.crf = BinaryTreeCRF(output_dim[-1])
        else:
            self.crf = BinaryTreeCRF(output_dim)

    def loss(self, tree):
        label_holder = []
        tree.collect_golden_labels(label_holder)
        idx_list = [i for i in range(len(label_holder))]
        label_holder = torch.tensor(label_holder).to(self.device)
        output = self.forward(tree, label=label_holder)
        set_logits_back(tree, output['logits'], idx_list)
        loss = self.crf.loss(tree)
        output['loss'] = loss

        return output

    def predict(self, tree):
        label_holder = []
        tree.collect_golden_labels(label_holder)
        golden_label = np.array(label_holder)
        idx_list = [i for i in range(len(label_holder))]
        output_dict = self.forward(tree)
        set_logits_back(tree, output_dict['logits'], idx_list)
        labels = np.array(self.crf.predict(tree))

        output_dict['label'] = labels
        # fine gain target
        corr = np.equal(labels, golden_label).astype(float)

        # binary target
        binary_mask = np.not_equal(golden_label, 2).astype(int)
        binary_preds_1 = np.greater(labels, 2)
        binary_preds_2 = np.greater_equal(labels, 2)
        binary_lables = np.greater(golden_label, 2)
        binary_corr_1 = (np.equal(binary_preds_1, binary_lables) * binary_mask).astype(float)
        binary_corr_2 = (np.equal(binary_preds_2, binary_lables) * binary_mask).astype(float)

        output_dict['corr'] = corr
        output_dict['binary_mask'] = binary_mask
        output_dict['binary_corr'] = [binary_corr_1, binary_corr_2]
        output_dict['binary_pred'] = [binary_preds_1, binary_preds_2]
        return output_dict
