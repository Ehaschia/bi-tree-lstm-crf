__author__ = 'max'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .tree_lstm_cell import TreeLSTMCell_flod


class BiAAttention(nn.Module):
    """
    Bi-Affine attention layer.
    """

    def __init__(self, input_size_encoder, input_size_decoder, num_labels, biaffine=True, **kwargs):
        """

        Args:
            input_size_encoder: int
                the dimension of the encoder input.
            input_size_decoder: int
                the dimension of the decoder input.
            num_labels: int
                the number of labels of the crf layer
            biaffine: bool
                if apply bi-affine parameter.
            **kwargs:
        """
        super(BiAAttention, self).__init__()
        self.input_size_encoder = input_size_encoder
        self.input_size_decoder = input_size_decoder
        self.num_labels = num_labels
        self.biaffine = biaffine

        self.W_d = Parameter(torch.Tensor(self.num_labels, self.input_size_decoder))
        self.W_e = Parameter(torch.Tensor(self.num_labels, self.input_size_encoder))
        self.b = Parameter(torch.Tensor(self.num_labels, 1, 1))
        if self.biaffine:
            self.U = Parameter(torch.Tensor(self.num_labels, self.input_size_decoder, self.input_size_encoder))
        else:
            self.register_parameter('U', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform(self.W_d)
        nn.init.xavier_uniform(self.W_e)
        nn.init.constant(self.b, 0.)
        if self.biaffine:
            nn.init.xavier_uniform(self.U)

    def forward(self, input_d, input_e, mask_d=None, mask_e=None):
        """

        Args:
            input_d: Tensor
                the decoder input tensor with shape = [batch, length_decoder, input_size]
            input_e: Tensor
                the child input tensor with shape = [batch, length_encoder, input_size]
            mask_d: Tensor or None
                the mask tensor for decoder with shape = [batch, length_decoder]
            mask_e: Tensor or None
                the mask tensor for encoder with shape = [batch, length_encoder]

        Returns: Tensor
            the energy tensor with shape = [batch, num_label, length, length]

        """
        assert input_d.size(0) == input_e.size(0), 'batch sizes of encoder and decoder are requires to be equal.'
        batch, length_decoder, _ = input_d.size()
        _, length_encoder, _ = input_e.size()

        # compute decoder part: [num_label, input_size_decoder] * [batch, input_size_decoder, length_decoder]
        # the output shape is [batch, num_label, length_decoder]
        out_d = torch.matmul(self.W_d, input_d.transpose(1, 2)).unsqueeze(3)
        # compute decoder part: [num_label, input_size_encoder] * [batch, input_size_encoder, length_encoder]
        # the output shape is [batch, num_label, length_encoder]
        out_e = torch.matmul(self.W_e, input_e.transpose(1, 2)).unsqueeze(2)

        # output shape [batch, num_label, length_decoder, length_encoder]
        if self.biaffine:
            # compute bi-affine part
            # [batch, 1, length_decoder, input_size_decoder] * [num_labels, input_size_decoder, input_size_encoder]
            # output shape [batch, num_label, length_decoder, input_size_encoder]
            output = torch.matmul(input_d.unsqueeze(1), self.U)
            # [batch, num_label, length_decoder, input_size_encoder] * [batch, 1, input_size_encoder, length_encoder]
            # output shape [batch, num_label, length_decoder, length_encoder]
            output = torch.matmul(output, input_e.unsqueeze(1).transpose(2, 3))

            output = output + out_d + out_e + self.b
        else:
            output = out_d + out_d + self.b

        if mask_d is not None:
            output = output * mask_d.unsqueeze(1).unsqueeze(3) * mask_e.unsqueeze(1).unsqueeze(2)

        return output


class ConcatAttention(nn.Module):
    """
    Concatenate attention layer.
    """
    # TODO test it!

    def __init__(self, input_size_encoder, input_size_decoder, hidden_size, num_labels, **kwargs):
        """

        Args:
            input_size_encoder: int
                the dimension of the encoder input.
            input_size_decoder: int
                the dimension of the decoder input.
            hidden_size: int
                the dimension of the hidden.
            num_labels: int
                the number of labels of the crf layer
            biaffine: bool
                if apply bi-affine parameter.
            **kwargs:
        """
        super(ConcatAttention, self).__init__()
        self.input_size_encoder = input_size_encoder
        self.input_size_decoder = input_size_decoder
        self.hidden_size = hidden_size
        self.num_labels = num_labels

        self.W_d = Parameter(torch.Tensor(self.input_size_decoder, self.hidden_size))
        self.W_e = Parameter(torch.Tensor(self.input_size_encoder, self.hidden_size))
        self.b = Parameter(torch.Tensor(self.hidden_size))
        self.v = Parameter(torch.Tensor(self.hidden_size, self.num_labels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform(self.W_d)
        nn.init.xavier_uniform(self.W_e)
        nn.init.xavier_uniform(self.v)
        nn.init.constant(self.b, 0.)

    def forward(self, input_d, input_e, mask_d=None, mask_e=None):
        """

        Args:
            input_d: Tensor
                the decoder input tensor with shape = [batch, length_decoder, input_size]
            input_e: Tensor
                the child input tensor with shape = [batch, length_encoder, input_size]
            mask_d: Tensor or None
                the mask tensor for decoder with shape = [batch, length_decoder]
            mask_e: Tensor or None
                the mask tensor for encoder with shape = [batch, length_encoder]

        Returns: Tensor
            the energy tensor with shape = [batch, num_label, length, length]

        """
        assert input_d.size(0) == input_e.size(0), 'batch sizes of encoder and decoder are requires to be equal.'
        batch, length_decoder, _ = input_d.size()
        _, length_encoder, _ = input_e.size()

        # compute decoder part: [batch, length_decoder, input_size_decoder] * [input_size_decoder, hidden_size]
        # the output shape is [batch, length_decoder, hidden_size]
        # then --> [batch, 1, length_decoder, hidden_size]
        out_d = torch.matmul(input_d, self.W_d).unsqueeze(1)
        # compute decoder part: [batch, length_encoder, input_size_encoder] * [input_size_encoder, hidden_size]
        # the output shape is [batch, length_encoder, hidden_size]
        # then --> [batch, length_encoder, 1, hidden_size]
        out_e = torch.matmul(input_e, self.W_e).unsqueeze(2)

        # add them together [batch, length_encoder, length_decoder, hidden_size]
        out = F.tanh(out_d + out_e + self.b)

        # product with v
        # [batch, length_encoder, length_decoder, hidden_size] * [hidden, num_label]
        # [batch, length_encoder, length_decoder, num_labels]
        # then --> [batch, num_labels, length_decoder, length_encoder]
        return torch.matmul(out, self.v).transpose(1, 3)


class CoAttention(nn.Module):

    def __init__(self, input_dim, mid_dim):
        # fixme the dimention of softmax
        super(CoAttention, self).__init__()

        self.softmax1 = nn.Softmax(dim=1)
        # here I guess only has two tree format
        # fixme only implement other tree lstm here
        self.tree_input_dim = input_dim
        self.tree_output_dim = mid_dim
        self.tree_cell = TreeLSTMCell_flod(input_dim, mid_dim)

    def forward(self, tree):
        hidden_holder = []
        tree.collect_hidden_state(hidden_holder)
        hidden_mat = torch.cat(hidden_holder, dim=0)

        attention_logits = hidden_mat.mm(hidden_mat.t())
        attention_weights = self.softmax1(attention_logits)
        # encoded_text = self.weighted_sum(hidden_mat, attention_weights)
        encoded_text = attention_weights.mm(hidden_mat)

        integrate_cat = torch.cat([hidden_mat, hidden_mat - encoded_text, hidden_mat * encoded_text], dim=-1)

        self.tree_forward(tree, 0, integrate_cat)

    def tree_forward(self, tree, idx, node_state):
        for child in tree.children:
            idx = self.tree_forward(child, idx, node_state)

        inputs = node_state[idx]
        if tree.is_leaf():
            l_c = inputs.detach().new(self.tree_output_dim).fill_(0.).requires_grad_()
            r_c = inputs.detach().new(self.tree_output_dim).fill_(0.).requires_grad_()

            l_h = inputs.detach().new(self.tree_output_dim).fill_(0.).requires_grad_()
            r_h = inputs.detach().new(self.tree_output_dim).fill_(0.).requires_grad_()
            l = {'h': l_h, 'c': l_c}
            r = {'h': r_h, 'c': r_c}
            tree.bu_state = self.tree_cell(l, r, inputs)
        else:
            l = tree.get_child(0).bu_state
            r = tree.get_child(1).bu_state
            tree.td_state = {}
            tree.bu_state = self.tree_cell(l, r, inputs)
        return idx + 1

    def weighted_sum(self, matrix: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
        """
        Takes a matrix of vectors and a set of weights over the rows in the matrix (which we call an
        "attention" vector), and returns a weighted sum of the rows in the matrix.  This is the typical
        computation performed after an attention mechanism.
        Note that while we call this a "matrix" of vectors and an attention "vector", we also handle
        higher-order tensors.  We always sum over the second-to-last dimension of the "matrix", and we
        assume that all dimensions in the "matrix" prior to the last dimension are matched in the
        "vector".  Non-matched dimensions in the "vector" must be `directly after the batch dimension`.
        For example, say I have a "matrix" with dimensions ``(batch_size, num_queries, num_words,
        embedding_dim)``.  The attention "vector" then must have at least those dimensions, and could
        have more. Both:
            - ``(batch_size, num_queries, num_words)`` (distribution over words for each query)
            - ``(batch_size, num_documents, num_queries, num_words)`` (distribution over words in a
              query for each document)
        are valid input "vectors", producing tensors of shape:
        ``(batch_size, num_queries, embedding_dim)`` and
        ``(batch_size, num_documents, num_queries, embedding_dim)`` respectively.
        """
        # We'll special-case a few settings here, where there are efficient (but poorly-named)
        # operations in pytorch that already do the computation we need.
        if attention.dim() == 2 and matrix.dim() == 3:
            return attention.unsqueeze(1).bmm(matrix).squeeze(1)
        if attention.dim() == 3 and matrix.dim() == 3:
            return attention.bmm(matrix)
        if matrix.dim() - 1 < attention.dim():
            expanded_size = list(matrix.size())
            for i in range(attention.dim() - matrix.dim() + 1):
                matrix = matrix.unsqueeze(1)
                expanded_size.insert(i + 1, attention.size(i + 1))
            matrix = matrix.expand(*expanded_size)
        intermediate = attention.unsqueeze(-1).expand_as(matrix) * matrix
        return intermediate.sum(dim=-2)

# class Attention(nn.Module):
#     # TODO