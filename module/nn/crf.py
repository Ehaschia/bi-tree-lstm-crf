import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from module.util import logsumexp


class TreeCRF(nn.Module):
    '''
    Tree CRF layer.
    '''

    def __init__(self, input_size, num_labels, attention=True, biaffine=True,
                 only_bu=True, pred_mode=None, softmax_in_dim=64, need_pred_dense=False, bert_dim=0, trans_mat=None):
        '''

        Args:
            input_size: int
                the dimension of the input.
            num_labels: int
                the number of labels of the crf layer
            biaffine: bool
                if apply bi-affine parameter.
            **kwargs:
        '''

        super(TreeCRF, self).__init__()
        self.input_size = input_size
        self.num_labels = num_labels
        self.use_attention = attention
        self.trans_matrix = Parameter(torch.Tensor(self.num_labels, self.num_labels))
        self.only_bu = self.use_attention or only_bu
        self.need_pred_dense = need_pred_dense
        self.dense_softmax = None
        self.pred_layer = None
        self.pred_mode = pred_mode
        self.bert_dim = bert_dim
        if pred_mode == 'single_h':
            pred_input_dim = input_size if self.only_bu else 2 * input_size
            pred_input_dim = pred_input_dim + bert_dim
            self.generate_pred_layer(pred_input_dim , softmax_in_dim, num_labels)
            self.get_emission_score = self.single_h_pred
        elif pred_mode == 'avg_h':
            pred_input_dim = 2 * input_size if self.only_bu else 4 * input_size
            pred_input_dim = pred_input_dim + bert_dim
            self.generate_pred_layer(pred_input_dim, softmax_in_dim, num_labels)
            self.get_emission_score = self.avg_h_pred
        elif pred_mode == 'td_avg_h':
            assert self.only_bu is False
            pred_input_dim = input_size * 3
            pred_input_dim = pred_input_dim + bert_dim
            self.generate_pred_layer(pred_input_dim, softmax_in_dim, num_labels)
            self.get_emission_score = self.td_avg_pred
        else:
            raise NotImplementedError("the pred model " + pred_mode + " is not implemented!")
        # self.attention = BiAAttention(input_size, input_size, num_labels, biaffine=biaffine)
        self.reset_parameter(trans_mat)

    def generate_pred_layer(self, input_size, softmax_in_dim, num_labels):
        if self.need_pred_dense:
            self.dense_softmax = nn.Linear(input_size, softmax_in_dim)
            self.pred_layer = nn.Linear(softmax_in_dim, num_labels)
        else:
            self.pred_layer = nn.Linear(input_size, num_labels)

    def collect_avg_hidden(self, tree):
        hidden_collector = tree.collect_hidden_state([])
        hiddens = torch.cat(hidden_collector, dim=0)
        avg_hidden = torch.mean(hiddens, dim=0)
        return avg_hidden

    def calcualte_avg(self, tree):
        # here calcualte avg is same with bilex paper
        cover_leaf = []
        for child in tree.children:
            cover_leaf += self.calcualte_avg(child)
        if tree.is_leaf():
            tree.td_state['output_h'] = tree.td_state['h']
            return [tree.td_state['output_h']]
        else:
            # alert the dim is hard code
            tree.td_state['output_h'] = torch.mean(torch.stack(cover_leaf, dim=0), dim=0)
            return cover_leaf

    def single_h_pred(self, hidden, avg_h):
        if self.dense_softmax is not None:
            softmax_in = F.relu(self.dense_softmax(hidden))
            pred = self.pred_layer(softmax_in)
        else:
            pred = self.pred_layer(hidden)
        return pred

    def avg_h_pred(self, hidden, avg_h):
        # based on https://www.transacl.org/ojs/index.php/tacl/article/view/925
        hidden = torch.cat([hidden, avg_h], dim=0)
        if self.dense_softmax is not None:
            softmax_in = F.relu(self.dense_softmax(hidden))
            pred = self.pred_layer(softmax_in)
        else:
            pred = self.pred_layer(hidden)

        return pred

    def td_avg_pred(self, hidden, avg_h):
        # based on https://www.transacl.org/ojs/index.php/tacl/article/view/925
        if self.dense_softmax is not None:
            softmax_in = F.relu(self.dense_softmax(hidden))
            pred = self.pred_layer(softmax_in)
        else:
            pred = self.pred_layer(hidden)
        return pred

    def reset_parameter(self, trans_mat):
        if trans_mat is None:
            nn.init.xavier_normal_(self.trans_matrix)
        else:
            self.trans_matrix = Parameter(torch.tensor(self.trans_matrix).float())

    def forward(self, tree, avg_h):
        # just calculate the inside score
        children_score = []
        for child in tree.children:
            children_score.append(self.forward(child, avg_h))

        emission_score = self.calcualte_emission_score(tree, avg_h)
        if tree.is_leaf():
            return emission_score
        else:
            # alert binary tree so only has 2 child
            assert len(children_score) == 2
            children_score[0] = (children_score[0].unsqueeze(0) + self.trans_matrix).unsqueeze(2)
            children_score[1] = (children_score[1].unsqueeze(0) + self.trans_matrix).unsqueeze(1)
            emission_score = emission_score.reshape([self.num_labels, 1, 1])
            for child_score in children_score:
                emission_score = child_score + emission_score

        inside_score = logsumexp(emission_score.reshape(self.num_labels, -1), dim=1)

        return inside_score

    def calcualte_emission_score(self, tree, avg_h):
        if self.only_bu:
            h = tree.bu_state['h']
        else:
            h = torch.cat([tree.bu_state['h'], tree.td_state['h']], dim=0)
        if self.pred_mode == 'td_avg_h' and not self.only_bu:
            h = torch.cat([h, tree.td_state['output_h']], dim=0)
        if self.bert_dim != 0:
            h = torch.cat([h, tree.bert_phase], dim=0)
        emission_score = self.get_emission_score(h, avg_h)
        tree.crf_cache = {"emission_score": emission_score}
        return emission_score

    def golden_score(self, tree):
        golden_score = 0.0
        for child in tree.children:
            golden_score += self.golden_score(child)

        if tree.parent is not None:
            golden_score += (tree.crf_cache['emission_score'].unsqueeze(0) + self.trans_matrix)[
                tree.parent.label, tree.label]
        else:
            golden_score += tree.crf_cache['emission_score'][tree.label]
        return golden_score

    def loss(self, tree):
        if self.pred_mode == 'td_avg_h':
            self.calcualte_avg(tree)
            avg_h = None
        elif self.pred_mode == 'avg_h':
            avg_h = self.collect_avg_hidden(tree)
        else:
            avg_h = None

        inside_score = self.forward(tree, avg_h)
        energy = logsumexp(inside_score, dim=0)

        golden_score = self.golden_score(tree)
        return energy - golden_score

    def viterbi_forward(self, tree, avg_h):
        for child in tree.children:
            self.viterbi_forward(child, avg_h)
        if tree.is_leaf():
            emission_score = self.calcualte_emission_score(tree, avg_h)
            tree.crf_cache['max_score'] = emission_score
        else:
            emission_score = self.calcualte_emission_score(tree, avg_h)
            left_child_max = tree.get_child(0).crf_cache['max_score']
            right_child_max = tree.get_child(1).crf_cache['max_score']

            left_child_max = (left_child_max.unsqueeze(0) + self.trans_matrix).unsqueeze(2)
            right_child_max = (right_child_max.unsqueeze(0) + self.trans_matrix).unsqueeze(1)
            score = emission_score.reshape([self.num_labels, 1, 1]) + left_child_max + right_child_max
            score = score.reshape([self.num_labels, -1])
            max_score, max_idx = torch.max(score, dim=1)
            tree.crf_cache['max_score'] = max_score
            max_left_idx = max_idx // self.num_labels
            max_right_idx = torch.fmod(max_idx, self.num_labels)
            tree.crf_cache['left_idx'] = max_left_idx
            tree.crf_cache['right_idx'] = max_right_idx

    def viterbi_backward(self, tree):
        if tree.is_leaf():
            return

        idx = tree.crf_cache['max_label']
        tree.get_child(0).crf_cache['max_label'] = tree.crf_cache['left_idx'][idx]
        tree.get_child(1).crf_cache['max_label'] = tree.crf_cache['right_idx'][idx]

        for child in tree.children:
            self.viterbi_backward(child)

    def collect_pred(self, tree, holder):
        for child in tree.children:
            self.collect_pred(child, holder)
        holder.append(tree.crf_cache['max_label'])
        return holder

    def predict(self, tree):
        if self.pred_mode == 'td_avg_h':
            self.calcualte_avg(tree)
            avg_h = None
        elif self.pred_mode == 'avg_h':
            avg_h = self.collect_avg_hidden(tree)
        else:
            avg_h = None
        self.viterbi_forward(tree, avg_h)
        max_scores = tree.crf_cache['max_score']
        max_score, idx = torch.max(max_scores, 0)
        tree.crf_cache['max_label'] = idx
        self.viterbi_backward(tree)
        pred = self.collect_pred(tree, [])
        return pred


class BinaryTreeCRF(nn.Module):

    def __init__(self, input_size, num_labels, attention=True,only_bu=True,
                 pred_mode=None, softmax_in_dim=64, need_pred_dense=False, bert_dim=0, trans_mat=None):

        '''

        Args:
            input_size: int
                the dimension of the input.
            num_labels: int
                the number of labels of the crf layer
            biaffine: bool
                if apply bi-affine parameter.
            **kwargs:
        '''

        super(BinaryTreeCRF, self).__init__()
        self.input_size = input_size
        self.num_labels = num_labels
        self.use_attention = attention
        self.trans_matrix = Parameter(torch.Tensor(self.num_labels, self.num_labels, self.num_labels))
        self.only_bu = self.use_attention or only_bu
        self.pred_mode = pred_mode
        self.need_pred_dense = need_pred_dense
        self.dense_softmax = None
        self.bert_dim = bert_dim

        if pred_mode == 'single_h':
            pred_input_dim = input_size if self.only_bu else 2 * input_size
            pred_input_dim = pred_input_dim + bert_dim
            self.generate_pred_layer(pred_input_dim, softmax_in_dim, num_labels)
            self.get_emission_score = self.single_h_pred
        elif pred_mode == 'avg_h':
            pred_input_dim = 2 * input_size if self.only_bu else 4 * input_size
            pred_input_dim = pred_input_dim + bert_dim
            self.generate_pred_layer(pred_input_dim, softmax_in_dim, num_labels)
            self.pred_layer = nn.Linear(softmax_in_dim, num_labels)
            self.get_emission_score = self.avg_h_pred
        elif pred_mode == 'td_avg_h':
            assert self.only_bu is False
            pred_input_dim = input_size * 3
            pred_input_dim = pred_input_dim + bert_dim
            self.generate_pred_layer(pred_input_dim, softmax_in_dim, num_labels)
            self.get_emission_score = self.td_avg_pred
        else:
            raise NotImplementedError("the pred model " + pred_mode + " is not implemented!")
        self.reset_parameter(trans_mat)

    def reset_parameter(self, trans_mat):
        if trans_mat is None:
            nn.init.xavier_normal_(self.trans_matrix)
        else:
            self.trans_matrix = Parameter(torch.tensor(trans_mat).float())

    def collect_avg_hidden(self, tree):
        hidden_collector = tree.collect_hidden_state([])
        hiddens = torch.cat(hidden_collector, dim=0)
        avg_hidden = torch.mean(hiddens, dim=0)
        return avg_hidden

    def calcualte_avg(self, tree):
        # here calcualte avg is same with bilex paper
        cover_leaf = []
        for child in tree.children:
            cover_leaf += self.calcualte_avg(child)
        if tree.is_leaf():
            tree.td_state['output_h'] = tree.td_state['h']
            return [tree.td_state['output_h']]
        else:
            # alert the dim is hard code
            tree.td_state['output_h'] = torch.mean(torch.stack(cover_leaf, dim=0), dim=0)
            return cover_leaf

    def generate_pred_layer(self, input_size, softmax_in_dim, num_labels):
        if self.need_pred_dense:
            self.dense_softmax = nn.Linear(input_size, softmax_in_dim)
            self.pred_layer = nn.Linear(softmax_in_dim, num_labels)
        else:
            self.pred_layer = nn.Linear(input_size, num_labels)

    def single_h_pred(self, hidden, avg_h):
        if self.dense_softmax is not None:
            softmax_in = F.relu(self.dense_softmax(hidden))
            pred = self.pred_layer(softmax_in)
        else:
            pred = self.pred_layer(hidden)
        return pred

    def avg_h_pred(self, hidden, avg_h):
        # based on https://www.transacl.org/ojs/index.php/tacl/article/view/925
        hidden = torch.cat([hidden, avg_h], dim=0)
        softmax_in = F.relu(self.dense_softmax(hidden))
        return self.pred_layer(softmax_in)

    def td_avg_pred(self, hidden, avg_h):
        # based on https://www.transacl.org/ojs/index.php/tacl/article/view/925
        if self.dense_softmax is not None:
            softmax_in = F.relu(self.dense_softmax(hidden))
            pred = self.pred_layer(softmax_in)
        else:
            pred = self.pred_layer(hidden)
        return pred

    def calcualte_emission_score(self, tree, avg_h):
        if self.only_bu:
            h = tree.bu_state['h']
        else:
            h = torch.cat([tree.bu_state['h'], tree.td_state['h']], dim=0)
        if self.pred_mode == 'td_avg_h' and not self.only_bu:
            h = torch.cat([h, tree.td_state['output_h']], dim=0)
        if self.bert_dim != 0:
            h = torch.cat([h, tree.bert_phase], dim=0)
        emission_score = self.get_emission_score(h, avg_h)
        tree.crf_cache = {"emission_score": emission_score}
        return emission_score

    def forward(self, tree, avg_h):

        children_score = []
        for child in tree.children:
            children_score.append(self.forward(child, avg_h))
        emission_score = self.calcualte_emission_score(tree, avg_h)
        if tree.is_leaf():
            return emission_score
        else:
            assert len(children_score) == 2

            left_child_score = children_score[0].reshape(1, self.num_labels, 1)
            right_child_score = children_score[1].reshape(1, 1, self.num_labels)
            emission_score = emission_score.reshape([self.num_labels, 1, 1])
            inside_score = emission_score + right_child_score + left_child_score + self.trans_matrix
            inside_score = logsumexp(inside_score.reshape(self.num_labels, -1), dim=1)
            return inside_score

    def golden_score(self, tree):
        golden_score = 0.0
        for child in tree.children:
            golden_score += self.golden_score(child)

        emission_score = tree.crf_cache['emission_score'][tree.label]

        if tree.is_leaf():
            return emission_score
        else:
            return golden_score + emission_score + self.trans_matrix[tree.label,
                                                                     tree.children[0].label,
                                                                     tree.children[1].label]

    def loss(self, tree):
        if self.pred_mode == 'td_avg_h':
            self.calcualte_avg(tree)
            avg_h = None
        elif self.pred_mode == 'avg_h':
            avg_h = self.collect_avg_hidden(tree)
        else:
            avg_h = None

        inside_score = self.forward(tree, avg_h)
        energy = logsumexp(inside_score, dim=0)

        golden_score = self.golden_score(tree)
        return energy - golden_score

    def viterbi_forward(self, tree, avg_h):
        for child in tree.children:
            self.viterbi_forward(child, avg_h)
        emission_score = self.calcualte_emission_score(tree, avg_h)
        if tree.is_leaf():
            tree.crf_cache['max_score'] = emission_score
        else:
            left_child_max = tree.get_child(0).crf_cache['max_score'].reshape(1, self.num_labels, 1)
            right_child_max = tree.get_child(1).crf_cache['max_score'].reshape(1, 1, self.num_labels)

            score = emission_score.reshape([self.num_labels, 1, 1]) + left_child_max + \
                    right_child_max + self.trans_matrix
            score = score.reshape([self.num_labels, -1])
            max_score, max_idx = torch.max(score, dim=1)
            tree.crf_cache['max_score'] = max_score
            max_left_idx = max_idx // self.num_labels
            max_right_idx = torch.fmod(max_idx, self.num_labels)
            tree.crf_cache['left_idx'] = max_left_idx
            tree.crf_cache['right_idx'] = max_right_idx

    def viterbi_backward(self, tree):
        if tree.is_leaf():
            return

        idx = tree.crf_cache['max_label']
        tree.get_child(0).crf_cache['max_label'] = tree.crf_cache['left_idx'][idx]
        tree.get_child(1).crf_cache['max_label'] = tree.crf_cache['right_idx'][idx]

        for child in tree.children:
            self.viterbi_backward(child)

    def collect_pred(self, tree, holder):
        for child in tree.children:
            self.collect_pred(child, holder)
        holder.append(tree.crf_cache['max_label'])
        return holder

    def predict(self, tree):
        if self.pred_mode == 'td_avg_h':
            self.calcualte_avg(tree)
            avg_h = None
        elif self.pred_mode == 'avg_h':
            avg_h = self.collect_avg_hidden(tree)
        else:
            avg_h = None
        self.viterbi_forward(tree, avg_h)
        max_scores = tree.crf_cache['max_score']
        max_score, idx = torch.max(max_scores, 0)
        tree.crf_cache['max_label'] = idx
        self.viterbi_backward(tree)
        pred = self.collect_pred(tree, [])
        return pred
