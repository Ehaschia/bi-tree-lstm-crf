import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from module.util import logsumexp


class TreeCRF(nn.Module):
    '''
    Tree CRF layer.
    '''

    def __init__(self, num_labels: int):
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
        self.num_labels = num_labels
        self.trans_matrix = Parameter(torch.Tensor(self.num_labels, self.num_labels))
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_normal_(self.trans_matrix)

    def forward(self, tree):
        # just calculate the inside score
        children_score = []
        for child in tree.children:
            children_score.append(self.forward(child))

        emission_score = tree.crf_cache['be_hidden']

        if tree.is_leaf():
            return emission_score
        else:
            assert len(children_score) == 2
            children_score[0] = (children_score[0].unsqueeze(0) + self.trans_matrix).unsqueeze(2)
            children_score[1] = (children_score[1].unsqueeze(0) + self.trans_matrix).unsqueeze(1)
            emission_score = emission_score.reshape([self.num_labels, 1, 1])
            for child_score in children_score:
                emission_score = child_score + emission_score

        inside_score = logsumexp(emission_score.reshape(self.num_labels, -1), dim=1)

        return inside_score

    def golden_score(self, tree):
        golden_score = 0.0
        for child in tree.children:
            golden_score += self.golden_score(child)

        if tree.parent is not None:
            golden_score += (tree.crf_cache['be_hidden'].unsqueeze(0) + self.trans_matrix)[
                tree.parent.label, tree.label]
        else:
            golden_score += tree.crf_cache['be_hidden'][tree.label]
        return golden_score

    def loss(self, tree):
        inside_score = self.forward(tree)
        energy = logsumexp(inside_score, dim=0)

        golden_score = self.golden_score(tree)
        return energy - golden_score

    def viterbi_forward(self, tree):
        for child in tree.children:
            self.viterbi_forward(child)
        if tree.is_leaf():
            emission_score = tree.crf_cache['be_hidden']
            tree.crf_cache['max_score'] = emission_score
        else:
            emission_score = tree.crf_cache['be_hidden']
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
        self.viterbi_forward(tree)
        max_scores = tree.crf_cache['max_score']
        max_score, idx = torch.max(max_scores, 0)
        tree.crf_cache['max_label'] = idx
        self.viterbi_backward(tree)
        pred = self.collect_pred(tree, [])
        return pred


class BinaryTreeCRF(nn.Module):

    def __init__(self, num_labels):

        """

        Args:
            num_labels: int
                the number of labels of the crf layer
            **kwargs:
        """

        super(BinaryTreeCRF, self).__init__()
        self.num_labels = num_labels
        self.trans_matrix = Parameter(torch.Tensor(self.num_labels, self.num_labels, self.num_labels))
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_normal_(self.trans_matrix)

    def forward(self, tree):

        children_score = []
        for child in tree.children:
            children_score.append(self.forward(child))
        emission_score = tree.crf_cache['crf_hidden']
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

        emission_score = tree.crf_cache['crf_hidden'][tree.label]

        if tree.is_leaf():
            return emission_score
        else:
            return golden_score + emission_score + self.trans_matrix[tree.label,
                                                                     tree.children[0].label,
                                                                     tree.children[1].label]

    def loss(self, tree):
        inside_score = self.forward(tree)
        energy = logsumexp(inside_score, dim=0)

        golden_score = self.golden_score(tree)
        return energy - golden_score

    def viterbi_forward(self, tree):
        for child in tree.children:
            self.viterbi_forward(child)
        emission_score = tree.crf_cache['crf_hidden']
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
        self.viterbi_forward(tree)
        max_scores = tree.crf_cache['max_score']
        max_score, idx = torch.max(max_scores, 0)
        tree.crf_cache['max_label'] = idx
        self.viterbi_backward(tree)
        pred = self.collect_pred(tree, [])
        return pred
