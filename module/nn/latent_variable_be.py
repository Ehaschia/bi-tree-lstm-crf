import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
from module.util import logsumexp, detect_nan


class BinaryTreeLatentVariable(nn.Module):

    def __init__(self, num_label, comp, trans_mat=None):
        super(BinaryTreeLatentVariable, self).__init__()
        self.num_label = num_label
        self.comp = comp
        self.trans_matrix = Parameter(torch.Tensor(self.num_label, self.num_label, self.num_label,
                                                   comp, comp, comp))
        self.reset_parameter(trans_mat)

    def reset_parameter(self, trans_mat):
        if trans_mat is None:
            nn.init.xavier_normal_(self.trans_matrix)
        else:
            if self.comp != 1:
                base = torch.tensor(trans_mat).float().reshape(self.num_label, self.num_label, self.num_label,
                                                               self.comp, self.comp, self.comp)
                nn.init.xavier_normal_(self.trans_matrix)
                with torch.no_grad():
                    self.trans_matrix = Parameter(self.trans_matrix + base)
            else:
                self.trans_matrix = Parameter(torch.tensor(trans_mat).float()).unsqueeze(-1)

    def forward(self, tree):

        # calcualte inside score of a node
        children_inside_score = []
        for child in tree.children:
            children_inside_score.append(self.forward(child))

        state_weight = tree.crf_cache['state_weight']

        if tree.is_leaf():
            tree.crf_cache['in_weight'] = state_weight
            return state_weight
        else:
            assert len(children_inside_score) == 2
            # inside score shape
            # [num_label, comp, gaussian_dim]

            left_child_part = tree.children[0].crf_cache['in_weight'].reshape(1, self.num_label, 1, 1, self.comp, 1)

            right_child_part = tree.children[1].crf_cache['in_weight'].reshape(1, 1, self.num_label, 1, 1, self.comp)

            p_part = tree.crf_cache['state_weight'].reshape(self.num_label, 1, 1, self.comp, 1, 1)
            inside_score = p_part + left_child_part + right_child_part + self.trans_matrix
            inside_score = inside_score.permute(0, 3, 1, 2, 4, 5).reshape(self.num_label, self.comp, -1)
            inside_score = logsumexp(inside_score, dim=2)
            # shape [num_label, comp]
            tree.crf_cache['in_weight'] = inside_score
            return inside_score

    def golden_score(self, tree):
        children_in_scores = []
        for child in tree.children:
            children_in_scores.append(self.golden_score(child))

        if tree.is_leaf():
            # shape [comp]
            return tree.crf_cache['state_weight'][tree.label]

        assert len(children_in_scores) == 2

        p_label = tree.label
        l_label = tree.children[0].label
        r_label = tree.children[1].label

        left_child_part = children_in_scores[0].reshape(1, self.comp, 1)
        right_child_part = children_in_scores[1].reshape(1, 1, self.comp)
        parent_part = tree.crf_cache['state_weight'][p_label].reshape(self.comp, 1, 1)
        golden_score = left_child_part + right_child_part + parent_part + \
                       self.trans_matrix[p_label, l_label, r_label]

        golden_score = logsumexp(golden_score.reshape(self.comp, -1), dim=1)

        return golden_score

    def loss(self, tree):

        inside_score = self.forward(tree)

        energy = logsumexp(inside_score)

        golden_in_score = self.golden_score(tree)

        golden_score = logsumexp(golden_in_score)
        loss = energy - golden_score
        return loss

    def outside(self, tree):
        if tree.is_leaf():
            return

        # root part
        if tree.parent is None:
            tree.crf_cache['out_weight'] = tree.crf_cache['in_weight'].new(self.num_label, self.comp).fill_(0.0)

        # left part
        p_out_score = tree.crf_cache['out_weight'].reshape(self.num_label, 1, 1, self.comp, 1, 1)

        rc_in_score = tree.children[1].crf_cache['in_weight'].reshape(1, 1, self.num_label, 1, 1, self.comp)
        lc_in_score = tree.children[0].crf_cache['in_weight'].reshape(1, self.num_label, 1, 1, self.comp, 1)

        lc_out_score = rc_in_score + p_out_score + self.trans_matrix

        # shape [num_label, num_label, num_label, comp, comp, comp]
        lc_out_score = lc_out_score.permute(1, 4, 0, 2, 3, 5).reshape(self.num_label, self.comp, -1)
        lc_out_score = logsumexp(lc_out_score, dim=2)
        tree.children[0].crf_cache['out_weight'] = lc_out_score

        # right part
        rc_out_score = lc_in_score + p_out_score + self.trans_matrix
        rc_out_score = rc_out_score.permute(2, 5, 0, 1, 3, 4).reshape(self.num_label, self.comp, -1)
        rc_out_score = logsumexp(rc_out_score, dim=2)
        tree.children[1].crf_cache['out_weight'] = rc_out_score

        for child in tree.children:
            self.outside(child)

    def maxrule_parsing(self, tree):
        num_label = self.num_label
        for child in tree.children:
            self.maxrule_parsing(child)

        if tree.is_leaf():
            tree.crf_cache['max_score'] = tree.crf_cache['state_weight']
        else:
            # calcualte expected count

            # get children's inside score and parent's outside score
            lc_in_score = tree.children[0].crf_cache['in_weight'].reshape(1, self.num_label, 1, 1, self.comp, 1)

            rc_in_score = tree.children[1].crf_cache['in_weight'].reshape(1, 1, self.num_label, 1, 1, self.comp)
            p_out_score = tree.crf_cache['out_weight'].reshape(self.num_label, 1, 1, self.comp, 1, 1)

            expected_count_fined = lc_in_score + rc_in_score + p_out_score + self.trans_matrix
            expected_count_fined = expected_count_fined.reshape(self.num_label, self.num_label, self.num_label, -1)
            expected_count = logsumexp(expected_count_fined, dim=3)
            expected_count = expected_count - logsumexp(expected_count.reshape(self.num_label, -1), dim=1).reshape(
                self.num_label, 1, 1)
            max_label = torch.argmax(expected_count.reshape(num_label, -1), dim=1).cpu().numpy().astype(int)
            tree.crf_cache['expected_count'] = expected_count
            tree.crf_cache['max_labels'] = max_label

    def get_max_tree(self, tree):
        if tree.is_leaf():
            return

        label = tree.crf_cache['max_labels'][tree.crf_cache['max_label']]
        left_label = label // self.num_label
        right_label = label % self.num_label

        tree.children[0].crf_cache['max_label'] = left_label
        tree.children[1].crf_cache['max_label'] = right_label

        for child in tree.children:
            self.get_max_tree(child)

    def collect_pred(self, tree, holder):
        for child in tree.children:
            self.collect_pred(child, holder)
        holder.append(tree.crf_cache['max_label'])
        return holder

    def predict(self, tree):
        inside_score = self.forward(tree)
        self.outside(tree)

        self.maxrule_parsing(tree)

        total_tree_score = logsumexp(inside_score, dim=1)
        max_label = torch.argmax(total_tree_score)
        tree.crf_cache['max_label'] = max_label.cpu().item()
        self.get_max_tree(tree)
        pred = self.collect_pred(tree, [])
        return pred
