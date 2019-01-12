import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
from module.util import logsumexp, detect_nan


class BinaryTreeLatentVariable(nn.Module):

    def __init__(self, input_dim, num_label, comp, attention=True, only_bu=True,
                 pred_mode=None, softmax_in_dim=64, need_pred_dense=False, bert_dim=0, trans_mat=None):
        super(BinaryTreeLatentVariable, self).__init__()
        self.input_dim = input_dim
        self.num_label = num_label
        self.attention = attention
        self.only_bu = attention or only_bu
        self.pred_mode = pred_mode
        self.softmax_in_dim = softmax_in_dim
        self.comp = comp
        self.dense_state = None
        self.need_pred_dense = need_pred_dense
        self.state_weight_layer = None
        self.bert_dim = bert_dim

        self.subtype_num = num_label * comp
        self.trans_matrix = Parameter(torch.Tensor(self.num_label, self.num_label, self.num_label,
                                                   comp, comp, comp))

        self.pred_mode = pred_mode
        if pred_mode == 'single_h':
            pred_input_dim = input_dim if self.only_bu else input_dim * 2
            pred_input_dim = pred_input_dim + bert_dim
            self.generate_state_layer(pred_input_dim)

            self.get_emission_gm = self.single_h_pred
        elif pred_mode == 'avg_h':
            pred_input_dim = input_dim * 2 if self.only_bu else input_dim * 4
            pred_input_dim = pred_input_dim + bert_dim
            self.generate_state_layer(pred_input_dim)

            self.get_emission_gm = self.avg_h_pred
        elif pred_mode == 'td_avg_h':
            assert self.only_bu is False
            pred_input_dim = input_dim * 3
            pred_input_dim = pred_input_dim + bert_dim
            self.generate_state_layer(pred_input_dim)
            self.get_emission_gm = self.td_avg_pred
        else:
            raise NotImplementedError("the pred model " + pred_mode + " is not implemented!")

        self.reset_parameter(trans_mat)

    def generate_state_layer(self, input_dim):
        if self.need_pred_dense:
            self.dense_state = nn.Linear(input_dim, self.softmax_in_dim)
            self.state_weight_layer = nn.Linear(self.softmax_in_dim, self.subtype_num)
        else:
            self.state_weight_layer = nn.Linear(input_dim, self.subtype_num)

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
        if self.dense_state is None:
            state_in = hidden
        else:
            state_in = F.relu(self.dense_state(hidden))
        state_weight = self.state_weight_layer(state_in)
        return state_weight

    def avg_h_pred(self, hidden, avg_h):
        # based on https://www.transacl.org/ojs/index.php/tacl/article/view/925
        hidden = torch.cat([hidden, avg_h], dim=0)
        if self.dense_state is None:
            state_in = hidden
        else:
            state_in = F.relu(self.dense_state(hidden))
        state_weight = self.state_weight_layer(state_in)
        return state_weight

    def td_avg_pred(self, hidden, avg_h):
        # based on https://www.transacl.org/ojs/index.php/tacl/article/view/925
        if self.dense_state is None:
            state_in = hidden
        else:
            state_in = F.relu(self.dense_state(hidden))
        state_weight = self.state_weight_layer(state_in)
        return state_weight

    def calcualte_emission_gm(self, tree, avg_h):
        #
        # if self.debug:
        #     if tree.is_leaf():
        #         tree.crf_cache["in_weight"] = tree.crf_cache["state_weight"]
        #         tree.crf_cache["in_mu"] = tree.crf_cache["state_mu"]
        #         tree.crf_cache["in_var"] = tree.crf_cache["state_var"]
        #     return tree.crf_cache["state_weight"], tree.crf_cache["state_mu"], tree.crf_cache["state_var"]

        if self.only_bu:
            h = tree.bu_state['h']
        else:
            h = torch.cat([tree.bu_state['h'], tree.td_state['h']], dim=0)
        if self.pred_mode == 'td_avg_h' and not self.only_bu:
            h = torch.cat([h, tree.td_state['output_h']], dim=0)
        if self.bert_dim != 0:
            h = torch.cat([h, tree.bert_phase], dim=0)
        state_w = self.get_emission_gm(h, avg_h)

        tree.crf_cache["state_weight"] = state_w.reshape(self.num_label, self.comp)
        if tree.is_leaf():
            tree.crf_cache["in_weight"] = tree.crf_cache["state_weight"]
        return state_w

    def forward(self, tree, avg_h):

        # calcualte inside score of a node
        children_inside_score = []
        for child in tree.children:
            children_inside_score.append(self.forward(child, avg_h))

        state_weight = self.calcualte_emission_gm(tree, avg_h)

        if tree.is_leaf():
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

        if self.pred_mode == 'td_avg_h':
            self.calcualte_avg(tree)
            avg_h = None
        elif self.pred_mode == 'avg_h':
            avg_h = self.collect_avg_hidden(tree)
        else:
            avg_h = None
        inside_score = self.forward(tree, avg_h)

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
            expected_count = expected_count - logsumexp(expected_count.reshape(self.num_label, -1), dim=1).reshape(self.num_label, 1, 1)
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
        if self.pred_mode == 'td_avg_h':
            self.calcualte_avg(tree)
            avg_h = None
        elif self.pred_mode == 'avg_h':
            avg_h = self.collect_avg_hidden(tree)
        else:
            avg_h = None
        inside_score = self.forward(tree, avg_h)
        self.outside(tree)

        self.maxrule_parsing(tree)

        total_tree_score = logsumexp(inside_score, dim=1)
        max_label = torch.argmax(total_tree_score)
        tree.crf_cache['max_label'] = max_label.cpu().item()
        self.get_max_tree(tree)
        pred = self.collect_pred(tree, [])
        return pred
