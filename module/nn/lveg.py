import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
from module.util import logsumexp, detect_nan


class BinaryTreeLVeG(nn.Module):

    def __init__(self, input_dim, num_label, comp, gaussian_dim, attention=True, only_bu=True,
                 pred_mode=None, softmax_in_dim=64, need_pred_dense=False, bert_dim=0, trans_mat=None):
        # alert may we can use attention as mixing weight
        # TODO let weight as attentnion
        # alert we only use the hidden state to predict the gaussian parameter, maybe we can change here
        # alert why we need transition matrix here??
        super(BinaryTreeLVeG, self).__init__()
        self.input_dim = input_dim
        self.num_label = num_label
        self.gaussian_dim = gaussian_dim
        self.attention = attention
        self.only_bu = attention or only_bu
        self.pred_mode = pred_mode
        self.softmax_in_dim = softmax_in_dim
        self.comp = comp
        self.dense_state = None
        self.need_pred_dense = need_pred_dense
        self.state_weight_layer = None
        self.state_mu_layer = None
        self.state_var_layer = None
        self.bert_dim = bert_dim

        self.trans_weight = Parameter(torch.Tensor(num_label, num_label, num_label, comp))
        self.trans_mu_p = Parameter(torch.Tensor(num_label, num_label, num_label, comp, gaussian_dim))
        self.trans_mu_lc = Parameter(torch.Tensor(num_label, num_label, num_label, comp, gaussian_dim))
        self.trans_mu_rc = Parameter(torch.Tensor(num_label, num_label, num_label, comp, gaussian_dim))
        self.trans_var_p = Parameter(torch.Tensor(num_label, num_label, num_label, comp, gaussian_dim))
        self.trans_var_lc = Parameter(torch.Tensor(num_label, num_label, num_label, comp, gaussian_dim))
        self.trans_var_rc = Parameter(torch.Tensor(num_label, num_label, num_label, comp, gaussian_dim))

        self.trans_root_mu = Parameter(torch.Tensor(num_label, comp, gaussian_dim))
        self.trans_root_var = Parameter(torch.Tensor(num_label, comp, gaussian_dim))
        # is this useful?
        self.trans_root_weight = Parameter(torch.Tensor(num_label, comp))
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
            self.state_weight_layer = nn.Linear(self.softmax_in_dim, self.num_label * self.comp)
            self.state_mu_layer = nn.Linear(self.softmax_in_dim, self.num_label * self.comp * self.gaussian_dim)
            self.state_var_layer = nn.Linear(self.softmax_in_dim, self.num_label * self.comp * self.gaussian_dim)
        else:
            self.state_weight_layer = nn.Linear(input_dim, self.num_label * self.comp)
            self.state_mu_layer = nn.Linear(input_dim, self.num_label * self.comp * self.gaussian_dim)
            self.state_var_layer = nn.Linear(input_dim, self.num_label * self.comp * self.gaussian_dim)

    def reset_parameter(self, trans_mat):
        if trans_mat is None:
            nn.init.xavier_normal_(self.trans_weight)
        else:
            if self.comp != 1:
                base = torch.tensor(trans_mat).float().unsqueeze(-1)
                nn.init.xavier_normal_(self.trans_weight)
                with torch.no_grad():
                    self.trans_weight = Parameter(self.trans_weight + base)
            else:
                self.trans_weight = Parameter(torch.tensor(trans_mat).float()).unsqueeze(-1)
        nn.init.xavier_normal_(self.trans_mu_p)
        nn.init.xavier_normal_(self.trans_mu_lc)
        nn.init.xavier_normal_(self.trans_mu_rc)
        nn.init.constant_(self.trans_var_p, 0.0)
        nn.init.constant_(self.trans_var_lc, 0.0)
        nn.init.constant_(self.trans_var_rc, 0.0)
        nn.init.xavier_normal_(self.trans_root_weight)
        nn.init.xavier_normal_(self.trans_root_mu)
        nn.init.constant_(self.trans_root_var, 0.0)

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
        state_weight = self.state_weight_layer(state_in).reshape(self.num_label, self.comp)
        state_mu = self.state_mu_layer(state_in).reshape(self.num_label, self.comp, self.gaussian_dim)
        state_var = self.state_var_layer(state_in).reshape(self.num_label, self.comp, self.gaussian_dim)
        return state_weight, state_mu, state_var

    def avg_h_pred(self, hidden, avg_h):
        # based on https://www.transacl.org/ojs/index.php/tacl/article/view/925
        hidden = torch.cat([hidden, avg_h], dim=0)
        if self.dense_state is None:
            state_in = hidden
        else:
            state_in = F.relu(self.dense_state(hidden))
        state_weight = self.state_weight_layer(state_in).reshape(self.num_label, self.comp)
        state_mu = self.state_mu_layer(state_in).reshape(self.num_label, self.comp, self.gaussian_dim)
        state_var = self.state_var_layer(state_in).reshape(self.num_label, self.comp, self.gaussian_dim)
        return state_weight, state_mu, state_var

    def td_avg_pred(self, hidden, avg_h):
        # based on https://www.transacl.org/ojs/index.php/tacl/article/view/925
        if self.dense_state is None:
            state_in = hidden
        else:
            state_in = F.relu(self.dense_state(hidden))
        state_weight = self.state_weight_layer(state_in).reshape(self.num_label, self.comp)
        state_mu = self.state_mu_layer(state_in).reshape(self.num_label, self.comp, self.gaussian_dim)
        state_var = self.state_var_layer(state_in).reshape(self.num_label, self.comp, self.gaussian_dim)
        return state_weight, state_mu, state_var

    def calcualte_emission_gm(self, tree, avg_h):
        #
        # if self.debug:
        #     if tree.is_leaf():
        #         tree.lveg_cache["in_weight"] = tree.lveg_cache["state_weight"]
        #         tree.lveg_cache["in_mu"] = tree.lveg_cache["state_mu"]
        #         tree.lveg_cache["in_var"] = tree.lveg_cache["state_var"]
        #     return tree.lveg_cache["state_weight"], tree.lveg_cache["state_mu"], tree.lveg_cache["state_var"]

        if self.only_bu:
            h = tree.bu_state['h']
        else:
            h = torch.cat([tree.bu_state['h'], tree.td_state['h']], dim=0)
        if self.pred_mode == 'td_avg_h' and not self.only_bu:
            h = torch.cat([h, tree.td_state['output_h']], dim=0)
        if self.bert_dim != 0:
            h = torch.cat([h, tree.bert_phase], dim=0)
        state_w, state_mu, state_var = self.get_emission_gm(h, avg_h)

        state_var = torch.clamp(state_var, min=-10.0, max=10.0)

        tree.lveg_cache["state_weight"] = state_w
        tree.lveg_cache["state_mu"] = state_mu
        tree.lveg_cache["state_var"] = state_var
        if tree.is_leaf():
            tree.lveg_cache["in_weight"] = tree.lveg_cache["state_weight"]
            tree.lveg_cache["in_mu"] = tree.lveg_cache["state_mu"]
            tree.lveg_cache["in_var"] = tree.lveg_cache["state_var"]
        return state_w, state_mu, state_var

    def gaussian_multi(self, n1_mu, n1_var, n2_mu, n2_var):
        n1_var_square = torch.exp(2.0 * n1_var)
        n2_var_square = torch.exp(2.0 * n2_var)
        var_square_add = n1_var_square + n2_var_square
        var_log_square_add = torch.log(var_square_add)

        scale = -0.5 * (math.log(math.pi * 2) + var_log_square_add + torch.pow(n1_mu - n2_mu, 2.0) / var_square_add)

        mu = (n1_mu * n2_var_square + n2_mu * n1_var_square) / var_square_add

        var = n1_var + n2_var - 0.5 * var_log_square_add

        scale = torch.sum(scale, dim=-1)
        return scale, mu, var

    def forward(self, tree, avg_h):

        # calcualte inside score of a node
        children_inside_score = []
        for child in tree.children:
            children_inside_score.append(self.forward(child, avg_h))

        state_weight, state_mu, state_var = self.calcualte_emission_gm(tree, avg_h)

        if tree.is_leaf():
            return state_weight, state_mu, state_var
        else:
            assert len(children_inside_score) == 2
            # inside score shape
            # [num_label, comp, gaussian_dim]

            left_in_weight, left_in_mu, left_in_var = tree.children[0].lveg_cache['in_weight'], \
                                                      tree.children[0].lveg_cache['in_mu'], \
                                                      tree.children[0].lveg_cache['in_var']

            left_child_part = self.gaussian_mutlisum(left_in_weight, left_in_mu, left_in_var,
                                                     None, self.trans_mu_lc, self.trans_var_lc,
                                                     [1, self.num_label, 1, 1, -1], 4, 4)

            right_in_weight, right_in_mu, right_in_var = tree.children[1].lveg_cache['in_weight'], \
                                                         tree.children[1].lveg_cache['in_mu'], \
                                                         tree.children[1].lveg_cache['in_var']

            right_child_part = self.gaussian_mutlisum(right_in_weight, right_in_mu, right_in_var,
                                                      None, self.trans_mu_rc, self.trans_var_rc,
                                                      [1, 1, self.num_label, 1, -1], 4, 4)

            child_scores = left_child_part + right_child_part + self.trans_weight

            p_score, p_mu, p_var = self.inout_multi(state_weight, state_mu, state_var, child_scores,
                                                    self.trans_mu_p, self.trans_var_p,
                                                    [self.num_label, 1, 1, -1, 1], 3)

            p_score = p_score.reshape(self.num_label, -1)
            p_mu = p_mu.reshape(self.num_label, -1, self.gaussian_dim)
            p_var = p_var.reshape(self.num_label, -1, self.gaussian_dim)
            tree.lveg_cache['in_weight'] = p_score
            tree.lveg_cache['in_mu'] = p_mu
            tree.lveg_cache['in_var'] = p_var

            return p_score, p_mu, p_var

    def golden_score(self, tree):
        children_in_scores = []
        for child in tree.children:
            children_in_scores.append(self.golden_score(child))

        if tree.is_leaf():
            return tree.lveg_cache['state_weight'][tree.label], \
                   tree.lveg_cache['state_mu'][tree.label], \
                   tree.lveg_cache['state_var'][tree.label]

        left_label = tree.get_child(0).label
        right_label = tree.get_child(1).label
        p_label = tree.label

        trans_weight = self.trans_weight[p_label, left_label, right_label]
        trans_mu_p = self.trans_mu_p[p_label, left_label, right_label]
        trans_mu_lc = self.trans_mu_lc[p_label, left_label, right_label]
        trans_mu_rc = self.trans_mu_rc[p_label, left_label, right_label]
        trans_var_p = self.trans_var_p[p_label, left_label, right_label]
        trans_var_lc = self.trans_var_lc[p_label, left_label, right_label]
        trans_var_rc = self.trans_var_rc[p_label, left_label, right_label]

        left_in_w, left_in_mu, left_in_var = children_in_scores[0]
        right_in_w, right_in_mu, right_in_var = children_in_scores[1]
        p_state_w, p_state_mu, p_state_var = tree.lveg_cache['state_weight'][tree.label], \
                                             tree.lveg_cache['state_mu'][tree.label], \
                                             tree.lveg_cache['state_var'][tree.label]
        # fixme
        left_score, _, _ = self.gaussian_multi(left_in_mu.unsqueeze(1), left_in_var.unsqueeze(1),
                                               trans_mu_lc.unsqueeze(0), trans_var_lc.unsqueeze(0))
        right_score, _, _ = self.gaussian_multi(right_in_mu.unsqueeze(1), right_in_var.unsqueeze(1),
                                                trans_mu_rc.unsqueeze(0), trans_var_rc.unsqueeze(0))
        # left score [comp, comp]
        left_score = logsumexp(left_score + left_in_w.unsqueeze(1), dim=0)
        right_score = logsumexp(right_score + right_in_w.unsqueeze(1), dim=0)

        trans_weight = trans_weight + left_score + right_score

        p_score, p_mu, p_var = self.gaussian_multi(p_state_mu.unsqueeze(1), p_state_var.unsqueeze(1),
                                                   trans_mu_p.unsqueeze(0), trans_var_p.unsqueeze(0))
        p_score = p_score + trans_weight.unsqueeze(0) + p_state_w.unsqueeze(1)

        return p_score.reshape(-1), p_mu.reshape(-1, self.gaussian_dim), p_var.reshape(-1, self.gaussian_dim)

    def loss(self, tree):
        # if self.debug:
        #     avg_h = None
        # else:

        if self.pred_mode == 'td_avg_h':
            self.calcualte_avg(tree)
            avg_h = None
        elif self.pred_mode == 'avg_h':
            avg_h = self.collect_avg_hidden(tree)
        else:
            avg_h = None
        inside_score, inside_mu, inside_var = self.forward(tree, avg_h)

        energy, _, _ = self.gaussian_multi(inside_mu.unsqueeze(1), inside_var.unsqueeze(1),
                                           self.trans_root_mu.unsqueeze(2), self.trans_root_var.unsqueeze(2))
        # energy shape [num_label, comp, inside_comp]
        energy = logsumexp(energy + inside_score.unsqueeze(1) + self.trans_root_weight.unsqueeze(2))

        golden_in_score, golden_in_mu, golden_in_var = self.golden_score(tree)

        golden_score, _, _ = self.gaussian_multi(golden_in_mu.unsqueeze(1), golden_in_var.unsqueeze(1),
                                                 self.trans_root_mu[tree.label].unsqueeze(0),
                                                 self.trans_root_var[tree.label].unsqueeze(0))
        # shape [inside_comp, comp]
        golden_score = logsumexp(
            golden_score + golden_in_score.unsqueeze(1) + self.trans_root_weight[tree.label].unsqueeze(0))
        loss = energy - golden_score
        return loss

    def outside(self, tree):
        num_label = self.num_label
        # here should we got a outside vector?
        # TODO difference use the bi-tree lstm

        if tree.is_leaf():
            return

        # left part
        p_out_weight, p_out_mu, p_out_var = tree.lveg_cache['out_weight'], \
                                            tree.lveg_cache['out_mu'], \
                                            tree.lveg_cache['out_var']

        rc_in_weight, rc_in_mu, rc_in_var = tree.children[1].lveg_cache['in_weight'], \
                                            tree.children[1].lveg_cache['in_mu'], \
                                            tree.children[1].lveg_cache['in_var']

        lc_in_weight, lc_in_mu, lc_in_var = tree.children[0].lveg_cache['in_weight'], \
                                            tree.children[0].lveg_cache['in_mu'], \
                                            tree.children[0].lveg_cache['in_var']

        right_in_score = self.gaussian_mutlisum(rc_in_weight, rc_in_mu, rc_in_var, None, self.trans_mu_rc,
                                                self.trans_var_rc, [1, 1, num_label, 1, -1], 4, 4)

        p_out_score = self.gaussian_mutlisum(p_out_weight, p_out_mu, p_out_var, None, self.trans_mu_p,
                                             self.trans_var_p, [num_label, 1, 1, 1, -1], 4, 4)

        lc_state_weight, lc_state_mu, lc_state_var = tree.children[0].lveg_cache['state_weight'], \
                                                     tree.children[0].lveg_cache['state_mu'], \
                                                     tree.children[0].lveg_cache['state_var']

        child_score = right_in_score + p_out_score + self.trans_weight
        lc_out_score, lc_out_mu, lc_out_var = self.inout_multi(lc_state_weight, lc_state_mu, lc_state_var,
                                                               child_score, self.trans_mu_lc, self.trans_var_lc,
                                                               [1, num_label, 1, 1, -1], 4)

        # shape [num_label, num_label, num_label, comp, in_comp]
        lc_out_score = lc_out_score.transpose(0, 1).reshape(self.num_label, -1)
        lc_out_mu = lc_out_mu.transpose(0, 1).reshape(self.num_label, -1, self.gaussian_dim)
        lc_out_var = lc_out_var.transpose(0, 1).reshape(self.num_label, -1, self.gaussian_dim)

        tree.children[0].lveg_cache['out_weight'] = lc_out_score
        tree.children[0].lveg_cache['out_mu'] = lc_out_mu
        tree.children[0].lveg_cache['out_var'] = lc_out_var

        # right part
        left_in_score = self.gaussian_mutlisum(lc_in_weight, lc_in_mu, lc_in_var, None, self.trans_mu_lc,
                                               self.trans_var_lc, [1, num_label, 1, 1, -1], 4, 4)

        rc_state_weight, rc_state_mu, rc_state_var = tree.children[1].lveg_cache['state_weight'], \
                                                     tree.children[1].lveg_cache['state_mu'], \
                                                     tree.children[1].lveg_cache['state_var']
        child_score = left_in_score + p_out_score + self.trans_weight
        rc_out_score, rc_out_mu, rc_out_var = self.inout_multi(rc_state_weight, rc_state_mu, rc_state_var,
                                                               child_score, self.trans_mu_rc, self.trans_var_rc,
                                                               [1, 1, num_label, 1, -1], 4)
        # rc_out_score shape [num_label, num_label, num_label, num_label, comp, state_comp]

        rc_out_score = rc_out_score.transpose(0, 2).reshape(self.num_label, -1)
        rc_out_mu = rc_out_mu.transpose(0, 2).reshape(self.num_label, -1)
        rc_out_var = rc_out_var.transpose(0, 2).reshape(self.num_label, -1)

        tree.children[1].lveg_cache['out_weight'] = rc_out_score
        tree.children[1].lveg_cache['out_mu'] = rc_out_mu
        tree.children[1].lveg_cache['out_var'] = rc_out_var
        for child in tree.children:
            self.outside(child)

    def inout_multi(self, s_weight, s_mu, s_var,
                    t_weight, t_mu, t_var, s_shape, t_dim):
        # inside or outside score multiply with transition rule
        g_shape = s_shape + [self.gaussian_dim]
        multi_score, multi_mu, multi_var = self.gaussian_multi(s_mu.reshape(g_shape),
                                                               s_var.reshape(g_shape),
                                                               t_mu.unsqueeze(t_dim),
                                                               t_var.unsqueeze(t_dim))
        if t_weight is not None:
            multi_score_tmp = multi_score + s_weight.reshape(s_shape) + t_weight.unsqueeze(t_dim)
        else:
            multi_score_tmp = multi_score + s_weight.reshape(s_shape)
        return multi_score_tmp, multi_mu, multi_var

    def gaussian_mutlisum(self, s_weight, s_mu, s_var, t_weight,
                          t_mu, t_var, s_shape, t_dim, sum_dim):
        # inside or outside score multiply with transition rule then sum
        # here we just need score
        multi_score, _, _ = self.inout_multi(s_weight, s_mu, s_var, t_weight, t_mu, t_var, s_shape, t_dim)
        summed_score = logsumexp(multi_score, dim=sum_dim)

        return summed_score

    def maxrule_parsing(self, tree):
        num_label = self.num_label
        for child in tree.children:
            self.maxrule_parsing(child)

        if tree.is_leaf():
            tree.lveg_cache['max_score'] = tree.lveg_cache['state_weight']
        else:
            # calcualte expected count

            # get children's inside score and parent's outside score
            lc_in_weight, lc_in_mu, lc_in_var = tree.children[0].lveg_cache['in_weight'], \
                                                tree.children[0].lveg_cache['in_mu'], \
                                                tree.children[0].lveg_cache['in_var']

            rc_in_weight, rc_in_mu, rc_in_var = tree.children[1].lveg_cache['in_weight'], \
                                                tree.children[1].lveg_cache['in_mu'], \
                                                tree.children[1].lveg_cache['in_var']

            p_out_weight, p_out_mu, p_out_var = tree.lveg_cache['out_weight'], \
                                                tree.lveg_cache['out_mu'], \
                                                tree.lveg_cache['out_var']

            left_score = self.gaussian_mutlisum(lc_in_weight, lc_in_mu, lc_in_var, None, self.trans_mu_lc,
                                                self.trans_var_lc, [1, num_label, 1, 1, -1], 4, 4)

            right_score = self.gaussian_mutlisum(rc_in_weight, rc_in_mu, rc_in_var, None, self.trans_mu_rc,
                                                 self.trans_var_rc, [1, 1, num_label, 1, -1], 4, 4)

            p_score = self.gaussian_mutlisum(p_out_weight, p_out_mu, p_out_var, None, self.trans_mu_p,
                                             self.trans_var_p, [num_label, 1, 1, 1, -1], 4, 4)

            expected_count = logsumexp(left_score + right_score + p_score + self.trans_weight, dim=3)

            expected_count = expected_count - logsumexp(expected_count.reshape(num_label, -1), dim=1).reshape(num_label,
                                                                                                              1, 1)
            max_label = torch.argmax(expected_count.reshape(num_label, -1), dim=1).cpu().numpy().astype(int)
            tree.lveg_cache['expected_count'] = expected_count
            tree.lveg_cache['max_labels'] = max_label

    def get_max_tree(self, tree):
        if tree.is_leaf():
            return

        label = tree.lveg_cache['max_labels'][tree.lveg_cache['max_label']]
        left_label = label // self.num_label
        right_label = label % self.num_label

        tree.children[0].lveg_cache['max_label'] = left_label
        tree.children[1].lveg_cache['max_label'] = right_label

        for child in tree.children:
            self.get_max_tree(child)

    def collect_pred(self, tree, holder):
        for child in tree.children:
            self.collect_pred(child, holder)
        holder.append(tree.lveg_cache['max_label'])
        return holder

    def predict(self, tree):
        # alert should we calculate the outside score with the state score?

        tree.lveg_cache['out_weight'] = self.trans_root_weight
        tree.lveg_cache['out_mu'] = self.trans_root_mu
        tree.lveg_cache['out_var'] = self.trans_root_var
        # if self.debug:
        #     avg_h = None
        # else:
        if self.pred_mode == 'td_avg_h':
            self.calcualte_avg(tree)
            avg_h = None
        elif self.pred_mode == 'avg_h':
            avg_h = self.collect_avg_hidden(tree)
        else:
            avg_h = None
        inside_score, inside_mu, inside_var = self.forward(tree, avg_h)
        self.outside(tree)

        self.maxrule_parsing(tree)

        total_tree_score = self.gaussian_mutlisum(inside_score, inside_mu, inside_var, self.trans_root_weight,
                                                  self.trans_root_mu, self.trans_root_var, [self.num_label, 1, -1],
                                                  2, 1)
        total_tree_score = logsumexp(total_tree_score, dim=1)
        max_label = torch.argmax(total_tree_score)
        tree.lveg_cache['max_label'] = max_label.cpu().item()
        self.get_max_tree(tree)
        pred = self.collect_pred(tree, [])
        return pred
