import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter


# alert parameter init
class BinaryTreeLSTMCell(nn.Module):
    # alert debug this cell!
    # implement dropout http://arxiv.org/pdf/1603.05118.pdf
    # Based on https://www.aclweb.org/anthology/P15-1150
    def __init__(self, in_dim, mem_dim, p_tree=0.0):
        super(BinaryTreeLSTMCell, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        self.tree_dropout = nn.Dropout(p_tree)
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.constant_(self.iouh.bias, 0)
        nn.init.constant_(self.ioux.bias, 0)
        nn.init.constant_(self.fx.bias, 0)
        nn.init.constant_(self.fh.bias, 0)

        nn.init.xavier_normal_(self.iouh.weight)
        nn.init.xavier_normal_(self.ioux.weight)
        nn.init.xavier_normal_(self.fx.weight)
        nn.init.xavier_normal_(self.fh.weight)

    def forward(self, l, r, inputs=None, dim=0):
        l_h, l_c = l['h'].unsqueeze(0), l['c'].unsqueeze(0)
        r_h, r_c = r['h'].unsqueeze(0), r['c'].unsqueeze(0)
        if inputs is None:
            inputs = l_h.detach().new(1, self.in_dim).fill_(0.).requires_grad_()
        child_h = torch.cat([l_h, r_h], dim=dim)
        child_c = torch.cat([l_c, r_c], dim=dim)
        child_h_sum = torch.sum(child_h, dim=dim, keepdim=True)

        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = F.sigmoid(i), F.sigmoid(o), self.tree_dropout(F.tanh(u))

        f = F.sigmoid(
            self.fh(child_h) +
            self.fx(inputs).repeat(len(child_h), 1)
        )
        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, F.tanh(c))
        return {'h': h.squeeze(0), 'c': c.squeeze(0)}


class TreeLSTMCell_flod(nn.Module):
    # copy from tensorfolw_flod
    # implement dropout http://arxiv.org/pdf/1603.05118.pdf
    def __init__(self, in_dim, mem_dim, p_tree=0.0):
        super(TreeLSTMCell_flod, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.ifoux = nn.Linear(in_dim + 2 * mem_dim, 5 * mem_dim)
        self.tree_dropout = nn.Dropout(p_tree)
        self._forget_bias = Parameter(torch.Tensor(mem_dim))
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.constant_(self._forget_bias, 0.0)

    def forward(self, l, r, inputs=None, dim=0):
        l_h, l_c = l['h'].unsqueeze(0), l['c'].unsqueeze(0)
        r_h, r_c = r['h'].unsqueeze(0), r['c'].unsqueeze(0)
        if inputs is None:
            inputs = l_h.detach().new(1, self.in_dim).fill_(0.).requires_grad_()
        h_concat = torch.cat([inputs, l_h, r_h], dim=1)
        concat = self.ifoux(h_concat)
        i, j, l_f, r_f, o = torch.split(concat, concat.size()[1] // 5, dim=1)
        j = self.tree_dropout(F.tanh(j))
        new_c = (l_c * F.sigmoid(l_f + self._forget_bias) +
                 r_c * F.sigmoid(r_f + self._forget_bias) +
                 F.sigmoid(i) * j)
        new_h = F.tanh(new_c) * F.sigmoid(o)
        return {'h': new_h.squeeze(0), 'c': new_c.squeeze(0)}


class BinaryTreeLSTM_v2_p(nn.Module):
    # based on https://arxiv.org/abs/1707.02786
    def __init__(self, hidden_dim, p_tree=0.0):
        super(BinaryTreeLSTM_v2_p, self).__init__()
        self.hidden_dim = hidden_dim
        self.comp_linear = nn.Linear(in_features=2 * hidden_dim,
                                     out_features=5 * hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        init.orthogonal_(self.comp_linear.weight.data)
        init.constant_(self.comp_linear.bias.data, val=0)

    def forward(self, l, r):
        """
        Args:
            l: A (h_l, c_l) tuple, where each value has the size
                (batch_size, max_length, hidden_dim).
            r: A (h_r, c_r) tuple, where each value has the size
                (batch_size, max_length, hidden_dim).
        Returns:
            h, c: The hidden and cell state of the composed parent,
                each of which has the size
                (batch_size, hidden_dim).
        """

        hl, cl = l
        hr, cr = r
        hlr_cat = torch.cat([hl, hr], dim=1)
        treelstm_vector = self.comp_linear(hlr_cat)
        i, fl, fr, u, o = treelstm_vector.chunk(chunks=5, dim=1)
        c = (cl * (fl + 1).sigmoid() + cr * (fr + 1).sigmoid()
             + u.tanh() * i.sigmoid())
        h = o.sigmoid() * c.tanh()
        return h, c


class BinarySLSTM_s(nn.Module):
    # Based on https://arxiv.org/pdf/1503.04881
    # alert single version
    # TODO remove

    def __init__(self, input_dim, out_dim, p_tree=0.0):
        super(BinarySLSTM_s, self).__init__()

        self.input_dim = input_dim
        self.output_dim = out_dim

        self.i = nn.Linear(4 * self.input_dim, self.output_dim)
        self.f_l = nn.Linear(4 * self.input_dim, self.output_dim)
        self.f_r = nn.Linear(4 * self.input_dim, self.output_dim)
        self.x = nn.Linear(2 * self.input_dim, self.output_dim)
        self.o = nn.Linear(3 * self.input_dim, self.output_dim)
        self.activate_func1 = nn.Sigmoid()
        self.activate_func2 = nn.Tanh()

    def node_forward(self, inputs, child_c, child_h):
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)

        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)

        f = F.sigmoid(
            self.fh(child_h) +
            self.fx(inputs).repeat(len(child_h), 1)
        )
        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, F.tanh(c))
        return c, h

    def forward(self, tree, inputs):
        # fixme here is wrong fix it!
        for child in tree.children:
            self.forward(child, inputs)
        if tree.is_leaf():
            child_c = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
            child_h = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
        else:
            child_c, child_h = zip(*map(lambda x: x.state, tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)

        tree.state = self.node_forward(inputs[tree.idx], child_c, child_h)
        return tree.state


class BinarySLSTMCell(nn.Module):
    # Based on https://arxiv.org/pdf/1503.04881
    # implement dropout http://arxiv.org/pdf/1603.05118.pdf
    # alert parallel version

    def __init__(self, input_dim, out_dim, p_tree=0.0):
        super(BinarySLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.output_dim = out_dim

        self.i = nn.Linear(4 * self.input_dim, self.output_dim)
        self.f_l = nn.Linear(4 * self.input_dim, self.output_dim)
        self.f_r = nn.Linear(4 * self.input_dim, self.output_dim)
        self.x = nn.Linear(2 * self.input_dim, self.output_dim)
        self.o = nn.Linear(3 * self.input_dim, self.output_dim)
        self.tree_dropout = nn.Dropout(p_tree)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.i.bias, 0)
        nn.init.constant_(self.f_l.bias, 0)
        nn.init.constant_(self.f_r.bias, 0)
        nn.init.constant_(self.x.bias, 0)
        nn.init.constant_(self.o.bias, 0)

        nn.init.xavier_normal_(self.i.weight)
        nn.init.xavier_normal_(self.f_l.weight)
        nn.init.xavier_normal_(self.f_r.weight)
        nn.init.xavier_normal_(self.x.weight)
        nn.init.xavier_normal_(self.o.weight)

    def forward(self, l, r, inputs=None, dim=1):
        # input size [batch, input_size]
        h_l, c_l = l['h'], l['c']
        h_r, c_r = r['h'], r['c']
        concat_input = torch.cat((h_l, h_r, c_l, c_r), dim=dim)
        concat_h = torch.cat((h_l, h_r), dim=dim)
        # input gate
        i_t = F.sigmoid(self.i(concat_input))
        # forget gate
        f_lt = F.sigmoid(self.f_l(concat_input))
        f_rt = F.sigmoid(self.f_r(concat_input))

        x_t = self.x(concat_h)
        # cell vector
        # here implement dropout
        c_t = f_lt * c_l + f_rt * c_r + i_t * self.tree_dropout(F.tanh(x_t))
        # output gate
        concat_tmp = torch.cat((h_l, h_r, c_t), dim=dim)
        o_t = F.sigmoid(self.o(concat_tmp))
        h_t = o_t * F.tanh(c_t)

        return {'h': h_t, 'c': c_t}


class BUSLSTMCell(nn.Module):
    # based on https://www.transacl.org/ojs/index.php/tacl/article/view/925
    # implement dropout http://arxiv.org/pdf/1603.05118.pdf
    def __init__(self, head_dim, input_dim, output_dim, p_tree=0.0):
        super(BUSLSTMCell, self).__init__()

        self.head_dim = head_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.i = nn.Linear(4 * self.input_dim, self.output_dim)
        self.f_l = nn.Linear(4 * self.input_dim, self.output_dim)
        self.f_r = nn.Linear(4 * self.input_dim, self.output_dim)
        self.x_in = nn.Linear(self.head_dim, 4 * self.output_dim, bias=False)
        self.o = nn.Linear(3 * self.input_dim, self.output_dim)
        self.z = nn.Linear(2 * self.head_dim, self.head_dim)
        self.g = nn.Linear(2 * self.input_dim, self.output_dim)
        self.activate_func1 = nn.Sigmoid()
        self.activate_func2 = nn.Tanh()
        self.tree_dropout = nn.Dropout(p_tree)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.i.bias, 0)
        nn.init.constant_(self.f_l.bias, 0)
        nn.init.constant_(self.f_r.bias, 0)

        nn.init.constant_(self.o.bias, 0)
        nn.init.constant_(self.z.bias, 0)
        nn.init.constant_(self.g.bias, 0)

        nn.init.xavier_normal_(self.i.weight)
        nn.init.xavier_normal_(self.f_l.weight)
        nn.init.xavier_normal_(self.f_r.weight)
        nn.init.xavier_normal_(self.x_in.weight)
        nn.init.xavier_normal_(self.o.weight)
        nn.init.xavier_normal_(self.z.weight)
        nn.init.xavier_normal_(self.g.weight)

    def forward(self, l, r, inputs=None, dim=0):
        x_l, h_l, c_l = l['x'], l['h'], l['c']
        x_r, h_r, c_r = r['x'], r['h'], r['c']
        child_x = torch.cat([x_l, x_r], dim=dim)

        z_t = F.sigmoid(self.z(child_x))
        x_t = z_t * x_l + (1.0 - z_t) * x_r
        x_in_t = self.x_in(x_t)
        x_i, x_f, x_o, x_g = x_in_t.chunk(4, dim=dim)
        concat_input = torch.cat((h_l, h_r, c_l, c_r), dim=dim)
        i_t = F.sigmoid(self.i(concat_input) + x_i)
        f_lt = F.sigmoid(self.f_l(concat_input) + x_f)
        f_rt = F.sigmoid(self.f_r(concat_input) + x_f)
        concat_g = torch.cat([h_l, h_r], dim=dim)
        g_t = self.tree_dropout(F.tanh(x_g + self.g(concat_g)))
        c_t = f_lt * c_l + f_rt * c_r + i_t * g_t
        concat_o = torch.cat([h_l, h_r, c_t], dim=dim)
        o_t = F.sigmoid(x_o + self.o(concat_o))

        h_t = o_t * F.tanh(c_t)

        return {'x': x_t, 'h': h_t, 'c': c_t}


class TDLSTMCell(nn.Module):
    # based on https://www.transacl.org/ojs/index.php/tacl/article/view/925
    # implement dropout http://arxiv.org/pdf/1603.05118.pdf
    def __init__(self, head_dim, input_dim, output_dim, p_tree=0.0):

        super(TDLSTMCell, self).__init__()

        self.head_dim = head_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.i_l = nn.Linear(self.head_dim + self.input_dim * 2, self.output_dim)
        self.i_r = nn.Linear(self.head_dim + self.input_dim * 2, self.output_dim)
        self.f_l = nn.Linear(self.head_dim + self.input_dim * 2, self.output_dim)
        self.f_r = nn.Linear(self.head_dim + self.input_dim * 2, self.output_dim)
        self.o_l = nn.Linear(self.head_dim + self.input_dim * 2, self.output_dim)
        self.o_r = nn.Linear(self.head_dim + self.input_dim * 2, self.output_dim)
        self.g_l = nn.Linear(self.head_dim + self.input_dim, self.output_dim)
        self.g_r = nn.Linear(self.head_dim + self.input_dim, self.output_dim)

        self.activate_func1 = nn.Sigmoid()
        self.activate_func2 = nn.Tanh()
        self.dropout_tree = nn.Dropout(p_tree)
        self.reset_parameters()

    def reset_parameters(self):

        nn.init.constant_(self.i_l.bias, 0)
        nn.init.constant_(self.i_r.bias, 0)
        nn.init.constant_(self.f_l.bias, 0)
        nn.init.constant_(self.f_r.bias, 0)
        nn.init.constant_(self.f_r.bias, 0)
        nn.init.constant_(self.o_l.bias, 0)
        nn.init.constant_(self.o_r.bias, 0)

        nn.init.xavier_normal_(self.i_l.weight)
        nn.init.xavier_normal_(self.i_r.weight)
        nn.init.xavier_normal_(self.f_l.weight)
        nn.init.xavier_normal_(self.f_r.weight)
        nn.init.xavier_normal_(self.o_l.weight)
        nn.init.xavier_normal_(self.o_r.weight)
        nn.init.xavier_normal_(self.g_l.weight)
        nn.init.xavier_normal_(self.g_r.weight)

    def forward(self, p, left, dim=0, root=False):
        if root:
            p_x, p_h, p_c = p.bu_state['x'], p.bu_state['h'], p.bu_state['c']
        else:
            p_x, p_h, p_c = p.bu_state['x'], p.td_state['h'], p.td_state['c']

        input_concat = torch.cat([p_x, p_h, p_c], dim=dim)
        g_concat = torch.cat([p_x, p_h], dim=dim)

        if left:
            i_t = F.sigmoid(self.i_l(input_concat))
            f_t = F.sigmoid(self.f_l(input_concat))
            o_t = F.sigmoid(self.o_l(input_concat))
            g_t = self.dropout_tree(F.tanh(self.g_l(g_concat)))
        else:
            i_t = F.sigmoid(self.i_r(input_concat))
            f_t = F.sigmoid(self.f_r(input_concat))
            o_t = F.sigmoid(self.o_r(input_concat))
            g_t = self.dropout_tree(F.tanh(self.g_r(g_concat)))
        h_t = o_t * self.activate_func2(p_c)
        c_t = f_t * p_c + g_t * i_t

        return {'h': h_t, 'c': c_t}
