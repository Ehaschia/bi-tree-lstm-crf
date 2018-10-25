from .tree_lstm_cell import *
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.utils.rnn as rnn_utils
from .crf import TreeCRF


def reset_embedding(init_embedding, embedding_layer, embedding_dim, trainable):
    if init_embedding is None:
        scale = np.sqrt(3.0 / embedding_dim)
        embedding_layer.weight.data.uniform_(-scale, scale)
    else:
        embedding_layer.load_state_dict({'weight': init_embedding})

    embedding_layer.weight.requires_grad = trainable


class TreeLstm(nn.Module):
    # traditional Bottom up model
    def __init__(self, tree_mode, seq_mode, pred_mode, word_dim, num_words, tree_input_dim, output_dim,
                 softmax_in_dim, seq_layer_num, num_labels, embedd_word=None, embedd_trainable=True,
                 p_in=0.5, p_leaf=0.5, p_tree=0.5, p_pred=0.5, leaf_rnn=False, bi_leaf_rnn=False, device=None):
        # TODO Attention
        super(TreeLstm, self).__init__()

        if tree_mode == 'SLSTM':
            self.bu_rnn_cell = BinarySLSTMCell(tree_input_dim, output_dim)
        elif tree_mode == 'TreeLSTM':
            # alert here input size
            self.bu_rnn_cell = BinaryTreeLSTMCell(tree_input_dim, output_dim)
        elif tree_mode == "BUTreeLSTM":
            self.bu_rnn_cell = BUSLSTMCell(word_dim, tree_input_dim, output_dim)
        else:
            raise NotImplementedError("the tree model " + tree_mode + " is not implemented!")

        self.device = device

        self.tree_input_dim = tree_input_dim
        self.word_embedding = nn.Embedding(num_words, word_dim)
        reset_embedding(embedd_word, self.word_embedding, word_dim, embedd_trainable)
        self.dropout_in = nn.Dropout(p=p_in)
        self.dropout_leaf = nn.Dropout(p=p_leaf)
        self.p_tree = p_tree
        self.dropout_pred = nn.Dropout(p=p_pred)
        self.leaf_rnn = leaf_rnn
        self.tree_mode = tree_mode
        self.pred_mode = pred_mode
        self.target = []
        self.softmax = nn.LogSoftmax(dim=0)

        self.nll_loss = nn.NLLLoss(size_average=False, reduce=False)

        if pred_mode == 'single_h':
            self.pred_layer = nn.Linear(output_dim, num_labels)
            self.pred = self.single_h_pred
        elif pred_mode == 'avg_h':
            self.dense_softmax = nn.Linear(4*output_dim, softmax_in_dim)
            self.pred_layer = nn.Linear(softmax_in_dim, num_labels)
            self.pred = self.avg_h_pred
        elif pred_mode == 'avg_seq_h':
            # TODO
            self.pred_layer = nn.Linear(output_dim + 2 * tree_input_dim, num_labels)
            self.pred = self.avg_seq_h_pred
        else:
            raise NotImplementedError("the pred model " + pred_mode + " is not implemented!")

        if leaf_rnn:
            if seq_mode == 'RNN':
                RNN = nn.RNN
            elif seq_mode == 'LSTM':
                RNN = nn.LSTM
            elif seq_mode == 'GRU':
                RNN = nn.GRU
            tree_input_dim = (tree_input_dim // 2) if bi_leaf_rnn else tree_input_dim
            self.rnn = RNN(word_dim, tree_input_dim, num_layers=seq_layer_num,
                           batch_first=True, bidirectional=bi_leaf_rnn, dropout=p_leaf)
            self.leaf_affine = None
        else:
            self.rnn = None
            # alert non-linear layer?
            self.leaf_affine = nn.Linear(word_dim, tree_input_dim)

    def recursive_tree(self, tree, seq_out, embedding):
        # tree part
        # alert :
        # for leaf node:
        # tree head word (x) is embedding
        # hidden dim use seq_out
        for idx in range(len(tree.children)):
            self.recursive_tree(tree.children[idx], seq_out, embedding)

        if tree.is_leaf():
            if self.tree_mode == 'SLSTM':
                h = seq_out[tree.idx].unsqueeze(0)
                c = h.detach().new(1, self.tree_input_dim).fill_(0.).requires_grad_()
                tree.bu_state = {'h': h, 'c': c}
            elif self.tree_mode == 'TreeLSTM':
                # because here the framework only depend on leaf input, so here inputs is seq_out
                inputs = seq_out[tree.idx]
                l_c = inputs.detach().new(1, self.tree_input_dim).fill_(0.).requires_grad_()
                r_c = inputs.detach().new(1, self.tree_input_dim).fill_(0.).requires_grad_()

                l_h = inputs.detach().new(1, self.tree_input_dim).fill_(0.).requires_grad_()
                r_h = inputs.detach().new(1, self.tree_input_dim).fill_(0.).requires_grad_()
                l = {'h': l_h, 'c': l_c}
                r = {'h': r_h, 'c': r_c}
                tree.bu_state = self.bu_rnn_cell(l, r, inputs=inputs)
            elif self.tree_mode == 'BUTreeLSTM':
                inputs = embedding[tree.idx]
                c = inputs.detach().new(self.tree_input_dim).fill_(0.).requires_grad_()
                tree.bu_state = {'x': inputs, 'h': seq_out[tree.idx], 'c': c}
            else:
                raise NotImplementedError("the tree model " + self.tree_mode + " is not implemented!")
        else:
            assert len(tree.children) == 2
            inputs = None
            l = tree.get_child(0).bu_state
            r = tree.get_child(1).bu_state
            tree.bu_state = self.bu_rnn_cell(l, r, inputs=inputs)

    def collect_hidden_state(self, tree):
        hidden_collector, self.target = tree.collect_hidden_state([], label_holder=[])
        hiddens = torch.cat(hidden_collector, dim=0)
        self.target = torch.from_numpy(np.array(self.target))
        return hiddens

    def single_h_pred(self, tree, seq_out):
        hidden = self.collect_hidden_state(tree)
        # alert may cause multi dropout here
        hidden = self.dropout_pred(hidden)
        return self.pred_layer(hidden)

    def avg_h_pred(self, tree, seq_out):
        # based on https://www.transacl.org/ojs/index.php/tacl/article/view/925
        hidden = self.collect_hidden_state(tree)
        hidden_size = hidden.size()
        avg_hidden = torch.mean(hidden, dim=0, keepdim=True).expand(hidden_size)
        hidden = torch.cat([avg_hidden, hidden], dim=0)
        softmax_in = F.relu(self.dense_softmax(hidden))
        softmax_in = self.dropout_pred(softmax_in)
        return self.pred_layer(softmax_in)

    def avg_seq_h_pred(self, tree, seq_out):
        # avg_h_pred + seq lstm
        # TODO
        raise NotImplementedError("Here not implement!")

    def forward(self, tree):
        input_word = tree.get_yield()
        # TODO optimize the input word to gpu here
        word = self.word_embedding(torch.from_numpy(input_word).to(self.device))
        word = self.dropout_in(word)
        # sequence part
        if self.rnn is not None:
            word = word.unsqueeze(dim=0)
            lens = [word.size()[1]]
            seq_input = rnn_utils.pack_padded_sequence(word, lens, batch_first=True)
            # alert hx init?
            seq_output, hn = self.rnn(seq_input)
            seq_output, _ = rnn_utils.pad_packed_sequence(seq_output, batch_first=True)
        else:
            word = word.unsqueeze(dim=0)
            seq_output = word
        # tree part
        seq_output = seq_output.squeeze(0)
        word = word.squeeze(0)
        self.recursive_tree(tree, seq_output, word)
        return seq_output

    def loss(self, tree):
        seq_output = self.forward(tree)
        # pred part
        label_distribution = self.softmax(self.pred(tree, seq_output))
        return self.nll_loss(label_distribution, self.target.view(-1).to(self.device)).mean()

    def predict(self, tree):
        seq_output = self.forward(tree)
        label_distribution = self.softmax(self.pred(tree, seq_output))
        preds = torch.argmax(label_distribution, dim=1).cpu()
        return torch.eq(preds, self.target).float(), preds


class BiTreeLstm(TreeLstm):

    def __init__(self, tree_mode, seq_mode, pred_mode, word_dim, num_words, tree_input_dim, output_dim,
                 softmax_in_dim, seq_layer_num, num_labels, embedd_word=None, embedd_trainable=True,
                 p_in=0.5, p_leaf=0.5, p_tree=0.5, p_pred=0.5, leaf_rnn=False, bi_leaf_rnn=False, device=None):
        super(BiTreeLstm, self).__init__(tree_mode, seq_mode, pred_mode, word_dim, num_words, tree_input_dim,
                                         output_dim, softmax_in_dim, seq_layer_num, num_labels, embedd_word=embedd_word,
                                         embedd_trainable=embedd_trainable, p_in=p_in, p_leaf=p_leaf, p_tree=p_tree,
                                         p_pred=p_pred, leaf_rnn=leaf_rnn, bi_leaf_rnn=bi_leaf_rnn, device=device)
        # bi-directional only use for BUTreeLSTM
        assert tree_mode == 'BUTreeLSTM'
        self.td_rnn_cell = TDLSTMCell(word_dim, tree_input_dim, output_dim)

    def collect_hidden_state(self, tree):
        hidden_collector, self.target = tree.collect_hidden_state([], label_holder=[], bidirectional=True)
        hiddens = torch.cat(hidden_collector, dim=0)
        self.target = torch.from_numpy(np.array(self.target))
        return hiddens

    def recursive_tree(self, tree, seq_out, embedding):
        # tree part
        # alert :
        # for leaf node:
        # tree head word (x) is embedding
        # hidden dim use seq_out
        for idx in range(len(tree.children)):
            self.recursive_tree(tree.children[idx], seq_out, embedding)

        if tree.is_leaf():
            inputs = embedding[tree.idx]
            c = inputs.detach().new(self.tree_input_dim).fill_(0.).requires_grad_()
            tree.bu_state = {'x': inputs, 'h': seq_out[tree.idx], 'c': c}
        else:
            assert len(tree.children) == 2
            inputs = None
            l = tree.get_child(0).bu_state
            r = tree.get_child(1).bu_state
            tree.bu_state = self.bu_rnn_cell(l, r, inputs=inputs)

    def backward_recursive_tree(self, tree, left=False):
        if tree.parent is not None:
            tree.td_state = self.td_rnn_cell(tree.parent, left=left, root=False)

        if tree.is_leaf():
            return

        for idx in range(len(tree.children)):
            if idx == 0:
                self.backward_recursive_tree(tree.children[0], left=True)
            else:
                self.backward_recursive_tree(tree.children[1], left=False)

    def forward(self, tree):
        input_word = tree.get_yield()
        # TODO optimize the input word to gpu here
        word = self.word_embedding(torch.from_numpy(input_word).to(self.device))
        word = self.dropout_in(word)
        # sequence part
        if self.rnn is not None:
            word = word.unsqueeze(dim=0)
            lens = [word.size()[1]]
            seq_input = rnn_utils.pack_padded_sequence(word, lens, batch_first=True)
            # alert hx init?
            seq_output, hn = self.rnn(seq_input)
            seq_output, _ = rnn_utils.pad_packed_sequence(seq_output, batch_first=True)
        else:
            word = word.unsqueeze(dim=0)
            seq_output = word

        # tree part
        seq_output = seq_output.squeeze(0)
        word = word.squeeze(0)
        self.recursive_tree(tree, seq_output, word)
        # alert here we deal with the root as left branch
        tree.td_state = self.td_rnn_cell(tree, True, root=True)
        self.backward_recursive_tree(tree, left=False)

        return seq_output


class CRFTreeLstm(TreeLstm):
    def __init__(self, tree_mode, seq_mode, pred_mode, word_dim, num_words, tree_input_dim, output_dim,
                 softmax_in_dim, seq_layer_num, num_labels, embedd_word=None, embedd_trainable=True,
                 p_in=0.5, p_leaf=0.5, p_tree=0.5, p_pred=0.5, leaf_rnn=False, bi_leaf_rnn=False, device=None):
        super(CRFTreeLstm, self).__init__(tree_mode, seq_mode, pred_mode, word_dim, num_words, tree_input_dim,
                                          output_dim, softmax_in_dim, seq_layer_num, num_labels,
                                          embedd_word=embedd_word, embedd_trainable=embedd_trainable, p_in=p_in,
                                          p_leaf=p_leaf, p_tree=p_tree, p_pred=p_pred, leaf_rnn=leaf_rnn,
                                          bi_leaf_rnn=bi_leaf_rnn, device=device)

        self.crf = TreeCRF(output_dim, num_labels, attention=False, pred_mode=pred_mode)

        self.softmax = None
        self.nll_loss = None
        self.pred_layer = None
        self.pred = None

    def loss(self, tree):
        seq_output = self.forward(tree)
        return self.crf.loss(tree)

    def predict(self, tree):
        # fixme make it clear
        seq_output = self.forward(tree)
        preds = self.crf.predict(tree)
        preds = torch.Tensor(preds).cpu()
        target = tree.collect_golden_labels([])
        target = torch.Tensor(target)
        return torch.eq(preds, target).float(), preds


class CRFBiTreeLstm(BiTreeLstm):

    def __init__(self, tree_mode, seq_mode, pred_mode, word_dim, num_words, tree_input_dim, output_dim,
                 softmax_in_dim, seq_layer_num, num_labels, embedd_word=None, embedd_trainable=True,
                 p_in=0.5, p_leaf=0.5, p_tree=0.5, p_pred=0.5, leaf_rnn=False, bi_leaf_rnn=False, device=None):
        super(CRFBiTreeLstm, self).__init__(tree_mode, seq_mode, pred_mode, word_dim, num_words, tree_input_dim,
                                            output_dim, softmax_in_dim, seq_layer_num, num_labels,
                                            embedd_word=embedd_word, embedd_trainable=embedd_trainable, p_in=p_in,
                                            p_leaf=p_leaf, p_tree=p_tree, p_pred=p_pred, leaf_rnn=leaf_rnn,
                                            bi_leaf_rnn=bi_leaf_rnn, device=device)

        self.crf = TreeCRF(output_dim, num_labels, attention=False, pred_mode=pred_mode, only_bu=False)
        self.softmax = None
        self.nll_loss = None
        self.pred_layer = None
        self.pred = None

    def loss(self, tree):
        seq_out = self.forward(tree)
        loss = self.crf.loss(tree)
        return loss

    def predict(self, tree):
        seq_output = self.forward(tree)
        preds = self.crf.predict(tree)
        preds = torch.Tensor(preds).cpu()
        target = tree.collect_golden_labels([])
        target = torch.Tensor(target)
        return torch.eq(preds, target).float(), preds
