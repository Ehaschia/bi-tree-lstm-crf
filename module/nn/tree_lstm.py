from .tree_lstm_cell import *
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.utils.rnn as rnn_utils
from .crf import TreeCRF, BinaryTreeCRF
from .lveg import BinaryTreeLVeG
from .attention import CoAttention
from allennlp.modules.elmo import Elmo, batch_to_ids


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
                 p_in=0.5, p_leaf=0.5, p_tree=0.5, p_pred=0.5, leaf_rnn=False, bi_leaf_rnn=False, device=None,
                 pred_dense_layer=False, attention=False, coattention_dim=150, elmo='None', elmo_weight=None,
                 elmo_config=None):
        super(TreeLstm, self).__init__()

        if elmo == 'none':
            self.elmo = None
            self.word_embedding = nn.Embedding(num_words, word_dim)
            reset_embedding(embedd_word, self.word_embedding, word_dim, embedd_trainable)
        elif elmo == 'only':
            self.elmo = Elmo(elmo_config, elmo_weight, 1, requires_grad=False, dropout=p_in)
            self.word_embedding = None
            word_dim = 1024
        elif elmo == 'cat':
            self.elmo = Elmo(elmo_config, elmo_weight, 1, requires_grad=False, dropout=p_in)
            self.word_embedding = nn.Embedding(num_words, word_dim)
            reset_embedding(embedd_word, self.word_embedding, word_dim, embedd_trainable)
            word_dim += 1024
        else:
            raise ValueError('Elmo error!')

        if tree_mode == 'SLSTM':
            self.bu_rnn_cell = BinarySLSTMCell(tree_input_dim, output_dim, p_tree=p_tree)
            # self.bu_rnn_cell = BinarySLSTMCell_chain(tree_input_dim, output_dim, p_tree=p_tree)
        elif tree_mode == 'TreeLSTM':
            head_word_dim = tree_input_dim if leaf_rnn else word_dim
            # self.bu_rnn_cell = BinaryTreeLSTMCell(head_word_dim, output_dim, p_tree=p_tree)
            self.bu_rnn_cell = TreeLSTMCell_flod(head_word_dim, output_dim, p_tree=p_tree)
        elif tree_mode == "BUTreeLSTM":
            self.bu_rnn_cell = BUSLSTMCell(word_dim, tree_input_dim, output_dim, p_tree=p_tree)
        else:
            raise NotImplementedError("the tree model " + tree_mode + " is not implemented!")

        self.device = device
        self.use_attention = attention
        if self.use_attention:
            self.attention = CoAttention(output_dim * 3, coattention_dim)

        self.tree_input_dim = tree_input_dim

        self.dropout_in = nn.Dropout(p=p_in)
        self.dropout_leaf = nn.Dropout(p=p_leaf)
        self.p_tree = p_tree
        self.dropout_pred = nn.Dropout(p=p_pred)
        self.leaf_rnn = leaf_rnn
        self.tree_mode = tree_mode
        self.pred_mode = pred_mode
        # self.softmax = nn.LogSoftmax(dim=1)
        self.pred_dense_layer = pred_dense_layer
        self.dense_softmax = None
        self.pred_layer = None

        # cross_entropy loss
        self.ce_loss = nn.CrossEntropyLoss()

        if pred_mode == 'single_h':
            pred_input_dim = coattention_dim if self.use_attention else output_dim
            self.generate_pred_layer(pred_input_dim, softmax_in_dim, num_labels)
            self.pred = self.single_h_pred
        elif pred_mode == 'avg_h':
            pred_input_dim = 2 * coattention_dim if self.use_attention else 2 * output_dim
            self.generate_pred_layer(pred_input_dim, softmax_in_dim, num_labels)
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
            self.leaf_affine = None
            if tree_mode == 'SLSTM':
                self.leaf_affine = nn.Linear(word_dim, tree_input_dim)

    def generate_pred_layer(self, input_size, softmax_in_dim, num_labels):
        if self.pred_dense_layer:
            self.dense_softmax = nn.Linear(input_size, softmax_in_dim)
            self.pred_layer = nn.Linear(softmax_in_dim, num_labels)
        else:
            self.pred_layer = nn.Linear(input_size, num_labels)

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
                if self.leaf_affine is not None:
                    h = self.leaf_affine(seq_out[tree.idx])
                else:
                    h = seq_out[tree.idx]
                c = h.detach().new(self.tree_input_dim).fill_(0.).requires_grad_()
                tree.bu_state = {'h': h, 'c': c}
            elif self.tree_mode == 'TreeLSTM':
                # because here the framework only depend on leaf input, so here inputs is seq_out
                # may be we can use word embedding?
                inputs = seq_out[tree.idx]
                l_c = inputs.detach().new(self.tree_input_dim).fill_(0.).requires_grad_()
                r_c = inputs.detach().new(self.tree_input_dim).fill_(0.).requires_grad_()

                l_h = inputs.detach().new(self.tree_input_dim).fill_(0.).requires_grad_()
                r_h = inputs.detach().new(self.tree_input_dim).fill_(0.).requires_grad_()
                l = {'h': l_h, 'c': l_c}
                r = {'h': r_h, 'c': r_c}
                tree.bu_state = self.bu_rnn_cell(l, r, inputs=inputs)
            elif self.tree_mode == 'BUTreeLSTM':
                inputs = embedding[tree.idx]
                c = inputs.detach().new(self.tree_input_dim).fill_(0.).requires_grad_()
                if self.rnn is not None:
                    tree.bu_state = {'x': inputs, 'h': seq_out[tree.idx], 'c': c}
                else:
                    h = inputs.detach().new(self.tree_input_dim).fill_(0.).requires_grad_()
                    tree.bu_state = {'x': inputs, 'h': h, 'c': c}
            else:
                raise NotImplementedError("the tree model " + self.tree_mode + " is not implemented!")
        else:
            assert len(tree.children) == 2
            inputs = None
            l = tree.get_child(0).bu_state
            r = tree.get_child(1).bu_state
            tree.bu_state = self.bu_rnn_cell(l, r, inputs=inputs)

    def collect_hidden_state(self, tree):
        hidden_collector = tree.collect_hidden_state([])
        hiddens = torch.cat(hidden_collector, dim=0)
        return hiddens

    def single_h_pred(self, tree, seq_out):
        hidden = self.collect_hidden_state(tree)
        hidden = self.dropout_pred(hidden)
        if self.dense_softmax is not None:
            softmax_in = F.relu(self.dense_softmax(hidden))
            pred = self.pred_layer(softmax_in)
        else:
            pred = self.pred_layer(hidden)
        return pred

    def avg_h_pred(self, tree, seq_out):
        # based on https://www.transacl.org/ojs/index.php/tacl/article/view/925
        hidden = self.collect_hidden_state(tree)
        hidden_size = hidden.size()
        avg_hidden = torch.mean(hidden, dim=0, keepdim=True).expand(hidden_size)
        hidden = torch.cat([avg_hidden, hidden], dim=1)
        if self.dense_softmax is not None:
            softmax_in = F.relu(self.dense_softmax(hidden))
            pred = self.pred_layer(softmax_in)
        else:
            pred = self.pred_layer(hidden)
        return pred

    def avg_seq_h_pred(self, tree, seq_out):
        # avg_h_pred + seq lstm
        # TODO
        raise NotImplementedError("Here not implement!")

    def forward(self, tree):
        if self.elmo is not None:
            str_sentences = [tree.get_str_yield()]
            character_ids = batch_to_ids(str_sentences).to(self.device)
            elmo_word = self.elmo(character_ids)['elmo_representations'][0].squeeze(0)
        else:
            elmo_word = None
        if self.word_embedding is not None:
            input_word = tree.get_yield()
            # TODO optimize the input word to gpu here
            word = self.word_embedding(torch.from_numpy(input_word).to(self.device))
            embedding_word = self.dropout_in(word)
        else:
            embedding_word = None

        # ugly
        if embedding_word is not None and elmo_word is not None:
            word = torch.cat([elmo_word, embedding_word], dim=1)
        elif embedding_word is not None:
            word = embedding_word
        elif elmo_word is not None:
            word = elmo_word
        else:
            raise ValueError('Embedding Error!')
        # sequence part
        # fixme requires grad
        word = word.requires_grad_()
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
        if self.use_attention:
            self.attention(tree)
        return seq_output

    # def loss(self, tree):
    #     seq_output = self.forward(tree)
    #     # pred part
    #     label_distribution = self.softmax(self.pred(tree, seq_output))
    #     return self.nll_loss(label_distribution, self.target.view(-1).to(self.device)).mean()

    def loss(self, tree):
        seq_output = self.forward(tree)
        # pred part
        pred_score = self.pred(tree, seq_output)
        target = tree.collect_golden_labels([])
        target = torch.LongTensor(target)
        loss = self.ce_loss(pred_score, target.view(-1).to(self.device))
        return loss

    def predict(self, tree):
        seq_output = self.forward(tree)
        pred_score = self.pred(tree, seq_output)
        preds = torch.argmax(pred_score, dim=1).cpu()
        target = tree.collect_golden_labels([])
        target = torch.LongTensor(target)

        # fine gain target
        corr = torch.eq(preds, target).float()

        # binary target
        binary_mask = target.ne(2)
        binary_preds = ((pred_score[:, 3] + pred_score[:, 4]) > (pred_score[:, 1] + pred_score[:, 2])).cpu()
        binary_lables = target > 2
        binary_corr = (torch.eq(binary_preds, binary_lables) * binary_mask).float()
        return corr, preds, [binary_corr], [binary_preds], binary_mask


class BiTreeLstm(TreeLstm):

    def __init__(self, tree_mode, seq_mode, pred_mode, word_dim, num_words, tree_input_dim, output_dim,
                 softmax_in_dim, seq_layer_num, num_labels, embedd_word=None, embedd_trainable=True,
                 p_in=0.5, p_leaf=0.5, p_tree=0.5, p_pred=0.5, leaf_rnn=False, bi_leaf_rnn=False, device=None,
                 pred_dense_layer=False, attention=False, coattention_dim=150, elmo='none', elmo_weight=None,
                 elmo_config=None):
        super(BiTreeLstm, self).__init__(tree_mode, seq_mode, pred_mode, word_dim, num_words, tree_input_dim,
                                         output_dim, softmax_in_dim, seq_layer_num, num_labels, embedd_word=embedd_word,
                                         embedd_trainable=embedd_trainable, p_in=p_in, p_leaf=p_leaf, p_tree=p_tree,
                                         p_pred=p_pred, leaf_rnn=leaf_rnn, bi_leaf_rnn=bi_leaf_rnn, device=device,
                                         pred_dense_layer=pred_dense_layer, attention=attention,
                                         coattention_dim=coattention_dim, elmo=elmo, elmo_weight=elmo_weight,
                                         elmo_config=elmo_config)
        # bi-directional only use for BUTreeLSTM
        if elmo == 'none':
            pass
        elif elmo == 'only':
            word_dim = 1024
        elif elmo == 'cat':
            word_dim += 1024
        else:
            raise ValueError('Elmo error!')

        assert tree_mode == 'BUTreeLSTM'
        self.td_rnn_cell = TDLSTMCell(word_dim, tree_input_dim, output_dim)
        if self.use_attention:
            self.attention = CoAttention(output_dim * 6, coattention_dim)
        if pred_mode == 'single_h':
            pred_input_dim = coattention_dim if self.use_attention else 2 * output_dim
            self.generate_pred_layer(pred_input_dim, softmax_in_dim, num_labels)
            self.pred = self.single_h_pred
        elif pred_mode == 'avg_h':
            pred_input_dim = 2 * coattention_dim if self.use_attention else 4 * output_dim
            self.generate_pred_layer(pred_input_dim, softmax_in_dim, num_labels)
            self.pred = self.avg_h_pred
        elif pred_mode == 'avg_seq_h':
            # TODO
            self.pred_layer = nn.Linear(output_dim + 2 * tree_input_dim, num_labels)
            self.pred = self.avg_seq_h_pred
        else:
            raise NotImplementedError("the pred model " + pred_mode + " is not implemented!")

    def collect_hidden_state(self, tree):
        hidden_collector = tree.collect_hidden_state([])
        hiddens = torch.cat(hidden_collector, dim=0)
        return hiddens

    def generate_pred_layer(self, input_size, softmax_in_dim, num_labels):
        if self.pred_dense_layer:
            self.dense_softmax = nn.Linear(input_size, softmax_in_dim)
            self.pred_layer = nn.Linear(softmax_in_dim, num_labels)
        else:
            self.pred_layer = nn.Linear(input_size, num_labels)

    def single_h_pred(self, tree, seq_out):
        hidden = self.collect_hidden_state(tree)
        # alert may cause multi dropout here
        hidden = self.dropout_pred(hidden)
        if self.dense_softmax is not None:
            softmax_in = F.relu(self.dense_softmax(hidden))
            pred = self.pred_layer(softmax_in)
        else:
            pred = self.pred_layer(hidden)
        return pred

    def avg_h_pred(self, tree, seq_out):
        # based on https://www.transacl.org/ojs/index.php/tacl/article/view/925
        hidden = self.collect_hidden_state(tree)
        hidden_size = hidden.size()
        avg_hidden = torch.mean(hidden, dim=0, keepdim=True).expand(hidden_size)
        hidden = torch.cat([avg_hidden, hidden], dim=1)
        hidden = self.dropout_pred(hidden)
        if self.dense_softmax is not None:
            softmax_in = F.relu(self.dense_softmax(hidden))
            pred = self.pred_layer(softmax_in)
        else:
            pred = self.pred_layer(hidden)
        return pred

    def avg_seq_h_pred(self, tree, seq_out):
        # avg_h_pred + seq lstm
        # TODO
        raise NotImplementedError("Here not implement!")

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
            if self.rnn is not None:
                tree.bu_state = {'x': inputs, 'h': seq_out[tree.idx], 'c': c}
            else:
                h = inputs.detach().new(self.tree_input_dim).fill_(0.).requires_grad_()
                tree.bu_state = {'x': inputs, 'h': h, 'c': c}
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
        if self.elmo is not None:
            str_sentences = [tree.get_str_yield()]
            character_ids = batch_to_ids(str_sentences).to(self.device)
            elmo_word = self.elmo(character_ids)['elmo_representations'][0].squeeze(0)
        else:
            elmo_word = None
        if self.word_embedding is not None:
            input_word = tree.get_yield()
            # TODO optimize the input word to gpu here
            word = self.word_embedding(torch.from_numpy(input_word).to(self.device))
            embedding_word = self.dropout_in(word)
        else:
            embedding_word = None

        # ugly
        if embedding_word is not None and elmo_word is not None:
            word = torch.cat([elmo_word, embedding_word], dim=0)
        elif embedding_word is not None:
            word = embedding_word
        elif elmo_word is not None:
            word = elmo_word
        else:
            raise ValueError('Embedding Error!')
        # sequence part
        # fixme requires grad
        word = word.requires_grad_()
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
        if self.use_attention:
            self.attention(tree)
        return seq_output


class CRFTreeLstm(TreeLstm):
    def __init__(self, tree_mode, seq_mode, pred_mode, word_dim, num_words, tree_input_dim, output_dim,
                 softmax_in_dim, seq_layer_num, num_labels, embedd_word=None, embedd_trainable=True,
                 p_in=0.5, p_leaf=0.5, p_tree=0.5, p_pred=0.5, leaf_rnn=False, bi_leaf_rnn=False, device=None,
                 pred_dense_layer=False, attention=False, coattention_dim=150, elmo='none', elmo_weight=None,
                 elmo_config=None):
        super(CRFTreeLstm, self).__init__(tree_mode, seq_mode, pred_mode, word_dim, num_words, tree_input_dim,
                                          output_dim, softmax_in_dim, seq_layer_num, num_labels,
                                          embedd_word=embedd_word, embedd_trainable=embedd_trainable, p_in=p_in,
                                          p_leaf=p_leaf, p_tree=p_tree, p_pred=p_pred, leaf_rnn=leaf_rnn,
                                          bi_leaf_rnn=bi_leaf_rnn, device=device, pred_dense_layer=pred_dense_layer,
                                          attention=attention, coattention_dim=coattention_dim, elmo=elmo,
                                          elmo_weight=elmo_weight, elmo_config=elmo_config)
        crf_input_dim = coattention_dim if self.use_attention else output_dim
        self.crf = TreeCRF(crf_input_dim, num_labels, attention=self.use_attention, pred_mode=pred_mode,
                           only_bu=True)

        self.ce_loss = None
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
        # fine gain target
        corr = torch.eq(preds, target).float()
        # binary target
        binary_mask = target.ne(2)
        binary_preds_1 = preds > 2
        binary_preds_2 = preds >= 2
        binary_lables = target > 2
        binary_corr_1 = (torch.eq(binary_preds_1, binary_lables) * binary_mask).float()
        binary_corr_2 = (torch.eq(binary_preds_2, binary_lables) * binary_mask).float()
        return corr, preds, [binary_corr_1, binary_corr_2], [binary_preds_1, binary_preds_2], binary_mask


class CRFBiTreeLstm(BiTreeLstm):

    def __init__(self, tree_mode, seq_mode, pred_mode, word_dim, num_words, tree_input_dim, output_dim,
                 softmax_in_dim, seq_layer_num, num_labels, embedd_word=None, embedd_trainable=True,
                 p_in=0.5, p_leaf=0.5, p_tree=0.5, p_pred=0.5, leaf_rnn=False, bi_leaf_rnn=False, device=None,
                 pred_dense_layer=False, attention=False, coattention_dim=150, elmo='none', elmo_weight=None,
                 elmo_config=None):
        super(CRFBiTreeLstm, self).__init__(tree_mode, seq_mode, pred_mode, word_dim, num_words, tree_input_dim,
                                            output_dim, softmax_in_dim, seq_layer_num, num_labels,
                                            embedd_word=embedd_word, embedd_trainable=embedd_trainable, p_in=p_in,
                                            p_leaf=p_leaf, p_tree=p_tree, p_pred=p_pred, leaf_rnn=leaf_rnn,
                                            bi_leaf_rnn=bi_leaf_rnn, device=device, pred_dense_layer=pred_dense_layer,
                                            attention=attention, coattention_dim=coattention_dim, elmo=elmo,
                                            elmo_weight=elmo_weight, elmo_config=None)
        crf_input_dim = coattention_dim if self.use_attention else output_dim
        self.crf = TreeCRF(crf_input_dim, num_labels, attention=self.use_attention, pred_mode=pred_mode,
                           only_bu=False, softmax_in_dim=softmax_in_dim)
        self.ce_loss = None
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
        # fine gain target
        corr = torch.eq(preds, target).float()
        # binary target
        binary_mask = target.ne(2)
        binary_preds_1 = preds > 2
        binary_preds_2 = preds >= 2
        binary_lables = target > 2
        binary_corr_1 = (torch.eq(binary_preds_1, binary_lables) * binary_mask).float()
        binary_corr_2 = (torch.eq(binary_preds_2, binary_lables) * binary_mask).float()
        return corr, preds, [binary_corr_1, binary_corr_2], [binary_preds_1, binary_preds_2], binary_mask


class LVeGBiTreeLstm(BiTreeLstm):
    # alert LVeG not implement the pred dense num
    def __init__(self, tree_mode, seq_mode, pred_mode, word_dim, num_words, tree_input_dim, output_dim,
                 softmax_in_dim, seq_layer_num, num_labels, embedd_word=None, embedd_trainable=True, comp=1, g_dim=1,
                 p_in=0.5, p_leaf=0.5, p_tree=0.5, p_pred=0.5, leaf_rnn=False, bi_leaf_rnn=False, device=None,
                 pred_dense_layer=False, attention=False, coattention_dim=150, elmo='none', elmo_weight=None,
                 elmo_config=None):
        super(LVeGBiTreeLstm, self).__init__(tree_mode, seq_mode, pred_mode, word_dim, num_words, tree_input_dim,
                                             output_dim, softmax_in_dim, seq_layer_num, num_labels,
                                             embedd_word=embedd_word, embedd_trainable=embedd_trainable, p_in=p_in,
                                             p_leaf=p_leaf, p_tree=p_tree, p_pred=p_pred, leaf_rnn=leaf_rnn,
                                             bi_leaf_rnn=bi_leaf_rnn, device=device, pred_dense_layer=pred_dense_layer,
                                             attention=attention, coattention_dim=coattention_dim, elmo=elmo,
                                             elmo_weight=elmo_weight, elmo_config=elmo_config)
        lveg_input_dim = coattention_dim if self.use_attention else output_dim
        self.lveg = BinaryTreeLVeG(lveg_input_dim, num_labels, comp, g_dim, attention=self.use_attention,
                                   pred_mode=pred_mode, only_bu=False, softmax_in_dim=softmax_in_dim)

        self.nll_loss = None
        self.pred_layer = None
        self.pred = None
        self.dense_softmax = None

    def loss(self, tree):
        seq_out = self.forward(tree)
        loss = self.lveg.loss(tree)
        return loss

    def predict(self, tree):
        seq_out = self.forward(tree)
        preds = self.lveg.predict(tree)
        preds = torch.Tensor(preds)
        target = tree.collect_golden_labels([])
        target = torch.Tensor(target)
        # fine gain target
        corr = torch.eq(preds, target).float()

        # binary target
        binary_mask = target.ne(2)
        binary_preds_1 = preds > 2
        binary_preds_2 = preds >= 2
        binary_lables = target > 2
        binary_corr_1 = (torch.eq(binary_preds_1, binary_lables) * binary_mask).float()
        binary_corr_2 = (torch.eq(binary_preds_2, binary_lables) * binary_mask).float()
        return corr, preds, [binary_corr_1, binary_corr_2], [binary_preds_1, binary_preds_2], binary_mask


class LVeGTreeLstm(TreeLstm):
    # alert LVeG not implement the pred dense num
    def __init__(self, tree_mode, seq_mode, pred_mode, word_dim, num_words, tree_input_dim, output_dim,
                 softmax_in_dim, seq_layer_num, num_labels, embedd_word=None, embedd_trainable=True, comp=1, g_dim=1,
                 p_in=0.5, p_leaf=0.5, p_tree=0.5, p_pred=0.5, leaf_rnn=False, bi_leaf_rnn=False, device=None,
                 pred_dense_layer=False, attention=False, coattention_dim=150, elmo='none', elmo_weight=None,
                 elmo_config=None):
        super(LVeGTreeLstm, self).__init__(tree_mode, seq_mode, pred_mode, word_dim, num_words, tree_input_dim,
                                           output_dim, softmax_in_dim, seq_layer_num, num_labels,
                                           embedd_word=embedd_word, embedd_trainable=embedd_trainable, p_in=p_in,
                                           p_leaf=p_leaf, p_tree=p_tree, p_pred=p_pred, leaf_rnn=leaf_rnn,
                                           bi_leaf_rnn=bi_leaf_rnn, device=device, pred_dense_layer=pred_dense_layer,
                                           attention=attention, coattention_dim=coattention_dim, elmo=elmo,
                                           elmo_weight=elmo_weight, elmo_config=elmo_config)
        lveg_input_dim = coattention_dim if self.use_attention else output_dim
        self.lveg = BinaryTreeLVeG(lveg_input_dim, num_labels, comp, g_dim, attention=self.use_attention,
                                   pred_mode=pred_mode, only_bu=True, softmax_in_dim=softmax_in_dim)

        self.ce_loss = None
        self.pred_layer = None
        self.pred = None
        self.dense_softmax = None

    def loss(self, tree):
        seq_output = self.forward(tree)
        return self.lveg.loss(tree)

    def predict(self, tree):
        # fixme make it clear
        seq_output = self.forward(tree)
        preds = self.lveg.predict(tree)
        preds = torch.Tensor(preds).cpu()
        target = tree.collect_golden_labels([])
        target = torch.Tensor(target)
        # fine gain target
        corr = torch.eq(preds, target).float()

        # binary target
        binary_mask = target.ne(2)
        binary_preds_1 = preds > 2
        binary_preds_2 = preds >= 2
        binary_lables = target > 2
        binary_corr_1 = (torch.eq(binary_preds_1, binary_lables) * binary_mask).float()
        binary_corr_2 = (torch.eq(binary_preds_2, binary_lables) * binary_mask).float()
        return corr, preds, [binary_corr_1, binary_corr_2], [binary_preds_1, binary_preds_2], binary_mask


class BiCRFBiTreeLstm(BiTreeLstm):

    def __init__(self, tree_mode, seq_mode, pred_mode, word_dim, num_words, tree_input_dim, output_dim,
                 softmax_in_dim, seq_layer_num, num_labels, embedd_word=None, embedd_trainable=True,
                 p_in=0.5, p_leaf=0.5, p_tree=0.5, p_pred=0.5, leaf_rnn=False, bi_leaf_rnn=False, device=None,
                 pred_dense_layer=False, attention=False, coattention_dim=150, elmo='none', elmo_weight=None,
                 elmo_config=None):
        super(BiCRFBiTreeLstm, self).__init__(tree_mode, seq_mode, pred_mode, word_dim, num_words, tree_input_dim,
                                              output_dim, softmax_in_dim, seq_layer_num, num_labels,
                                              embedd_word=embedd_word, embedd_trainable=embedd_trainable, p_in=p_in,
                                              p_leaf=p_leaf, p_tree=p_tree, p_pred=p_pred, leaf_rnn=leaf_rnn,
                                              bi_leaf_rnn=bi_leaf_rnn, device=device, pred_dense_layer=pred_dense_layer,
                                              attention=attention, coattention_dim=coattention_dim, elmo=elmo,
                                              elmo_weight=elmo_weight, elmo_config=elmo_config)
        crf_input_dim = coattention_dim if self.use_attention else output_dim
        self.crf = BinaryTreeCRF(crf_input_dim, num_labels, attention=self.use_attention,
                                 pred_mode=pred_mode, only_bu=False, softmax_in_dim=softmax_in_dim)
        self.ce_loss = None
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
        # fine gain target
        corr = torch.eq(preds, target).float()

        # binary target
        binary_mask = target.ne(2)
        binary_preds_1 = preds > 2
        binary_preds_2 = preds >= 2
        binary_lables = target > 2
        binary_corr_1 = (torch.eq(binary_preds_1, binary_lables) * binary_mask).float()
        binary_corr_2 = (torch.eq(binary_preds_2, binary_lables) * binary_mask).float()
        return corr, preds, [binary_corr_1, binary_corr_2], [binary_preds_1, binary_preds_2], binary_mask


class BiCRFTreeLstm(TreeLstm):

    def __init__(self, tree_mode, seq_mode, pred_mode, word_dim, num_words, tree_input_dim, output_dim,
                 softmax_in_dim, seq_layer_num, num_labels, embedd_word=None, embedd_trainable=True,
                 p_in=0.5, p_leaf=0.5, p_tree=0.5, p_pred=0.5, leaf_rnn=False, bi_leaf_rnn=False, device=None,
                 pred_dense_layer=False, attention=False, coattention_dim=150, elmo='none', elmo_weight=None,
                 elmo_config=None):
        super(BiCRFTreeLstm, self).__init__(tree_mode, seq_mode, pred_mode, word_dim, num_words, tree_input_dim,
                                            output_dim, softmax_in_dim, seq_layer_num, num_labels,
                                            embedd_word=embedd_word, embedd_trainable=embedd_trainable, p_in=p_in,
                                            p_leaf=p_leaf, p_tree=p_tree, p_pred=p_pred, leaf_rnn=leaf_rnn,
                                            bi_leaf_rnn=bi_leaf_rnn, device=device, pred_dense_layer=pred_dense_layer,
                                            attention=attention, coattention_dim=coattention_dim, elmo=elmo,
                                            elmo_weight=elmo_weight, elmo_config=elmo_config)
        crf_input_dim = coattention_dim if self.use_attention else output_dim
        self.crf = BinaryTreeCRF(crf_input_dim, num_labels, attention=False, pred_mode=pred_mode,
                                 only_bu=True, softmax_in_dim=softmax_in_dim)
        self.ce_loss = None
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
        # fine gain target
        corr = torch.eq(preds, target).float()

        # binary target
        binary_mask = target.ne(2)
        binary_preds_1 = preds > 2
        binary_preds_2 = preds >= 2
        binary_lables = target > 2
        binary_corr_1 = (torch.eq(binary_preds_1, binary_lables) * binary_mask).float()
        binary_corr_2 = (torch.eq(binary_preds_2, binary_lables) * binary_mask).float()
        return corr, preds, [binary_corr_1, binary_corr_2], [binary_preds_1, binary_preds_2], binary_mask
