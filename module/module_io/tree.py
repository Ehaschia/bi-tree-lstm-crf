__author__ = 'Ehaschia'
import numpy as np
import torch


class Tree(object):

    def __init__(self, label, word_idx=None, str_word=None):
        self.parent = None
        self.label = label
        self.children = []
        self.word_idx = word_idx
        self.length = None
        # the depth from this node to the furthest child leaf.
        self.height = 0
        self.position_idx = 0
        self.bu_state = {}
        self.td_state = {}
        self.str_word = str_word
        self.crf_cache = {}
        self.lveg_cache = {}
        self.attention_cache = {}

    def add_child(self, child):
        child.parent = self
        self.children.append(child)
        self.height = (child.height + 1) if child.height >= self.height else self.height

    def set_children(self, children):
        for child in children:
            self.add_child(child)
            self.height = (child.height + 1) if child.height >= self.height else self.height

    def get_child(self, idx):
        if idx < len(self.children):
            return self.children[idx]
        else:
            raise IndexError("idx %d out of bound %d" % (idx, len(self.children)))

    def get_sentence(self, sent):
        if self.is_leaf():
            sent.append(self.word_idx)
        for child in self.children:
            child.get_sentence(sent)

    def is_leaf(self):
        return len(self.children) == 0

    def get_yield(self):
        """
        :return: word list, ndarray format
        """
        sents = []
        if self.is_leaf():
            self.length = 1
            sents.append(self.word_idx)
            return np.array(sents)

        for child in self.children:
            sents += child.get_yield().tolist()
        self.length = len(sents)
        return np.array(sents)

    def get_str_yield(self):
        """
        :return: string word list, list format
        """
        sents = []
        if self.is_leaf():
            self.length = 1
            sents.append(self.str_word)
            return sents

        for child in self.children:
            sents += child.get_str_yield()
        self.length = len(sents)
        return sents

    def get_yield_node(self):
        """
        :return: leaf nodes, list format
        """
        leaves = []
        if self.is_leaf():
            return [self]
        for child in self.children:
            leaves += child.get_yield_node()
        if self.length is None:
            self.length = len(leaves)
        return leaves

    def __len__(self):
        if self.length is not None:
            pass
        else:
            self.get_yield()
        return self.length

    def __eq__(self, other):
        if self.is_leaf():
            if other.is_leaf():
                return self.position_idx == self.position_idx and self.word_idx == self.word_idx
            else:
                return False
        else:
            if other.is_leaf():
                return False
            else:
                if len(other.children == self.children):
                    equal = True
                    for idx, child in enumerate(self.children):
                        equal &= (child == other.get_child(idx))
                else:
                    return False

    def sequence_idx(self, pad_length):
        # pad the sentence to pand_lenth, calculate the index in the chart
        #   *      5
        #  * *    3 4
        # * * *  0 1 2
        # calculate index in a spaned tree:
        # leftmost_left_node_index + length*(pad_length) + span_num*(span_num+1)/2 - 1
        # in above tree the span for "3" is "0" and "1" so here span=2
        if self.is_leaf():
            return self.position_idx

        span_num = self.height + 1
        span_idx_part = span_num * (span_num + 1) // 2 - 1
        lenth_idx_part = self.height * (pad_length - 1 - self.height)
        node = self.children[0]
        while not node.is_leaf():
            node = node.children[0]
        leftmost_idx = node.position_idx
        idx = span_idx_part + lenth_idx_part + leftmost_idx
        return int(idx)

    def collect_hidden_state(self, holder):
        for idx in range(len(self.children)):
            self.children[idx].collect_hidden_state(holder)
        if 'h' in self.td_state:
            hidden_state = torch.cat([self.bu_state['h'], self.td_state['h']], dim=0)
            if 'output_h' in self.td_state:
                hidden_state = torch.cat([hidden_state, self.td_state['output_h']], dim=0)
        else:
            hidden_state = self.bu_state['h']

        if hidden_state.size()[0] != 1:
            holder.append(hidden_state.unsqueeze(0))
        else:
            holder.append(hidden_state)
        return holder

    def collect_golden_labels(self, holder):
        for child in self.children:
            child.collect_golden_labels(holder)
        holder.append(self.label)
        return holder

    def label_size(self, cnt):
        for idx in range(len(self.children)):
            self.children[idx].label_size(cnt)
        cnt += 1
        return cnt

    def clean(self):
        for i in self.children:
            i.clean()
        self.bu_state = {}
        self.td_state = {}
        self.crf_cache = {}
        self.lveg_cache = {}

    def replace_unk(self, word_alphabet, embedding, isTraining=True):
        for child in self.children:
            child.replace_unk(word_alphabet, embedding, isTraining=isTraining)

        if self.is_leaf():
            count = word_alphabet.count(self.str_word)
            if isTraining:
                if self.str_word not in embedding:
                    lower = str.lower(self.str_word)
                    if lower in embedding:
                        self.word_idx = word_alphabet.get_idx(lower)
                    elif lower not in embedding and count == 1:
                        print('[TRN UNK]: ' + self.str_word)
                        self.word_idx = 0
                    elif lower not in embedding and count > 1:
                        print('[2]: ' + self.str_word)
                        self.word_idx = 0
            else:
                if self.str_word not in embedding:
                    lower = str.lower(self.str_word)
                    if lower in embedding:
                        self.word_idx = word_alphabet.get_idx(lower)
                    elif lower not in embedding:
                        print('[UNK]: ' + self.str_word)