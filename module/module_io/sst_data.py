import torch
from torch.utils.data import Dataset
from module.module_io.tree import Tree
import numpy as np
from random import Random

_buckets = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 140]
UNK_ID = 0
PAD_ID_WORD = 0
PAD_ID_TAG = 0


class SSTDataset(Dataset):

    def __len__(self):
        if self.length is not None:
            return self.length
        elif len(self.bucket_sizes) > 0:
            self.length = sum(self.bucket_sizes)
        else:
            for i in self.data:
                self.bucket_sizes.append(len(i))
                self.length = sum(self.bucket_sizes)

    def __init__(self, word_alphabet, random=None):
        self.data = [[] for _ in _buckets]
        self.full_data = []
        self.__word_alphabet = word_alphabet
        self.keep_growing = True
        self.bucket_sizes = []
        self.length = None
        self.tensor_data = []
        if random is None:
            self.random = Random(48)
        else:
            self.random = random

    def __getitem__(self, index):
        if len(self.full_data) > 0:
            return self.full_data[index]
        else:
            for bucket_idx, bucket_size in enumerate(self.bucket_sizes):
                if index < bucket_size:
                    return self.data[bucket_idx][index]
                else:
                    index -= bucket_size
            raise IndexError("list index out of range")

    def add_tree(self, tree):
        # alert may refresh the length every add
        if not self.keep_growing:
            raise UnboundLocalError("Dataset grow is Closed")
        tree_len = len(tree)
        for idx, i in enumerate(_buckets):
            if tree_len < i:
                self.data[idx].append(tree)
                break

    def close(self):
        self.keep_growing = False
        for i in self.data:
            self.bucket_sizes.append(len(i))
        self.length = sum(self.bucket_sizes)

    def get_batch_variable(self, batch_size, unk_replace=0.):
        # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
        # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
        # the size if i-th training bucket, as used later.

        buckets_scale = [sum(self.bucket_sizes[:i + 1]) / float(self.length) for i in range(len(self.bucket_sizes))]
        random_number = np.random.random_sample()
        bucket_id = min([i for i in range(len(buckets_scale)) if buckets_scale[i] > random_number])
        bucket_length = self.bucket_sizes[bucket_id]

        batch_size = min(bucket_length, batch_size)
        index = torch.randperm(bucket_length)[:batch_size].tolist()

        if self.tensor_data[bucket_id][0].is_cuda:
            index = index.cuda()
        batch_words = self.tensor_data[bucket_id][0][index]
        batch_labels = self.tensor_data[bucket_id][1][index]
        batch_words_mask = self.tensor_data[bucket_id][2][index]
        batch_labels_mask = self.tensor_data[bucket_id][3][index]

        return batch_words, batch_labels, batch_words_mask, batch_labels_mask

    def expand_label(self, tree, expand_sequnece):
        zero_label = np.zeros(expand_sequnece * (expand_sequnece + 1) // 2)
        zero_mask = np.zeros(expand_sequnece * (expand_sequnece + 1) // 2)

        node_list = [tree]

        for subtree in node_list:
            # alert should debug here
            node_list += subtree.children
            idx = subtree.sequence_idx(expand_sequnece)
            zero_label[idx] = subtree.label
        return zero_label, zero_mask

    def data_to_variable(self, device):
        # word_pard
        for bucket_id in range((len(_buckets))):
            bucket_size = self.bucket_sizes[bucket_id]
            if bucket_size == 0:
                self.tensor_data.append(1)
            bucket_length = _buckets[bucket_id]
            wid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int32)
            lid_inputs = np.empty([bucket_size, bucket_length * (bucket_length + 1) // 2], dtype=np.int32)

            word_masks = np.zeros([bucket_size, bucket_length])
            labels_masks = np.empty([bucket_size, bucket_length * (bucket_length + 1) // 2], dtype=np.int32)

            lengths = np.empty(bucket_size, dtype=np.int32)
            for idx, a_tree in enumerate(self.data[bucket_id]):
                wids = a_tree.get_yield()
                sent_len = len(wids)
                lengths[idx] = sent_len

                # word_part
                wid_inputs[idx, :sent_len] = wids
                wid_inputs[idx, sent_len:] = PAD_ID_WORD

                lid_inputs[idx], labels_masks[idx] = self.expand_label(a_tree, bucket_length)

                word_masks[idx, :sent_len] = 1.0
            words = torch.from_numpy(wid_inputs)
            labels = torch.from_numpy(lid_inputs)
            word_masks = torch.from_numpy(word_masks)
            labels_masks = torch.from_numpy(labels_masks)
            words = words.to(device)
            labels = labels.to(device)
            word_masks = word_masks.to(device)
            labels_masks = labels_masks.to(device)
            self.tensor_data.append((words, labels, word_masks, labels_masks))

    def merge_data(self):
        for bucket in self.data:
            self.full_data += bucket

    def shuffle(self):
        self.random.shuffle(self.full_data)

    def replace_unk(self, word_alphabet, embedding, isTraining=True):
        if len(self.full_data):
            for tree in self.full_data:
                tree.replace_unk(word_alphabet, embedding, isTraining=isTraining)
        else:
            for bucket in self.data:
                for tree in bucket:
                    tree.replace_unk(word_alphabet, embedding, isTraining=isTraining)

class SSTDataloader(object):
    def __init__(self, file_path, word_alphabet):
        self.__source_file = open(file_path, 'r')
        self.__word_alphabet = word_alphabet

    def close(self):
        self.__source_file.close()

    def getNext(self):
        line = self.__source_file.readline()
        # why the loop
        if line is not None and len(line.strip()) > 0:
            line = line.strip().replace('\\', '')
        else:
            return None
        tree = self.parse_tree(line)
        leaf_nodes = tree.get_yield_node()
        for i in range(len(leaf_nodes)):
            leaf_nodes[i].idx = i
        return tree

    def parse_tree(self, line):
        if line.startswith('(') and line.endswith(')'):
            sub_line = line[1:-1].strip()
            if sub_line.find('(') == -1 and sub_line.find(')') == -1:
                # leaf node
                blank_idx = sub_line.find(' ')
                label, str_word = int(sub_line[:blank_idx]), sub_line[blank_idx + 1:]

                if str_word == '-LRB-':
                    str_word = '('
                if str_word == '-RRB-':
                    str_word = ')'

                idx = self.__word_alphabet.get_idx(str_word)
                return Tree(label, word_idx=idx, str_word=str_word)
            else:
                blank_idx = sub_line.find(' ')
                label, str_children = int(sub_line[:blank_idx]), sub_line[blank_idx + 1:]
                # here is ugly, has some good method?
                counter = 0
                for i in range(len(str_children)):
                    # because only has 2 children
                    if str_children[i] == '(':
                        counter += 1
                    elif str_children[i] == ')':
                        counter -= 1
                    elif counter == 0:
                        break
                    else:
                        pass
                if i != len(str_children) - 1:
                    children = [str_children[:i], str_children[i + 1:]]
                else:
                    children = [str_children]
                children = [self.parse_tree(child) for child in children]
                tree = Tree(label)
                tree.set_children(children)
                return tree
        else:
            raise SyntaxError("Input sentence is not tree structure:" + line)


def read_sst_data(source_path, word_alphabet, max_size=None, random=None, merge=False):
    dataset = SSTDataset(word_alphabet, random=random)
    reader = SSTDataloader(source_path, word_alphabet)

    counter = 0
    a_tree = reader.getNext()
    while a_tree is not None and (not max_size or a_tree.length < max_size):
        counter += 1
        if counter % 5000 == 0:
            print("reading data: %d" % counter)
        dataset.add_tree(a_tree)
        a_tree = reader.getNext()
    reader.close()
    dataset.close()
    print("Total number of data: %d" % counter)
    if merge:
        dataset.merge_data()
    return dataset


def read_sst_data_to_variable(source_path, word_alphabet, device, max_size=None, merge=False):
    # this function is used for batch version code
    dataset = read_sst_data(source_path, word_alphabet, max_size)
    dataset.data_to_variable(device)
    if merge:
        dataset.merge_data()
    return dataset
