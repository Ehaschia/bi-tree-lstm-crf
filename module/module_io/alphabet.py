__author__ = 'Ehaschia'

from .logger import get_logger
import json
import os


class Alphabet(object):
    def __init__(self, name, default_value=True, keep_growing=True):
        self.__name = name
        self.instance2index = {}
        self.instances = []
        self.default_value = default_value
        self.default_index = None
        if self.default_value:
            self.instance2index['<_UNK>'] = 0
            self.instances.append('<_UNK>')
            self.default_index = 0
        self.keep_growing = keep_growing
        self.counter = {}

        self.logger = get_logger("Alphabet")

    def add(self, instance):
        self.instances.append(instance)
        self.instance2index[instance] = len(self.instance2index)
        self.counter[instance] = 1

    def get_idx(self, instance):
        if instance in self.instance2index:
            self.counter[instance] += 1
            return self.instance2index[instance]
        else:
            if self.keep_growing:
                self.add(instance)
                return self.instance2index[instance]
            elif self.default_value:
                return self.default_index
            else:
                raise KeyError("instance not found: " + instance)

    def get_instance(self, index):
        if self.default_value and index == self.default_index:
            return "<_UNK>"
        else:
            try:
                return self.instances[index]
            except IndexError:
                raise IndexError("unknown index: " + str(index))

    def size(self):
        return len(self.instances)

    def items(self):
        return self.instance2index.items()

    def open(self):
        self.keep_growing = True

    def close(self):
        # self.replace_bruckets()
        self.keep_growing = False

    def get_content(self):
        # alert not save default_value may cause error?
        return {"instance2index": self.instance2index,
                "instances": self.instances,
                "counter": self.counter}

    def count(self, word):
        if word in self.counter:
            return self.counter[word]
        else:
            return 0

    def save(self, out_directory, name=None):
        saving_name = name if name else self.__name
        try:
            if not os.path.exists(out_directory):
                os.makedirs(out_directory)
            json.dump(self.get_content(), open(os.path.join(out_directory, saving_name + '.json'), 'w'), indent=4)
        except Exception as e:
            self.logger.warn("Alphabet is not saved: %s" % repr(e))

    # def replace_bruckets(self):
    #     lrb_idx = self.instance2index['-LRB-']
    #     rrb_idx = self.instance2index['-RRB-']
    #     lrb_cnt = self.counter['-LRB-']
    #     rrb_cnt = self.counter['-RRB-']
    #     del self.instance2index['-LRB-']
    #     del self.instance2index['-RRB-']
    #     del self.counter['-LRB-']
    #     del self.counter['-RRB-']
    #     self.instance2index['('] = lrb_idx
    #     self.instance2index[')'] = rrb_idx
    #     self.counter['('] = lrb_cnt
    #     self.counter[')'] = rrb_cnt
    #     self.instances[lrb_idx] = '('
    #     self.instances[rrb_idx] = ')'

    def __from_json(self, data):
        self.instances = data['instances']
        self.instance2index = data['instance2index']
        self.counter = data['counter']

    def load(self, input_directory, name=None):
        loading_name = name if name else self.__name
        self.__from_json(json.load(open(os.path.join(input_directory, loading_name + '.json'))))
        self.keep_growing = False
