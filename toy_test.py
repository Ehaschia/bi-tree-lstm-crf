import numpy as np
import torch

from module.module_io.tree import Tree
from module.nn.crf import LowTreeCRF


class A(object):

    def __init__(self):
        pass

    def func_a(self, x):
        return x + 1

    def func_b(self, x):
        return self.func_a(x) + 1

    def func_c(self, x):
        return self.func_b(x) + 1


class B(A):

    def __init__(self):
        super(A, self).__init__()

    def func_a(self, x):
        return x + 2

    def func_c(self, x):
        return self.func_b(x) + 2


def main():
    # alert crf model debug
    # tree = Tree(1)
    # left_child = Tree(0)
    # child1 = Tree(0, word=1)
    # child2 = Tree(0, word=2)
    # child3 = Tree(1, word=3)
    #
    # left_child.add_child(child1)
    # left_child.add_child(child2)
    # tree.add_child(left_child)
    # tree.add_child(child3)
    #
    # child1.crf_cache['emission_score'] = torch.from_numpy(np.array([1.0, 0.0]))
    # child2.crf_cache['emission_score'] = torch.from_numpy(np.array([2.0, 0.0]))
    # child3.crf_cache['emission_score'] = torch.from_numpy(np.array([0.0, 3.0]))
    # left_child.crf_cache['emission_score'] = torch.from_numpy(np.array([1.0, 0.0]))
    # tree.crf_cache['emission_score'] = torch.from_numpy(np.array([0.0, 1.0]))
    # trans_matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
    # crf = LowTreeCRF(2, trans_matrix)
    # inside_score = crf.loss(tree)
    # pred = crf.predict(tree)
    # print(inside_score.item())
    # print(pred)
    model_b = B()
    x = model_b.func_c(0)
    print(x)


if __name__ == '__main__':
    main()
