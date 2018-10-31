__author__ = 'Ehaschia'

import argparse, sys, os
from pathlib import Path

current_path = os.path.realpath(__file__)
print(current_path)
sys.path.append(Path(current_path).parent.parent)
import time

import torch.optim as optim
from tqdm import tqdm

import module.module_io.utils as utils
from module.module_io.alphabet import Alphabet
from module.module_io.logger import *
from module.module_io.sst_data import *
from module.nn.tree_lstm import *


def main():
    parser = argparse.ArgumentParser(description='Tuning with bi-directional Tree-LSTM-CRF')
    parser.add_argument('--leaf_lstm', action='store_true', help='use leaf_lstm or not')
    parser.add_argument('--bi_leaf_lstm', action='store_true', help='use bi-directional leaf_lstm or not')
    parser.add_argument('--leaf_rnn_mode', choices=['RNN', 'GRU', 'LSTM'], help='architecture of leaf rnn',
                        required=True)
    parser.add_argument('--leaf_rnn_num', type=int, default=1, help='layer of leaf lstm')
    parser.add_argument('--tree_mode', choices=['SLSTM', 'TreeLSTM', 'BUTreeLSTM'],
                        help='architecture of tree lstm', required=True)
    parser.add_argument('--model_mode', choices=['TreeLSTM', 'BiTreeLSTM', 'CRFTreeLSTM', 'CRFBiTreeLSTM',
                                                 'BiCRFBiTreeLSTM', 'LVeGTreeLSTM', 'LVeGBiTreeLSTM'],
                        help='architecture of model', required=True)
    parser.add_argument('--pred_mode', choices=['single_h', 'avg_h', 'avg_seq_h'],
                        required=True, help='prediction layer mode')
    parser.add_argument('--batch_size', type=int, default=16, help='Number of batch')
    parser.add_argument('--epoch', type=int, default=50, help='run epoch')
    parser.add_argument('--hidden_size', type=int, default=150, help='Number of hidden units in tree structure')
    parser.add_argument('--softmax_dim', type=int, default=64)
    parser.add_argument('--optim_method', choices=['SGD', 'Adadelta', 'Adagrad', 'Adam', 'RMSprop'],
                        help='optimaize method')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum factor')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='Decay rate of learning rate')
    parser.add_argument('--gamma', type=float, default=0.0, help='weight for regularization')
    parser.add_argument('--schedule', type=int, help='schedule for learning rate decay')

    parser.add_argument('--embedding', choices=['glove', 'senna', 'sskip', 'polyglot', 'random'],
                        help='Embedding for words', required=True)
    parser.add_argument('--embedding_path', help='path for embedding dict')
    parser.add_argument('--train', type=str, default='/home/ehaschia/Code/dataset/sst/trees/train.txt')
    parser.add_argument('--dev', type=str, default='/home/ehaschia/Code/dataset/sst/trees/dev.txt')
    parser.add_argument('--test', type=str, default='/home/ehaschia/Code/dataset/sst/trees/test.txt')
    parser.add_argument('--num_labels', type=int, default=5)
    parser.add_argument('--p_in', type=float, default=0.5, help="Dropout prob for embedding")
    parser.add_argument('--p_leaf', type=float, default=0.5, help='Dropout prob for tree lstm input')
    parser.add_argument('--p_tree', type=float, default=0.5, help='Dropout prob in tree lstm node')
    parser.add_argument('--p_pred', type=float, default=0.5, help='Dropout prob for pred layer')
    parser.add_argument('--lveg_comp', type=int, default=1, help='the component number of mixture gaussian in LVeG')
    parser.add_argument('--gaussian_dim', type=int, default=1, help='the gaussian dim in LVeG')

    # load tree
    args = parser.parse_args()

    logger = get_logger("SSTLogger")

    batch_size = args.batch_size
    embedd_mode = args.embedding
    model_mode = args.model_mode
    # alphabet
    # TODO alphabet save
    word_alphabet = Alphabet('word', default_value=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Read data
    logger.info("Reading Data")

    myrandom = Random(48)

    train_dataset = read_sst_data(args.train, word_alphabet, random=myrandom)
    dev_dataset = read_sst_data(args.dev, word_alphabet, random=myrandom)
    test_dataset = read_sst_data(args.test, word_alphabet, random=myrandom)

    # TODO ugly, fix here
    train_dataset.merge_data()
    dev_dataset.merge_data()
    test_dataset.merge_data()

    # close word_alphabet
    word_alphabet.close()
    logger.info("Word Alphabet Size: %d" % word_alphabet.size())
    logger.info("Loading Embedding")

    # load embedding
    if embedd_mode == 'random':
        embedd_dim = 300
        embedd_dict = None
    else:
        embedd_dict, embedd_dim = utils.load_embedding_dict(embedd_mode, args.embedding_path)

    def construct_word_embedding_table():
        scale = np.sqrt(3.0 / embedd_dim)
        table = np.empty([word_alphabet.size(), embedd_dim], dtype=np.float32)
        oov = 0
        if embedd_mode == 'random':
            table = np.random.uniform(-scale, scale, [word_alphabet.size(), embedd_dim]).astype(dtype=np.float32)
        else:
            for word, index in word_alphabet.items():
                if word in embedd_dict:
                    a_embedding = embedd_dict[word]
                elif word.lower() in embedd_dict:
                    a_embedding = embedd_dict[word.lower()]
                else:
                    a_embedding = np.random.uniform(-scale, scale, [1, embedd_dim]).astype(np.float32)
                    oov += 1
                table[index, :] = a_embedding
            print('oov: %d' % oov)
        return torch.from_numpy(table)

    word_table = construct_word_embedding_table()
    logger.info("constructing network...")
    if model_mode == 'TreeLSTM':
        network = TreeLstm(args.tree_mode, args.leaf_rnn_mode, args.pred_mode, embedd_dim, word_alphabet.size(),
                           args.hidden_size, args.hidden_size, args.softmax_dim, args.leaf_rnn_num, args.num_labels,
                           embedd_word=word_table, p_in=args.p_in, p_leaf=args.p_leaf, p_tree=args.p_tree,
                           p_pred=args.p_pred, leaf_rnn=True, bi_leaf_rnn=True, device=device).to(device)
    elif model_mode == 'BiTreeLSTM':
        network = BiTreeLstm(args.tree_mode, args.leaf_rnn_mode, args.pred_mode, embedd_dim, word_alphabet.size(),
                             args.hidden_size, args.hidden_size, args.softmax_dim, args.leaf_rnn_num, args.num_labels,
                             embedd_word=word_table, p_in=args.p_in, p_leaf=args.p_leaf, p_tree=args.p_tree,
                             p_pred=args.p_pred, leaf_rnn=True, bi_leaf_rnn=True, device=device).to(device)
    elif model_mode == 'CRFTreeLSTM':
        network = CRFTreeLstm(args.tree_mode, args.leaf_rnn_mode, args.pred_mode, embedd_dim, word_alphabet.size(),
                              args.hidden_size, args.hidden_size, args.softmax_dim, args.leaf_rnn_num, args.num_labels,
                              embedd_word=word_table, p_in=args.p_in, p_leaf=args.p_leaf, p_tree=args.p_tree,
                              p_pred=args.p_pred, leaf_rnn=True, bi_leaf_rnn=True, device=device).to(device)
    elif model_mode == 'CRFBiTreeLSTM':
        network = CRFBiTreeLstm(args.tree_mode, args.leaf_rnn_mode, args.pred_mode, embedd_dim, word_alphabet.size(),
                                args.hidden_size, args.hidden_size, args.softmax_dim, args.leaf_rnn_num,
                                args.num_labels, embedd_word=word_table, p_in=args.p_in, p_leaf=args.p_leaf,
                                p_tree=args.p_tree, p_pred=args.p_pred, leaf_rnn=True, bi_leaf_rnn=True,
                                device=device).to(device)
    elif model_mode == 'LVeGTreeLSTM':
        network = LVeGTreeLstm(args.tree_mode, args.leaf_rnn_mode, args.pred_mode, embedd_dim, word_alphabet.size(),
                               args.hidden_size, args.hidden_size, args.softmax_dim, args.leaf_rnn_num,
                               args.num_labels, embedd_word=word_table, p_in=args.p_in, p_leaf=args.p_leaf,
                               p_tree=args.p_tree, p_pred=args.p_pred, leaf_rnn=True, bi_leaf_rnn=True,
                               device=device, comp=args.lveg_comp, g_dim=args.gaussian_dim).to(device)
    elif model_mode == 'LVeGBiTreeLSTM':
        network = LVeGBiTreeLstm(args.tree_mode, args.leaf_rnn_mode, args.pred_mode, embedd_dim, word_alphabet.size(),
                                 args.hidden_size, args.hidden_size, args.softmax_dim, args.leaf_rnn_num,
                                 args.num_labels, embedd_word=word_table, p_in=args.p_in, p_leaf=args.p_leaf,
                                 p_tree=args.p_tree, p_pred=args.p_pred, leaf_rnn=True, bi_leaf_rnn=True,
                                 device=device, comp=args.lveg_comp, g_dim=args.gaussian_dim).to(device)
    elif model_mode == 'BiCRFBiTreeLSTM':
        network = BiCRFBiTreeLstm(args.tree_mode, args.leaf_rnn_mode, args.pred_mode, embedd_dim, word_alphabet.size(),
                                  args.hidden_size, args.hidden_size, args.softmax_dim, args.leaf_rnn_num,
                                  args.num_labels, embedd_word=word_table, p_in=args.p_in, p_leaf=args.p_leaf,
                                  p_tree=args.p_tree, p_pred=args.p_pred, leaf_rnn=True, bi_leaf_rnn=True,
                                  device=device).to(device)
    else:
        raise NotImplementedError

    optim_method = args.optim_method
    lr = args.learning_rate
    momentum = args.momentum
    decay_rate = args.decay_rate
    gamma = args.gamma
    schedule = args.schedule

    # optim init
    if optim_method == 'SGD':
        optimizer = optim.SGD(network.parameters(), lr=lr, momentum=momentum, weight_decay=gamma)
    elif optim_method == 'Adam':
        # default lr is 0.001
        optimizer = optim.Adam(network.parameters(), lr=lr, weight_decay=gamma)
    elif optim_method == 'Adadelta':
        # default lr is 1.0
        optimizer = optim.Adadelta(network.parameters(), lr=lr, weight_decay=gamma)
    elif optim_method == 'Adagrad':
        # default lr is 0.01
        optimizer = optim.Adagrad(network.parameters(), lr=lr, weight_decay=gamma)
    elif optim_method == 'RMSprop':
        # default lr is 0.01
        optimizer = optim.RMSprop(network.parameters(), lr=lr, weight_decay=gamma, momentum=momentum)
    else:
        raise NotImplementedError("Not Implement optim Method: " + optim_method)
    logger.info("Optim mode: " + optim_method)

    # alert here best for max p correct
    dev_p_correct = 0.0
    dev_s_correct = 0.0
    best_epoch = 0
    test_p_correct = 0.0
    test_s_correct = 0.0
    test_p_total = 0

    for epoch in range(args.epoch):
        train_dataset.shuffle()

        print('Epoch %d (learning rate=%.4f, decay rate=%.4f (schedule=%d)): ' % (
            epoch, lr, decay_rate, schedule))
        time.sleep(1)
        start_time = time.time()
        train_err = 0.0
        train_p_total = 0.0

        network.train()
        optimizer.zero_grad()
        forest = []
        for i in tqdm(range(len(train_dataset))):
            tree = train_dataset[i]
            forest.append(tree)
            loss = network.loss(tree)
            a_tree_p_cnt = 2 * tree.length + 1
            loss.backward()

            train_err += loss.item()
            train_p_total += a_tree_p_cnt
            if i % batch_size == 0 and i != 0:
                optimizer.step()
                optimizer.zero_grad()
                for dealed_tree in forest:
                    dealed_tree.clean()
        train_time = time.time() - start_time

        time.sleep(1)

        logger.info('train: %d/%d loss: %.4f, time used : %.2fs' % (
            epoch + 1, args.epoch, train_err / len(train_dataset), train_time))

        network.eval()
        dev_s_corr = 0.0
        dev_p_corr = 0.0
        dev_s_total = len(dev_dataset)
        dev_p_total = 0
        for i in tqdm(range(len(dev_dataset))):
            tree = dev_dataset[i]
            p_corr, preds = network.predict(tree)

            dev_p_total += preds.size()[0]
            dev_p_corr += p_corr.sum().item()
            dev_s_corr += p_corr[-1].item()

            tree.clean()

        time.sleep(1)

        print('dev phase corr: %d, dev phase total: %d, dev phase acc: %.2f%%' % (
            dev_p_corr, dev_p_total, dev_p_corr * 100 / dev_p_total))
        print('dev sents corr: %d, dev sents total: %d, dev sents acc: %.2f%%' % (
            dev_s_corr, dev_s_total, dev_s_corr * 100 / dev_s_total))

        if dev_p_corr > dev_p_correct:
            dev_p_correct = dev_p_corr
            dev_s_correct = dev_s_corr
            best_epoch = epoch
            test_p_correct = 0.0
            test_s_correct = 0.0
            test_p_total = 0

            time.sleep(1)

            for i in tqdm(range(len(test_dataset))):
                tree = test_dataset[i]
                p_corr, preds = network.predict(tree)

                test_p_total += preds.size()[0]
                test_p_correct += p_corr.sum().item()
                test_s_correct += p_corr[-1].item()

                tree.clean()

            time.sleep(1)

            print('test phase corr: %d, test phase total: %d, test phase acc: %.2f%%' % (
                test_p_correct, test_p_total, test_p_correct * 100 / test_p_total))
            print('test sents corr: %d, test sents total: %d, test sents acc: %.2f%%' % (
                test_s_correct, len(test_dataset), test_s_correct * 100 / len(test_dataset)))

        print("best phase dev corr: %d, phase total: %d, phase acc: %.2f%% (epoch: %d)" % (
            dev_p_correct, dev_p_total, dev_p_correct * 100 / dev_p_total, best_epoch))
        print("best sents dev corr: %d, sents total: %d, sents acc: %.2f%% (epoch: %d)" % (
            dev_s_correct, dev_s_total, dev_s_correct * 100 / dev_s_total, best_epoch))
        print("best phase test corr: %d, phase total: %d, phase acc: %.2f%% (epoch: %d)" % (
            test_p_correct, test_p_total, test_p_correct * 100 / test_p_total, best_epoch))
        print("best sents test corr: %d, sents total: %d, sents acc: %.2f%% (epoch: %d)" % (
            test_s_correct, len(test_dataset), test_s_correct * 100 / len(test_dataset), best_epoch))
        # TODO Implement decay_rate and schedule


if __name__ == '__main__':
    torch.manual_seed(48)
    np.random.seed(48)
    main()
