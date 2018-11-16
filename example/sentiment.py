__author__ = 'Ehaschia'

import argparse, sys, os
# sys.path.append('/home/ehaschia/Code/bi-tree-lstm-crf')


import time

import torch.optim as optim
from tqdm import tqdm

import module.module_io.utils as utils
from module.module_io.alphabet import Alphabet
from module.module_io.logger import *
from module.module_io.sst_data import *
from module.nn.tree_lstm import *
from tensorboardX import SummaryWriter
from module.util import detect_nan

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
                                                 'BiCRFBiTreeLSTM', 'LVeGTreeLSTM', 'LVeGBiTreeLSTM', 'BiCRFTreeLSTM'],
                        help='architecture of model', required=True)
    parser.add_argument('--pred_mode', choices=['single_h', 'avg_h', 'avg_seq_h'],
                        required=True, help='prediction layer mode')
    parser.add_argument('--pred_dense_layer', action='store_true', help='dense_layer before predict')
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
    parser.add_argument('--tensorboard', action='store_true')
    parser.add_argument('--td_name', type=str, default='default', help='the name of this test')
    parser.add_argument('--td_dir', type=str, required=True)
    parser.add_argument('--root_acc', choices=['fine_phase', 'fine_sents', 'bin_phase', 'bin_sents'],
                        help='whether update of root or phase.')
    parser.add_argument('--attention', action='store_true')
    parser.add_argument('--coattention_dim', type=int, default=150)
    parser.add_argument('--elmo', action='store_true')

    # load tree
    args = parser.parse_args()
    print(args)
    logger = get_logger("SSTLogger")

    batch_size = args.batch_size
    embedd_mode = args.embedding
    model_mode = args.model_mode
    pred_dense_layer = args.pred_dense_layer
    leaf_rnn = args.leaf_lstm
    bi_rnn = args.bi_leaf_lstm
    root_acc = args.root_acc
    attention = args.attention
    coattention_dim = args.coattention_dim
    elmo = args.elmo
    if args.tensorboard:
        summary_writer = SummaryWriter(log_dir=args.td_dir + '/' + args.td_name)
        summary_writer.add_text('parameters', str(args))
    else:
        summary_writer = None

    def add_scalar_summary(summary_writer, name, value, step):
        if summary_writer is None:
            return
        if torch.is_tensor(value):
            value = value.item()
        summary_writer.add_scalar(tag=name, scalar_value=value,
                                  global_step=step)

    # alphabet
    # TODO alphabet save
    word_alphabet = Alphabet('word', default_value=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Read data
    logger.info("Reading Data")

    myrandom = Random(48)

    train_dataset = read_sst_data(args.train, word_alphabet, random=myrandom, merge=True)
    dev_dataset = read_sst_data(args.dev, word_alphabet, random=myrandom, merge=True)
    test_dataset = read_sst_data(args.test, word_alphabet, random=myrandom, merge=True)

    # close word_alphabet
    word_alphabet.close()
    logger.info("Word Alphabet Size: %d" % word_alphabet.size())
    logger.info("Loading Embedding")

    # load embedding
    if embedd_mode == 'random' or elmo:
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
    if elmo:
        word_table = None
    else:
        word_table = construct_word_embedding_table()
    embedd_dict = None
    logger.info("constructing network...")
    if model_mode == 'TreeLSTM':
        network = TreeLstm(args.tree_mode, args.leaf_rnn_mode, args.pred_mode, embedd_dim, word_alphabet.size(),
                           args.hidden_size, args.hidden_size, args.softmax_dim, args.leaf_rnn_num, args.num_labels,
                           embedd_word=word_table, p_in=args.p_in, p_leaf=args.p_leaf, p_tree=args.p_tree,
                           p_pred=args.p_pred, leaf_rnn=leaf_rnn, bi_leaf_rnn=bi_rnn, device=device,
                           pred_dense_layer=pred_dense_layer, attention=attention, coattention_dim=coattention_dim,
                           elmo=elmo).to(device)
    elif model_mode == 'BiTreeLSTM':
        network = BiTreeLstm(args.tree_mode, args.leaf_rnn_mode, args.pred_mode, embedd_dim, word_alphabet.size(),
                             args.hidden_size, args.hidden_size, args.softmax_dim, args.leaf_rnn_num, args.num_labels,
                             embedd_word=word_table, p_in=args.p_in, p_leaf=args.p_leaf, p_tree=args.p_tree,
                             p_pred=args.p_pred, leaf_rnn=leaf_rnn, bi_leaf_rnn=bi_rnn, device=device,
                             pred_dense_layer=pred_dense_layer, attention=attention, coattention_dim=coattention_dim,
                             elmo=elmo).to(device)
    elif model_mode == 'CRFTreeLSTM':
        network = CRFTreeLstm(args.tree_mode, args.leaf_rnn_mode, args.pred_mode, embedd_dim, word_alphabet.size(),
                              args.hidden_size, args.hidden_size, args.softmax_dim, args.leaf_rnn_num, args.num_labels,
                              embedd_word=word_table, p_in=args.p_in, p_leaf=args.p_leaf, p_tree=args.p_tree,
                              p_pred=args.p_pred, leaf_rnn=leaf_rnn, bi_leaf_rnn=bi_rnn, device=device,
                              pred_dense_layer=pred_dense_layer, attention=attention, coattention_dim=coattention_dim,
                              elmo=elmo).to(device)
    elif model_mode == 'CRFBiTreeLSTM':
        network = CRFBiTreeLstm(args.tree_mode, args.leaf_rnn_mode, args.pred_mode, embedd_dim, word_alphabet.size(),
                                args.hidden_size, args.hidden_size, args.softmax_dim, args.leaf_rnn_num,
                                args.num_labels, embedd_word=word_table, p_in=args.p_in, p_leaf=args.p_leaf,
                                p_tree=args.p_tree, p_pred=args.p_pred, leaf_rnn=leaf_rnn, bi_leaf_rnn=bi_rnn,
                                device=device, pred_dense_layer=pred_dense_layer, attention=attention,
                                coattention_dim=coattention_dim, elmo=elmo).to(device)
    elif model_mode == 'LVeGTreeLSTM':
        network = LVeGTreeLstm(args.tree_mode, args.leaf_rnn_mode, args.pred_mode, embedd_dim, word_alphabet.size(),
                               args.hidden_size, args.hidden_size, args.softmax_dim, args.leaf_rnn_num,
                               args.num_labels, embedd_word=word_table, p_in=args.p_in, p_leaf=args.p_leaf,
                               p_tree=args.p_tree, p_pred=args.p_pred, leaf_rnn=leaf_rnn, bi_leaf_rnn=bi_rnn,
                               device=device, comp=args.lveg_comp, g_dim=args.gaussian_dim,
                               pred_dense_layer=pred_dense_layer, attention=attention,
                               coattention_dim=coattention_dim, elmo=elmo).to(device)
    elif model_mode == 'LVeGBiTreeLSTM':
        network = LVeGBiTreeLstm(args.tree_mode, args.leaf_rnn_mode, args.pred_mode, embedd_dim, word_alphabet.size(),
                                 args.hidden_size, args.hidden_size, args.softmax_dim, args.leaf_rnn_num,
                                 args.num_labels, embedd_word=word_table, p_in=args.p_in, p_leaf=args.p_leaf,
                                 p_tree=args.p_tree, p_pred=args.p_pred, leaf_rnn=leaf_rnn, bi_leaf_rnn=bi_rnn,
                                 device=device, comp=args.lveg_comp, g_dim=args.gaussian_dim, attention=attention,
                                 coattention_dim=coattention_dim, elmo=elmo).to(device)
    elif model_mode == 'BiCRFBiTreeLSTM':
        network = BiCRFBiTreeLstm(args.tree_mode, args.leaf_rnn_mode, args.pred_mode, embedd_dim, word_alphabet.size(),
                                  args.hidden_size, args.hidden_size, args.softmax_dim, args.leaf_rnn_num,
                                  args.num_labels, embedd_word=word_table, p_in=args.p_in, p_leaf=args.p_leaf,
                                  p_tree=args.p_tree, p_pred=args.p_pred, leaf_rnn=leaf_rnn, bi_leaf_rnn=bi_rnn,
                                  device=device, pred_dense_layer=pred_dense_layer, attention=attention,
                                  coattention_dim=coattention_dim, elmo=elmo).to(device)
    elif model_mode == 'BiCRFTreeLSTM':
        network = BiCRFTreeLstm(args.tree_mode, args.leaf_rnn_mode, args.pred_mode, embedd_dim, word_alphabet.size(),
                                args.hidden_size, args.hidden_size, args.softmax_dim, args.leaf_rnn_num,
                                args.num_labels, embedd_word=word_table, p_in=args.p_in, p_leaf=args.p_leaf,
                                p_tree=args.p_tree, p_pred=args.p_pred, leaf_rnn=leaf_rnn, bi_leaf_rnn=bi_rnn,
                                device=device, pred_dense_layer=pred_dense_layer, attention=attention,
                                coattention_dim=coattention_dim, elmo=elmo).to(device)
    else:
        raise NotImplementedError

    optim_method = args.optim_method
    learning_rate = args.learning_rate
    lr = learning_rate
    momentum = args.momentum
    decay_rate = args.decay_rate
    gamma = args.gamma
    schedule = args.schedule

    # optim init
    if optim_method == 'SGD':
        optimizer = optim.SGD(network.parameters(), lr=lr, momentum=momentum, weight_decay=gamma, nesterov=True)
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

    dev_p_correct = 0.0
    dev_s_correct = 0.0
    dev_bin_p_correct = 0.0
    dev_bin_s_correct = 0.0
    best_epoch = 0
    test_p_correct = 0.0
    test_s_correct = 0.0
    test_p_total = 0
    test_bin_s_total = 0.0
    test_bin_p_total = 0.0
    test_bin_p_correct = 0.0
    test_bin_s_correct = 0.0
    for epoch in range(1, args.epoch + 1):
        train_dataset.shuffle()

        print('Epoch %d (optim_method=%s, learning rate=%.4f, decay rate=%.4f (schedule=%d)): ' % (
            epoch, optim_method, lr, decay_rate, schedule))
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
                for learned_tree in forest:
                    learned_tree.clean()
                forest = []

        optimizer.step()
        optimizer.zero_grad()
        for learned_tree in forest:
            learned_tree.clean()
        train_time = time.time() - start_time

        time.sleep(0.5)

        logger.info('train: %d/%d loss: %.4f, time used : %.2fs' % (
            epoch, args.epoch, train_err / len(train_dataset), train_time))

        add_scalar_summary(summary_writer, 'train/loss', train_err / len(train_dataset), epoch)

        network.eval()
        dev_s_corr = 0.0
        dev_p_corr = 0.0
        dev_s_total = len(dev_dataset)
        dev_p_total = 0
        dev_bin_p_total = 0.0
        dev_bin_s_total = 0.0
        dev_bin_p_corr = 0.0
        dev_bin_s_corr = 0.0
        for i in tqdm(range(len(dev_dataset))):
            tree = dev_dataset[i]
            p_corr, preds, bin_corr, bin_preds, bin_mask = network.predict(tree)

            dev_p_total += preds.size()[0]
            dev_p_corr += p_corr.sum().item()
            dev_s_corr += p_corr[-1].item()

            if tree.label != 2:
                dev_bin_p_corr += bin_corr.sum().item()
                dev_bin_p_total += bin_mask.sum().item()
                dev_bin_s_corr += bin_corr[-1].item()
                dev_bin_s_total += 1.0

            tree.clean()

        time.sleep(1)

        add_scalar_summary(summary_writer, 'dev/fine phase acc', (dev_p_corr * 100 / dev_p_total), epoch)
        add_scalar_summary(summary_writer, 'dev/fine sents acc', (dev_s_corr * 100 / dev_s_total), epoch)
        add_scalar_summary(summary_writer, 'dev/binary phase acc', (dev_bin_p_corr * 100 / dev_bin_p_total), epoch)
        add_scalar_summary(summary_writer, 'dev/binary sents acc', (dev_bin_s_corr * 100 / dev_bin_s_total), epoch)

        print('dev phase acc: %.2f%%, dev sents acc: %.2f%%, binary phase acc: %.2f%%, dev sents acc: %.2f%%'
              % (dev_p_corr * 100 / dev_p_total, dev_s_corr * 100 / dev_s_total,
                 dev_bin_p_corr * 100 / dev_bin_p_total, dev_bin_s_corr * 100 / dev_bin_s_total))

        if root_acc == 'fine_sents':
            update = True if dev_s_corr > dev_s_correct else False
        elif root_acc == 'fine_phase':
            update = True if dev_p_corr > dev_p_correct else False
        elif root_acc == 'bin_phase':
            update = True if dev_bin_p_corr > dev_bin_p_correct else False
        elif root_acc == 'bin_sents':
            update = True if dev_bin_s_corr > dev_bin_s_correct else False
        else:
            raise NotImplementedError
        # if dev_s_corr > dev_s_correct:

        if update:
            dev_p_correct = dev_p_corr
            dev_s_correct = dev_s_corr
            dev_bin_p_correct = dev_bin_p_corr
            dev_bin_s_correct = dev_bin_s_corr
            best_epoch = epoch
            test_p_correct = 0.0
            test_s_correct = 0.0
            test_p_total = 0
            test_bin_s_total = 0.0
            test_bin_p_total = 0.0
            test_bin_p_correct = 0.0
            test_bin_s_correct = 0.0

            time.sleep(1)

            for i in tqdm(range(len(test_dataset))):
                tree = test_dataset[i]
                p_corr, preds, bin_corr, bin_preds, bin_mask = network.predict(tree)

                test_p_total += preds.size()[0]
                test_p_correct += p_corr.sum().item()
                test_s_correct += p_corr[-1].item()

                if tree.label != 2:
                    test_bin_p_correct += bin_corr.sum().item()
                    test_bin_p_total += bin_mask.sum().item()
                    test_bin_s_correct += bin_corr[-1].item()
                    test_bin_s_total += 1.0

                tree.clean()

            time.sleep(1)

            print('test phase acc: %.2f%%, test sents acc: %.2f%%, binary phase acc: %.2f%%, binary sents acc: %.2f%%'
                  % (test_p_correct * 100 / test_p_total, test_s_correct * 100 / len(test_dataset),
                     test_bin_p_correct * 100 / test_bin_p_total, test_bin_s_correct * 100 / test_bin_s_total))

        print("best dev phase acc: %.2f%%, sents acc: %.2f%%, binary phase acc: %.2f%%, sents acc: %.2f%% (epoch: %d)" % (
            dev_p_correct * 100 / dev_p_total, dev_s_correct * 100 / dev_s_total,
            dev_bin_p_correct *100 / dev_bin_p_total, dev_bin_s_correct * 100 / dev_bin_s_total,
            best_epoch))
        print("best tst phase corr: %.2f%%, sents acc: %.2f%%, binary phase acc: %.2f%%, sents acc: %.2f%% (epoch: %d)" % (
            test_p_correct * 100 / test_p_total, test_s_correct * 100 / len(test_dataset),
            test_bin_p_correct * 100 / test_bin_p_total,  test_bin_s_correct * 100 / test_bin_s_total,
            best_epoch))

        add_scalar_summary(summary_writer, 'test/fine phase acc', (test_p_correct * 100 / test_p_total), epoch)
        add_scalar_summary(summary_writer, 'test/fine sents acc', (test_s_correct * 100 / len(test_dataset)), epoch)
        add_scalar_summary(summary_writer, 'test/binary phase acc', (test_bin_p_correct * 100 / test_bin_p_total), epoch)
        add_scalar_summary(summary_writer, 'test/binary sents acc', (test_bin_s_correct * 100 / test_bin_s_total), epoch)

        if optim_method == "SGD" and epoch % schedule == 0:
            lr = learning_rate / (1.0 + epoch * decay_rate)
            optimizer = optim.SGD(network.parameters(), lr=lr, momentum=momentum, weight_decay=gamma, nesterov=True)

    if args.tensorboard:
        summary_writer.close()
    else:
        pass

if __name__ == '__main__':
    torch.manual_seed(48)
    np.random.seed(48)
    main()
