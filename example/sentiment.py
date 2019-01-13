__author__ = 'Ehaschia'

import argparse, sys, os

sys.path.append('/home/ehaschia/Code/sentiment/bi-tree-lstm-crf')

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
                                                 'BiCRFBiTreeLSTM', 'LVeGTreeLSTM', 'LVeGBiTreeLSTM', 'BiCRFTreeLSTM',
                                                 'LABiCRFTreeLSTM', 'LACRFBiTreeLSTM'],
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
    parser.add_argument('--schedule', type=int, default=5, help='schedule for learning rate decay')

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
    parser.add_argument('--attention', action='store_true')
    parser.add_argument('--coattention_dim', type=int, default=150)
    parser.add_argument('--elmo', choices=['none', 'only', 'cat'])
    parser.add_argument('--elmo_weight', type=str,
                        default='/home/ehaschia/Code/dataset/elmo/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5')
    parser.add_argument('--elmo_config', type=str,
                        default='/home/ehaschia/Code/dataset/elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json')

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

    attention = args.attention
    coattention_dim = args.coattention_dim
    elmo = args.elmo
    elmo_weight = args.elmo_weight
    elmo_config = args.elmo_config

    num_labels = args.num_labels

    all_cite_version = ['fine_phase', 'fine_sents', 'bin_phase', 'bin_sents',
                        'bin_phase_v2', 'bin_sents_v2', 'full_bin_phase', 'full_bin_phase_v2']

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
    logger.info("Loading Embedding")

    # load embedding
    if embedd_mode == 'random':
        embedd_dim = 300
        embedd_dict = None
    else:
        embedd_dict, embedd_dim = utils.load_embedding_dict(embedd_mode, args.embedding_path)
    to_save_embedding = {}

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
                    to_save_embedding[word] = embedd_dict[word][0]
                # elif word.lower() in embedd_dict:
                #     a_embedding = embedd_dict[word.lower()]
                #     to_save_embedding[word]
                else:
                    a_embedding = np.random.uniform(-scale, scale, [1, embedd_dim]).astype(np.float32)
                    oov += 1
                table[index, :] = a_embedding
            print('oov: %d' % oov)
        return torch.from_numpy(table)

    def save_embedding(path, name):
        if not os.path.exists(path):
            os.makedirs(path)
        with open(path + '/' + name, 'w') as f:
            for word in to_save_embedding.keys():
                str = word
                # embedding = np.array2string(to_save_embedding[word], precision=4, separator=' ')[1:-1]
                f.write(str)
                # f.write(embedding)
                f.write('\n')
        print('Save emebdding to ' + path + '/' + name)

    if num_labels == 3:
        train_dataset.convert_to_3_class()
        dev_dataset.convert_to_3_class()
        test_dataset.convert_to_3_class()

    train_dataset.replace_unk(word_alphabet, embedd_dict, isTraining=True)
    print('DEV UNK')
    dev_dataset.replace_unk(word_alphabet, embedd_dict, isTraining=False)
    print('TEST UNK')
    test_dataset.replace_unk(word_alphabet, embedd_dict, isTraining=False)
    logger.info("Word Alphabet Size: %d" % word_alphabet.size())
    if elmo is 'only':
        word_table = None
    else:
        word_table = construct_word_embedding_table()
    word_alphabet.close()
    embedd_dict = None
    logger.info("constructing network...")
    if model_mode == 'TreeLSTM':
        network = TreeLstm(args.tree_mode, args.leaf_rnn_mode, args.pred_mode, embedd_dim, word_alphabet.size(),
                           args.hidden_size, args.hidden_size, args.softmax_dim, args.leaf_rnn_num, num_labels,
                           embedd_word=word_table, p_in=args.p_in, p_leaf=args.p_leaf, p_tree=args.p_tree,
                           p_pred=args.p_pred, leaf_rnn=leaf_rnn, bi_leaf_rnn=bi_rnn, device=device,
                           pred_dense_layer=pred_dense_layer, attention=attention, coattention_dim=coattention_dim,
                           elmo=elmo, elmo_weight=elmo_weight, elmo_config=elmo_config).to(device)
    elif model_mode == 'BiTreeLSTM':
        network = BiTreeLstm(args.tree_mode, args.leaf_rnn_mode, args.pred_mode, embedd_dim, word_alphabet.size(),
                             args.hidden_size, args.hidden_size, args.softmax_dim, args.leaf_rnn_num, num_labels,
                             embedd_word=word_table, p_in=args.p_in, p_leaf=args.p_leaf, p_tree=args.p_tree,
                             p_pred=args.p_pred, leaf_rnn=leaf_rnn, bi_leaf_rnn=bi_rnn, device=device,
                             pred_dense_layer=pred_dense_layer, attention=attention, coattention_dim=coattention_dim,
                             elmo=elmo, elmo_weight=elmo_weight, elmo_config=elmo_config).to(device)
    elif model_mode == 'CRFTreeLSTM':
        network = CRFTreeLstm(args.tree_mode, args.leaf_rnn_mode, args.pred_mode, embedd_dim, word_alphabet.size(),
                              args.hidden_size, args.hidden_size, args.softmax_dim, args.leaf_rnn_num, num_labels,
                              embedd_word=word_table, p_in=args.p_in, p_leaf=args.p_leaf, p_tree=args.p_tree,
                              p_pred=args.p_pred, leaf_rnn=leaf_rnn, bi_leaf_rnn=bi_rnn, device=device,
                              pred_dense_layer=pred_dense_layer, attention=attention, coattention_dim=coattention_dim,
                              elmo=elmo, elmo_weight=elmo_weight, elmo_config=elmo_config).to(device)
    elif model_mode == 'CRFBiTreeLSTM':
        network = CRFBiTreeLstm(args.tree_mode, args.leaf_rnn_mode, args.pred_mode, embedd_dim, word_alphabet.size(),
                                args.hidden_size, args.hidden_size, args.softmax_dim, args.leaf_rnn_num,
                                num_labels, embedd_word=word_table, p_in=args.p_in, p_leaf=args.p_leaf,
                                p_tree=args.p_tree, p_pred=args.p_pred, leaf_rnn=leaf_rnn, bi_leaf_rnn=bi_rnn,
                                device=device, pred_dense_layer=pred_dense_layer, attention=attention,
                                coattention_dim=coattention_dim, elmo=elmo, elmo_weight=elmo_weight,
                                elmo_config=elmo_config).to(device)
    elif model_mode == 'LVeGTreeLSTM':
        network = LVeGTreeLstm(args.tree_mode, args.leaf_rnn_mode, args.pred_mode, embedd_dim, word_alphabet.size(),
                               args.hidden_size, args.hidden_size, args.softmax_dim, args.leaf_rnn_num,
                               num_labels, embedd_word=word_table, p_in=args.p_in, p_leaf=args.p_leaf,
                               p_tree=args.p_tree, p_pred=args.p_pred, leaf_rnn=leaf_rnn, bi_leaf_rnn=bi_rnn,
                               device=device, comp=args.lveg_comp, g_dim=args.gaussian_dim,
                               pred_dense_layer=pred_dense_layer, attention=attention,
                               coattention_dim=coattention_dim, elmo=elmo, elmo_weight=elmo_weight,
                               elmo_config=elmo_config).to(device)
    elif model_mode == 'LVeGBiTreeLSTM':
        network = LVeGBiTreeLstm(args.tree_mode, args.leaf_rnn_mode, args.pred_mode, embedd_dim, word_alphabet.size(),
                                 args.hidden_size, args.hidden_size, args.softmax_dim, args.leaf_rnn_num,
                                 num_labels, embedd_word=word_table, p_in=args.p_in, p_leaf=args.p_leaf,
                                 p_tree=args.p_tree, p_pred=args.p_pred, leaf_rnn=leaf_rnn, bi_leaf_rnn=bi_rnn,
                                 device=device, comp=args.lveg_comp, g_dim=args.gaussian_dim, attention=attention,
                                 coattention_dim=coattention_dim, elmo=elmo, elmo_weight=elmo_weight,
                                 elmo_config=elmo_config).to(device)
    elif model_mode == 'BiCRFBiTreeLSTM':
        network = BiCRFBiTreeLstm(args.tree_mode, args.leaf_rnn_mode, args.pred_mode, embedd_dim, word_alphabet.size(),
                                  args.hidden_size, args.hidden_size, args.softmax_dim, args.leaf_rnn_num,
                                  num_labels, embedd_word=word_table, p_in=args.p_in, p_leaf=args.p_leaf,
                                  p_tree=args.p_tree, p_pred=args.p_pred, leaf_rnn=leaf_rnn, bi_leaf_rnn=bi_rnn,
                                  device=device, pred_dense_layer=pred_dense_layer, attention=attention,
                                  coattention_dim=coattention_dim, elmo=elmo, elmo_weight=elmo_weight,
                                  elmo_config=elmo_config).to(device)
    elif model_mode == 'BiCRFTreeLSTM':
        network = BiCRFTreeLstm(args.tree_mode, args.leaf_rnn_mode, args.pred_mode, embedd_dim, word_alphabet.size(),
                                args.hidden_size, args.hidden_size, args.softmax_dim, args.leaf_rnn_num,
                                num_labels, embedd_word=word_table, p_in=args.p_in, p_leaf=args.p_leaf,
                                p_tree=args.p_tree, p_pred=args.p_pred, leaf_rnn=leaf_rnn, bi_leaf_rnn=bi_rnn,
                                device=device, pred_dense_layer=pred_dense_layer, attention=attention,
                                coattention_dim=coattention_dim, elmo=elmo, elmo_weight=elmo_weight,
                                elmo_config=elmo_config).to(device)
    elif model_mode == 'LABiCRFTreeLSTM':
        network = LABiCRFTreeLstm(args.tree_mode, args.leaf_rnn_mode, args.pred_mode, embedd_dim, word_alphabet.size(),
                                  args.hidden_size, args.hidden_size, args.softmax_dim, args.leaf_rnn_num,
                                  args.num_labels, embedd_word=word_table, p_in=args.p_in, p_leaf=args.p_leaf,
                                  p_tree=args.p_tree, p_pred=args.p_pred, leaf_rnn=leaf_rnn, bi_leaf_rnn=bi_rnn,
                                  device=device, pred_dense_layer=pred_dense_layer, attention=attention,
                                  coattention_dim=coattention_dim, comp=args.lveg_comp).to(device)
    elif model_mode == 'LABiCRFBiTreeLSTM':
        network = LABiCRFBiTreeLstm(args.tree_mode, args.leaf_rnn_mode, args.pred_mode, embedd_dim,
                                    word_alphabet.size(), args.hidden_size, args.hidden_size, args.softmax_dim,
                                    args.leaf_rnn_num, args.num_labels, embedd_word=word_table, p_in=args.p_in,
                                    p_leaf=args.p_leaf, p_tree=args.p_tree, p_pred=args.p_pred, leaf_rnn=leaf_rnn,
                                    bi_leaf_rnn=bi_rnn, device=device, pred_dense_layer=pred_dense_layer,
                                    attention=attention, coattention_dim=coattention_dim, comp=args.lveg_comp).to(device)
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

    # dev and test
    dev_correct = {'fine_phase': 0.0, 'fine_sents': 0.0, 'bin_phase': 0.0, 'bin_sents': 0.0,
                   'bin_phase_v2': 0.0, 'bin_sents_v2': 0.0, 'full_bin_phase': 0.0, 'full_bin_phase_v2': 0.0}
    best_epoch = {'fine_phase': 0, 'fine_sents': 0, 'bin_phase': 0, 'bin_sents': 0,
                  'bin_phase_v2': 0, 'bin_sents_v2': 0, 'full_bin_phase': 0, 'full_bin_phase_v2': 0}
    test_correct = {}
    for key in all_cite_version:
        test_correct[key] = {'fine_phase': 0.0, 'fine_sents': 0.0, 'bin_phase': 0.0, 'bin_sents': 0.0,
                             'bin_phase_v2': 0.0, 'bin_sents_v2': 0.0, 'full_bin_phase': 0.0, 'full_bin_phase_v2': 0.0}
    test_total = {'fine_phase': 0.0, 'fine_sents': 0.0, 'bin_phase': 0.0, 'bin_sents': 0.0, 'full_bin_phase': 0.0}

    def log_print(name, fine_phase_acc, fine_sents_acc, bin_phase_acc, full_bin_phase_acc, bin_sents_acc,
                  bin_phase_v2_acc, full_bin_phase_v2_acc, bin_sents_v2_acc):
        print(name + ' phase acc: %.2f%%, sents acc: %.2f%%, binary phase acc: %.2f%%, full phase acc: %.2f%%, '
                     'sents acc: %.2f%%, binary phase v2 acc: %.2f%%, full phase v2 acc: %.2f%%, dev sents v2 acc: '
                     '%.2f%% '
              % (fine_phase_acc, fine_sents_acc, bin_phase_acc, full_bin_phase_acc, bin_sents_acc,
                 bin_phase_v2_acc, full_bin_phase_v2_acc, bin_sents_v2_acc))

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
        dev_corr = {'fine_phase': 0.0, 'fine_sents': 0.0, 'bin_phase': 0.0, 'bin_sents': 0.0,
                    'bin_phase_v2': 0.0, 'bin_sents_v2': 0.0, 'full_bin_phase': 0.0, 'full_bin_phase_v2': 0.0}
        dev_tot = {'fine_phase': 0.0, 'fine_sents': float(len(dev_dataset)), 'bin_phase': 0.0, 'bin_sents': 0.0,
                   'bin_phase_v2': 0.0, 'bin_sents_v2': 0.0, 'full_bin_phase': 0.0, 'full_bin_phase_v2': 0.0}
        final_test_corr = {'fine_phase': 0.0, 'fine_sents': 0.0, 'bin_phase': 0.0, 'bin_sents': 0.0,
                           'bin_phase_v2': 0.0, 'bin_sents_v2': 0.0, 'full_bin_phase': 0.0, 'full_bin_phase_v2': 0.0}
        for i in tqdm(range(len(dev_dataset))):
            tree = dev_dataset[i]
            p_corr, preds, bin_corr, bin_preds, bin_mask = network.predict(tree)

            dev_tot['fine_phase'] += preds.size()[0]

            dev_corr['fine_phase'] += p_corr.sum().item()
            dev_corr['fine_sents'] += p_corr[-1].item()
            dev_corr['full_bin_phase'] += bin_corr[0].sum().item()

            if len(bin_corr) == 2:
                dev_corr['full_bin_phase_v2'] += bin_corr[1].sum().item()
            else:
                dev_corr['full_bin_phase_v2'] = dev_corr['full_bin_phase']
            dev_tot['full_bin_phase'] += bin_mask.sum().item()

            if tree.label != int(num_labels//2):
                dev_corr['bin_phase'] += bin_corr[0].sum().item()
                dev_tot['bin_phase'] += bin_mask.sum().item()
                dev_corr['bin_sents'] += bin_corr[0][-1].item()
                if len(bin_corr) == 2:
                    dev_corr['bin_phase_v2'] += bin_corr[1].sum().item()
                    dev_corr['bin_sents_v2'] += bin_corr[1][-1].item()
                else:
                    dev_corr['bin_phase_v2'] = dev_corr['bin_phase']
                    dev_corr['bin_sents_v2'] = dev_corr['bin_sents']
                dev_tot['bin_sents'] += 1.0

            tree.clean()

        time.sleep(1)

        dev_tot['bin_phase_v2'] = dev_tot['bin_phase']
        dev_tot['bin_sents_v2'] = dev_tot['bin_sents']
        dev_tot['full_bin_phase_v2'] = dev_tot['full_bin_phase']

        for key in all_cite_version:
            add_scalar_summary(summary_writer, 'dev/' + key, (dev_corr[key] * 100 / dev_tot[key]), epoch)

        log_print('dev', dev_corr['fine_phase'] * 100 / dev_tot['fine_phase'],
                  dev_corr['fine_sents'] * 100 / dev_tot['fine_sents'],
                  dev_corr['bin_phase'] * 100 / dev_tot['bin_phase'],
                  dev_corr['full_bin_phase'] * 100 / dev_tot['full_bin_phase'],
                  dev_corr['bin_sents'] * 100 / dev_tot['bin_sents'],
                  dev_corr['bin_phase_v2'] * 100 / dev_tot['bin_phase'],
                  dev_corr['full_bin_phase_v2'] * 100 / dev_tot['full_bin_phase'],
                  dev_corr['bin_sents_v2'] * 100 / dev_tot['bin_sents'])

        update = []
        for key in all_cite_version:
            if dev_corr[key] > dev_correct[key]:
                update.append(key)

        # if dev_s_corr > dev_s_correct:

        if len(update) > 0:
            for key in update:
                dev_correct[key] = dev_corr[key]
                # update corresponding test dict cache
                test_correct[key] = {'fine_phase': 0.0, 'fine_sents': 0.0, 'bin_phase': 0.0, 'bin_sents': 0.0,
                                     'bin_phase_v2': 0.0, 'bin_sents_v2': 0.0, 'full_bin_phase': 0.0,
                                     'full_bin_phase_v2': 0.0}
                best_epoch[key] = epoch
            test_total = {'fine_phase': 0.0, 'fine_sents': float(len(test_dataset)), 'bin_phase': 0.0, 'bin_sents': 0.0,
                          'bin_phase_v2': 0.0, 'bin_sents_v2': 0.0, 'full_bin_phase': 0.0,
                          'full_bin_phase_v2': 0.0}

            time.sleep(1)

            for i in tqdm(range(len(test_dataset))):
                tree = test_dataset[i]
                p_corr, preds, bin_corr, bin_preds, bin_mask = network.predict(tree)

                # count total number
                test_total['fine_phase'] += preds.size()[0]
                test_total['full_bin_phase'] += bin_mask.sum().item()
                if tree.label != int(num_labels//2):
                    test_total['bin_phase'] += bin_mask.sum().item()
                    test_total['bin_sents'] += 1.0

                for key in update:
                    test_correct[key]['fine_phase'] += p_corr.sum().item()
                    test_correct[key]['fine_sents'] += p_corr[-1].item()
                    test_correct[key]['full_bin_phase'] += bin_corr[0].sum().item()

                    if len(bin_corr) == 2:
                        test_correct[key]['full_bin_phase_v2'] += bin_corr[1].sum().item()
                    else:
                        test_correct[key]['full_bin_phase_v2'] = test_correct[key]['full_bin_phase']

                    if tree.label != int(num_labels//2):
                        test_correct[key]['bin_phase'] += bin_corr[0].sum().item()
                        test_correct[key]['bin_sents'] += bin_corr[0][-1].item()

                        if len(bin_corr) == 2:
                            test_correct[key]['bin_phase_v2'] += bin_corr[1].sum().item()
                            test_correct[key]['bin_sents_v2'] += bin_corr[1][-1].item()
                        else:
                            test_correct[key]['bin_phase_v2'] = test_correct[key]['bin_phase']
                            test_correct[key]['bin_sents_v2'] = test_correct[key]['bin_sents']

                tree.clean()

            time.sleep(1)

            test_total['bin_phase_v2'] = test_total['bin_phase']
            test_total['bin_sents_v2'] = test_total['bin_sents']
            test_total['full_bin_phase_v2'] = test_total['full_bin_phase']

            for key in update:
                log_print('test ' + key, test_correct[key]['fine_phase'] * 100 / test_total['fine_phase'],
                          test_correct[key]['fine_sents'] * 100 / test_total['fine_sents'],
                          test_correct[key]['bin_phase'] * 100 / test_total['bin_phase'],
                          test_correct[key]['full_bin_phase'] * 100 / test_total['full_bin_phase'],
                          test_correct[key]['bin_sents'] * 100 / test_total['bin_sents'],
                          test_correct[key]['bin_phase_v2'] * 100 / test_total['bin_phase_v2'],
                          test_correct[key]['full_bin_phase_v2'] * 100 / test_total['full_bin_phase_v2'],
                          test_correct[key]['bin_sents_v2'] * 100 / test_total['bin_sents_v2'])

        for key in all_cite_version:
            log_print('Best Epoch ' + str(best_epoch[key]) + ' test_' + key,
                      test_correct[key]['fine_phase'] * 100 / test_total['fine_phase'],
                      test_correct[key]['fine_sents'] * 100 / test_total['fine_sents'],
                      test_correct[key]['bin_phase'] * 100 / test_total['bin_phase'],
                      test_correct[key]['full_bin_phase'] * 100 / test_total['full_bin_phase'],
                      test_correct[key]['bin_sents'] * 100 / test_total['bin_sents'],
                      test_correct[key]['bin_phase_v2'] * 100 / test_total['bin_phase_v2'],
                      test_correct[key]['full_bin_phase_v2'] * 100 / test_total['full_bin_phase_v2'],
                      test_correct[key]['bin_sents_v2'] * 100 / test_total['bin_sents_v2'])

        for key1 in all_cite_version:
            best_score = 0.0
            for key2 in all_cite_version:
                if test_correct[key2][key1] > best_score:
                    best_score = test_correct[key2][key1]
            final_test_corr[key1] = best_score

        for key in all_cite_version:
            add_scalar_summary(summary_writer, 'test/' + key, (final_test_corr[key] * 100 / test_total[key]), epoch)

        log_print('Best ' + str(epoch) + ' Final test_',
                  final_test_corr['fine_phase'] * 100 / test_total['fine_phase'],
                  final_test_corr['fine_sents'] * 100 / test_total['fine_sents'],
                  final_test_corr['bin_phase'] * 100 / test_total['bin_phase'],
                  final_test_corr['full_bin_phase'] * 100 / test_total['full_bin_phase'],
                  final_test_corr['bin_sents'] * 100 / test_total['bin_sents'],
                  final_test_corr['bin_phase_v2'] * 100 / test_total['bin_phase_v2'],
                  final_test_corr['full_bin_phase_v2'] * 100 / test_total['full_bin_phase_v2'],
                  final_test_corr['bin_sents_v2'] * 100 / test_total['bin_sents_v2'])

        if optim_method == "SGD" and epoch % schedule == 0:
            lr = learning_rate / (epoch * decay_rate)
            optimizer = optim.SGD(network.parameters(), lr=lr, momentum=momentum, weight_decay=gamma, nesterov=True)

    if args.tensorboard:
        summary_writer.close()
    else:
        pass


if __name__ == '__main__':
    torch.manual_seed(48)
    np.random.seed(48)
    main()
