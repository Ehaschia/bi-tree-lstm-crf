__author__ = 'Ehaschia'

import argparse
import sys
import os

import time
import datetime
import torch.optim as optim
from tqdm import tqdm

from module.module_io.alphabet import Alphabet
from module.module_io.logger import *
from module.module_io.sst_data import *
from module.nn.tree_lstm_eb import *
from tensorboardX import SummaryWriter
from pytorch_pretrained_bert import BertModel, BertTokenizer
from allennlp.modules.elmo import Elmo
from allennlp.data.dataset_readers import StanfordSentimentTreeBankDatasetReader
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer, SingleIdTokenIndexer
from allennlp.modules.token_embedders.embedding import *
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder


def main():
    parser = argparse.ArgumentParser(description='Tuning with bi-directional Tree-LSTM-CRF')
    parser.add_argument('--model_mode', choices=['elmo', 'elmo_crf', 'elmo_bicrf', 'elmo_lveg',
                                                 'bert', 'elmo_la'])
    parser.add_argument('--batch_size', type=int, default=16, help='Number of batch')
    parser.add_argument('--epoch', type=int, default=50, help='run epoch')
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
    parser.add_argument('--train', type=str, default='/path/to/SST/train.txt')
    parser.add_argument('--dev', type=str, default='/path/to/SST/dev.txt')
    parser.add_argument('--test', type=str, default='/path/to/SST/test.txt')
    parser.add_argument('--num_labels', type=int, default=5)
    parser.add_argument('--embedding_p', type=float, default=0.5, help="Dropout prob for embedding")
    parser.add_argument('--component_num', type=int, default=1, help='the component number of mixture gaussian in LVeG')
    parser.add_argument('--gaussian_dim', type=int, default=1, help='the gaussian dim in LVeG')
    parser.add_argument('--tensorboard', action='store_true')
    parser.add_argument('--td_name', type=str, default='default', help='the name of this test')
    parser.add_argument('--td_dir', type=str, required=True)
    parser.add_argument('--elmo_weight', type=str,
                        default='/path/to/elmo/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5')
    parser.add_argument('--elmo_config', type=str,
                        default='/path/to/elmo//elmo_2x4096_512_2048cnn_2xhighway_options.json')
    parser.add_argument('--elmo_input', action='store_true')
    parser.add_argument('--elmo_output', action='store_true')
    parser.add_argument('--elmo_preencoder_dim', type=str, default='300')
    parser.add_argument('--elmo_preencoder_p', type=str, default='0.25')
    parser.add_argument('--elmo_encoder_dim', type=int, default=300)
    parser.add_argument('--elmo_integrtator_dim', type=int, default=300)
    parser.add_argument('--elmo_integrtator_p', type=float, default=0.1)
    parser.add_argument('--elmo_output_dim', type=str, default='1200,600')
    parser.add_argument('--elmo_output_p', type=str, default='0.2,0.3,0.0')
    parser.add_argument('--elmo_output_pool_size', type=int, default=4)
    parser.add_argument('--bert_pred_dropout', type=float, default=0.1)
    parser.add_argument('--bert_dir', type=str, default='path/to/bert/')
    parser.add_argument('--bert_model', choices=['bert-base-uncased', 'bert-large-uncased', 'bert-base-cased',
                                                 'bert-large-cased'])
    parser.add_argument('--random_seed', type=int, default=48)
    parser.add_argument('--pcfg_init', action='store_true', help='init the crf or lveg weight according to the '
                                                                 'distribution of trainning dataset')
    parser.add_argument('--save_model', action='store_true', help='save_model')
    parser.add_argument('--load_model', action='store_true', help='load_model')
    parser.add_argument('--model_path', default='./model/')
    parser.add_argument('--model_name', default=None)

    # load tree
    args = parser.parse_args()
    print(args)
    logger = get_logger("SSTLogger")

    # set random seed
    random_seed = args.random_seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    myrandom = Random(random_seed)

    batch_size = args.batch_size
    embedd_mode = args.embedding
    model_mode = args.model_mode
    num_labels = args.num_labels

    elmo = model_mode.find('elmo') != -1
    bert = model_mode.find('bert') != -1

    elmo_weight = args.elmo_weight
    elmo_config = args.elmo_config

    load_model = args.load_model
    save_model = args.save_model
    model_path = args.model_path
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_name = args.model_name
    if save_model:
        model_name = model_path + '/' + model_mode + datetime.datetime.now().strftime("%H%M%S")
    if load_model:
        model_name = model_path + '/' + model_name

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

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

    # ELMO PART
    # allennlp prepare part
    # build Vocabulary
    if elmo:
        elmo_model = Elmo(elmo_config, elmo_weight, 1, requires_grad=False, dropout=0.0).to(device)
        token_indexers = {'tokens': SingleIdTokenIndexer(),
                          'elmo': ELMoTokenCharactersIndexer()}
        train_reader = StanfordSentimentTreeBankDatasetReader(token_indexers=token_indexers, use_subtrees=False)
        dev_reader = StanfordSentimentTreeBankDatasetReader(token_indexers=token_indexers, use_subtrees=False)

        allen_train_dataset = train_reader.read(args.train)
        allen_dev_dataset = dev_reader.read(args.dev)
        allen_test_dataset = dev_reader.read(args.test)
        allen_vocab = Vocabulary.from_instances(allen_train_dataset + allen_dev_dataset + allen_test_dataset,
                                                min_count={'tokens': 1})
        # Build Embddering Layer
        if embedd_mode != 'random':
            params = Params({'embedding_dim': 300,
                             'pretrained_file': args.embedding_path,
                             'trainable': False})

            embedding = Embedding.from_params(allen_vocab, params)
            embedder = BasicTextFieldEmbedder({'tokens': embedding})
        else:
            # alert not random init here!
            embedder = None
            pass
    else:
        elmo_model = None
        token_indexers = None
        embedder = None
        allen_vocab = None

    if bert:
        bert_path = args.bert_dir + '/' + args.bert_model
        bert_model = BertModel.from_pretrained(bert_path + '.tar.gz').to(device)
        if bert_path.find('large') != -1:
            bert_dim = 1024
        else:
            bert_dim = 768
        for parameter in bert_model.parameters():
            parameter.requires_grad = False
        bert_model.eval()
        bert_tokenizer = BertTokenizer.from_pretrained(bert_path + 'txt', do_lower_case=args.lower)
    else:
        bert_model = None
        bert_tokenizer = None
        bert_dim = 768

    logger.info("constructing network...")

    # alphabet
    word_alphabet = Alphabet('word', default_value=True)
    # Read data
    logger.info("Reading Data")

    train_dataset = read_sst_data(args.train, word_alphabet, random=myrandom, merge=True)
    dev_dataset = read_sst_data(args.dev, word_alphabet, random=myrandom, merge=True)
    test_dataset = read_sst_data(args.test, word_alphabet, random=myrandom, merge=True)
    word_alphabet.close()

    if num_labels == 3:
        train_dataset.convert_to_3_class()
        dev_dataset.convert_to_3_class()
        test_dataset.convert_to_3_class()

    # PCFG init
    if args.pcfg_init and (str.lower(model_mode).find('crf') != -1 or str.lower(model_mode).find('lveg') != -1):
        if str.lower(model_mode).find('bicrf') != -1 or str.lower(model_mode).find('lveg') != -1:
            dim = 3
        else:
            dim = 2
        trans_matrix = train_dataset.collect_rule_count(dim, num_labels, smooth=True)
    else:
        trans_matrix = None

    pre_encode_dim = [int(dim) for dim in args.elmo_preencoder_dim.split(',')]
    pre_encode_layer_dropout_prob = [float(prob) for prob in args.elmo_preencoder_p.split(',')]
    output_dim = [int(dim) for dim in args.elmo_output_dim.split(',')] + [num_labels]
    output_dropout = [float(prob) for prob in args.elmo_output_p.split(',')]

    if model_mode == 'elmo':
        # fixme ugly word dim
        network = Biattentive(vocab=allen_vocab, embedder=embedder, embedding_dropout_prob=args.embedding_p,
                              word_dim=300, use_input_elmo=args.elmo_input, pre_encode_dim=pre_encode_dim,
                              pre_encode_layer_dropout_prob=pre_encode_layer_dropout_prob,
                              encode_output_dim=args.elmo_encoder_dim,
                              integrtator_output_dim=args.elmo_integrtator_dim,
                              integrtator_dropout=args.elmo_integrtator_p,
                              use_integrator_output_elmo=args.elmo_output, output_dim=output_dim,
                              output_pool_size=args.elmo_output_pool_size, output_dropout=output_dropout,
                              elmo=elmo_model, token_indexer=token_indexers, device=device).to(device)
    elif model_mode == 'elmo_crf':
        network = CRFBiattentive(vocab=allen_vocab, embedder=embedder, embedding_dropout_prob=args.embedding_p,
                                 word_dim=300, use_input_elmo=args.elmo_input, pre_encode_dim=pre_encode_dim,
                                 pre_encode_layer_dropout_prob=pre_encode_layer_dropout_prob,
                                 encode_output_dim=args.elmo_encoder_dim,
                                 integrtator_output_dim=args.elmo_integrtator_dim,
                                 integrtator_dropout=args.elmo_integrtator_p,
                                 use_integrator_output_elmo=args.elmo_output, output_dim=output_dim,
                                 output_pool_size=args.elmo_output_pool_size, output_dropout=output_dropout,
                                 elmo=elmo_model, token_indexer=token_indexers, device=device,
                                 trans_mat=trans_matrix).to(device)
    elif model_mode == 'elmo_bicrf':
        network = BiCRFBiattentive(vocab=allen_vocab, embedder=embedder, embedding_dropout_prob=args.embedding_p,
                                   word_dim=300, use_input_elmo=args.elmo_input, pre_encode_dim=pre_encode_dim,
                                   pre_encode_layer_dropout_prob=pre_encode_layer_dropout_prob,
                                   encode_output_dim=args.elmo_encoder_dim,
                                   integrtator_output_dim=args.elmo_integrtator_dim,
                                   integrtator_dropout=args.elmo_integrtator_p,
                                   use_integrator_output_elmo=args.elmo_output, output_dim=output_dim,
                                   output_pool_size=args.elmo_output_pool_size, output_dropout=output_dropout,
                                   elmo=elmo_model, token_indexer=token_indexers, device=device,
                                   trans_mat=trans_matrix).to(device)
    elif model_mode == 'elmo_lveg':
        network = LVeGBiattentive(vocab=allen_vocab, embedder=embedder, embedding_dropout_prob=args.embedding_p,
                                  word_dim=300, use_input_elmo=args.elmo_input, pre_encode_dim=pre_encode_dim,
                                  pre_encode_layer_dropout_prob=pre_encode_layer_dropout_prob,
                                  encode_output_dim=args.elmo_encoder_dim,
                                  integrtator_output_dim=args.elmo_integrtator_dim,
                                  integrtator_dropout=args.elmo_integrtator_p,
                                  use_integrator_output_elmo=args.elmo_output, output_dim=output_dim,
                                  output_pool_size=args.elmo_output_pool_size, output_dropout=output_dropout,
                                  elmo=elmo_model, token_indexer=token_indexers, device=device,
                                  gaussian_dim=args.gaussian_dim, component_num=args.component_num,
                                  trans_mat=trans_matrix).to(device)
    elif model_mode == 'elmo_la':
        network = LABiattentive(vocab=allen_vocab, embedder=embedder, embedding_dropout_prob=args.embedding_p,
                                word_dim=300, use_input_elmo=args.elmo_input, pre_encode_dim=pre_encode_dim,
                                pre_encode_layer_dropout_prob=pre_encode_layer_dropout_prob,
                                encode_output_dim=args.elmo_encoder_dim,
                                integrtator_output_dim=args.elmo_integrtator_dim,
                                integrtator_dropout=args.elmo_integrtator_p,
                                use_integrator_output_elmo=args.elmo_output, output_dim=output_dim,
                                output_pool_size=args.elmo_output_pool_size, output_dropout=output_dropout,
                                elmo=elmo_model, token_indexer=token_indexers, device=device,
                                comp=args.component_num, trans_mat=trans_matrix).to(device)
    elif model_mode == 'bert':
        # alert should be 2 classification, should test original model first
        network = BertClassification(tokenizer=bert_tokenizer, pred_dim=bert_dim, pred_dropout=args.bert_pred_dropout,
                                     bert=bert_model, num_labels=num_labels, device=device)
    else:
        raise NotImplementedError

    if load_model:
        logger.info('Load model from:' + model_name)
        network.load_state_dict(torch.load(model_name))

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

    def log_print(name, fine_phase_acc, fine_sents_acc, bin_sents_acc,bin_phase_v2_acc):
        print(name + ' phase acc: %.2f%%, sents acc: %.2f%%, binary sents acc: %.2f%%, binary phase acc: %.2f%%,'
              % (fine_phase_acc, fine_sents_acc, bin_sents_acc, bin_phase_v2_acc))

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
            output_dict = network.loss(tree)
            loss = output_dict['loss']
            a_tree_p_cnt = 2 * tree.length - 1
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

        if save_model:
            logger.info('Save model to ' + model_name + '_' + str(epoch))
            torch.save(network.state_dict(), model_name + '_' + str(epoch))

        network.eval()
        dev_corr = {'fine_phase': 0.0, 'fine_sents': 0.0, 'bin_phase': 0.0, 'bin_sents': 0.0,
                    'bin_phase_v2': 0.0, 'bin_sents_v2': 0.0, 'full_bin_phase': 0.0, 'full_bin_phase_v2': 0.0}
        dev_tot = {'fine_phase': 0.0, 'fine_sents': float(len(dev_dataset)), 'bin_phase': 0.0, 'bin_sents': 0.0,
                   'bin_phase_v2': 0.0, 'bin_sents_v2': 0.0, 'full_bin_phase': 0.0, 'full_bin_phase_v2': 0.0}
        final_test_corr = {'fine_phase': 0.0, 'fine_sents': 0.0, 'bin_phase': 0.0, 'bin_sents': 0.0,
                           'bin_phase_v2': 0.0, 'bin_sents_v2': 0.0, 'full_bin_phase': 0.0, 'full_bin_phase_v2': 0.0}
        for i in tqdm(range(len(dev_dataset))):
            tree = dev_dataset[i]
            output_dict = network.predict(tree)
            p_corr, preds, bin_corr, bin_preds, bin_mask = output_dict['corr'], output_dict['label'], \
                                                           output_dict['binary_corr'], output_dict['binary_pred'], \
                                                           output_dict['binary_mask']

            dev_tot['fine_phase'] += preds.size

            dev_corr['fine_phase'] += p_corr.sum()
            dev_corr['fine_sents'] += p_corr[-1]
            dev_corr['full_bin_phase'] += bin_corr[0].sum()

            if len(bin_corr) == 2:
                dev_corr['full_bin_phase_v2'] += bin_corr[1].sum()
            else:
                dev_corr['full_bin_phase_v2'] = dev_corr['full_bin_phase']
            dev_tot['full_bin_phase'] += bin_mask.sum()

            if tree.label != int(num_labels / 2):
                dev_corr['bin_phase'] += bin_corr[0].sum()
                dev_tot['bin_phase'] += bin_mask.sum()
                dev_corr['bin_sents'] += bin_corr[0][-1]
                if len(bin_corr) == 2:
                    dev_corr['bin_phase_v2'] += bin_corr[1].sum()
                    dev_corr['bin_sents_v2'] += bin_corr[1][-1]
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
                  dev_corr['bin_sents'] * 100 / dev_tot['bin_sents'],
                  dev_corr['bin_phase_v2'] * 100 / dev_tot['bin_phase'])

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
                          'bin_phase_v2': 0.0, 'bin_sents_v2': 0.0, 'full_bin_phase': 0.0, 'full_bin_phase_v2': 0.0}

            time.sleep(1)

            for i in tqdm(range(len(test_dataset))):
                tree = test_dataset[i]
                output_dict = network.predict(tree)
                p_corr, preds, bin_corr, bin_preds, bin_mask = output_dict['corr'], output_dict['label'], \
                                                               output_dict['binary_corr'], output_dict['binary_pred'], \
                                                               output_dict['binary_mask']
                # count total number
                test_total['fine_phase'] += preds.size
                test_total['full_bin_phase'] += bin_mask.sum()
                if tree.label != int(num_labels / 2):
                    test_total['bin_phase'] += bin_mask.sum()
                    test_total['bin_sents'] += 1.0

                for key in update:
                    test_correct[key]['fine_phase'] += p_corr.sum()
                    test_correct[key]['fine_sents'] += p_corr[-1]
                    test_correct[key]['full_bin_phase'] += bin_corr[0].sum()

                    if len(bin_corr) == 2:
                        test_correct[key]['full_bin_phase_v2'] += bin_corr[1].sum()
                    else:
                        test_correct[key]['full_bin_phase_v2'] = test_correct[key]['full_bin_phase']

                    if tree.label != int(num_labels / 2):
                        test_correct[key]['bin_phase'] += bin_corr[0].sum()
                        test_correct[key]['bin_sents'] += bin_corr[0][-1]

                        if len(bin_corr) == 2:
                            test_correct[key]['bin_phase_v2'] += bin_corr[1].sum()
                            test_correct[key]['bin_sents_v2'] += bin_corr[1][-1]
                        else:
                            test_correct[key]['bin_phase_v2'] = test_correct[key]['bin_phase']
                            test_correct[key]['bin_sents_v2'] = test_correct[key]['bin_sents']

                tree.clean()

            time.sleep(1)

            test_total['bin_phase_v2'] = test_total['bin_phase']
            test_total['bin_sents_v2'] = test_total['bin_sents']
            test_total['full_bin_phase_v2'] = test_total['full_bin_phase']


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
                  final_test_corr['bin_sents'] * 100 / test_total['bin_sents'],
                  final_test_corr['bin_phase_v2'] * 100 / test_total['bin_phase_v2'])

        if optim_method == "SGD" and epoch % schedule == 0:
            lr = learning_rate / (epoch * decay_rate)
            optimizer = optim.SGD(network.parameters(), lr=lr, momentum=momentum, weight_decay=gamma, nesterov=True)

    if args.tensorboard:
        summary_writer.close()
    else:
        pass


if __name__ == '__main__':
    main()
