__author__ = 'Ehaschia'

import argparse
import sys


sys.path.append('/public/sist/home/zhanglw/code/sentiment/elmo/bi-tree-lstm-crf/')

import time
from tqdm import tqdm

from module.module_io.alphabet import Alphabet
from module.module_io.logger import *
from module.module_io.sst_data import *
from module.nn.tree_lstm_eb import *
from allennlp.modules.elmo import Elmo
from allennlp.data.dataset_readers import StanfordSentimentTreeBankDatasetReader
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer, SingleIdTokenIndexer
from allennlp.modules.token_embedders.embedding import *
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder


def main():
    parser = argparse.ArgumentParser(description='Tuning with bi-directional Tree-LSTM-CRF')
    parser.add_argument('--embedding_path', help='path for embedding dict')
    parser.add_argument('--train', type=str, default='/home/ehaschia/Code/dataset/sst/trees/train.txt')
    parser.add_argument('--dev', type=str, default='/home/ehaschia/Code/dataset/sst/trees/dev.txt')
    parser.add_argument('--test', type=str, default='/home/ehaschia/Code/dataset/sst/trees/test.txt')
    parser.add_argument('--num_labels', type=int, default=5)
    parser.add_argument('--embedding_p', type=float, default=0.5, help="Dropout prob for embedding")
    parser.add_argument('--component_num', type=int, default=1, help='the component number of mixture gaussian in LVeG')
    parser.add_argument('--gaussian_dim', type=int, default=1, help='the gaussian dim in LVeG')
    parser.add_argument('--tensorboard', action='store_true')
    parser.add_argument('--td_name', type=str, default='default', help='the name of this test')
    parser.add_argument('--td_dir', type=str, required=True)
    parser.add_argument('--elmo_weight', type=str,
                        default='/home/ehaschia/Code/dataset/elmo/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5')
    parser.add_argument('--elmo_config', type=str,
                        default='/home/ehaschia/Code/dataset/elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json')
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
    parser.add_argument('--random_seed', type=int, default=48)
    parser.add_argument('--model_path', default='./model/')
    parser.add_argument('--model_name', default=None)

    # load tree
    # load tree
    args = parser.parse_args()
    print(args)
    logger = get_logger("SSTLogger")

    # set random seed
    random_seed = args.random_seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    myrandom = Random(random_seed)

    # embedd_mode = args.embedding
    num_labels = args.num_labels

    elmo_weight = args.elmo_weight
    elmo_config = args.elmo_config
    model_name = args.model_path + '/' + args.model_name

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # ELMO PART
    # allennlp prepare part
    # build Vocabulary
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
    # if embedd_mode != 'random':
    #     params = Params({'embedding_dim': 300,
    #                      'pretrained_file': args.embedding_path,
    #                      'trainable': False})
    #
    #     embedding = Embedding.from_params(allen_vocab, params)
    #     embedder = BasicTextFieldEmbedder({'tokens': embedding})
    # else:
    #     # alert not random init here!
    #     embedder = None
    #     pass
    params = Params({'embedding_dim': 300,
                     'pretrained_file': args.embedding_path,
                     'trainable': False})

    embedding = Embedding.from_params(allen_vocab, params)
    embedder = BasicTextFieldEmbedder({'tokens': embedding})

    logger.info("constructing network...")

    # alphabet
    word_alphabet = Alphabet('word', default_value=True)
    # Read data
    logger.info("Reading Data")

    train_dataset = read_sst_data(args.train, word_alphabet, random=myrandom, merge=True)
    dev_dataset = read_sst_data(args.dev, word_alphabet, random=myrandom, merge=True)
    test_dataset = read_sst_data(args.test, word_alphabet, random=myrandom, merge=True)
    word_alphabet.close()

    pre_encode_dim = [int(dim) for dim in args.elmo_preencoder_dim.split(',')]
    pre_encode_layer_dropout_prob = [float(prob) for prob in args.elmo_preencoder_p.split(',')]
    output_dim = [int(dim) for dim in args.elmo_output_dim.split(',')] + [num_labels]
    output_dropout = [float(prob) for prob in args.elmo_output_p.split(',')]

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
                              trans_mat=None).to(device)

    logger.info('Load model from:' + model_name)
    network.load_state_dict(torch.load(model_name))

    # dev and test

    def log_print(name, fine_phase_acc, fine_sents_acc, bin_sents_acc,bin_phase_v2_acc):
        print(name + ' phase acc: %.2f%%, sents acc: %.2f%%, binary sents acc: %.2f%%, binary phase acc: %.2f%%,'
              % (fine_phase_acc, fine_sents_acc, bin_sents_acc, bin_phase_v2_acc))

    network.eval()
    dev_corr = {'fine_phase': 0.0, 'fine_sents': 0.0, 'bin_phase': 0.0, 'bin_sents': 0.0,
                'bin_phase_v2': 0.0, 'bin_sents_v2': 0.0, 'full_bin_phase': 0.0, 'full_bin_phase_v2': 0.0}
    dev_tot = {'fine_phase': 0.0, 'fine_sents': float(len(dev_dataset)), 'bin_phase': 0.0, 'bin_sents': 0.0,
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

    dev_tot['bin_phase_v2'] = dev_tot['bin_phase']
    dev_tot['bin_sents_v2'] = dev_tot['bin_sents']
    dev_tot['full_bin_phase_v2'] = dev_tot['full_bin_phase']

    time.sleep(1)

    log_print('dev', dev_corr['fine_phase'] * 100 / dev_tot['fine_phase'],
              dev_corr['fine_sents'] * 100 / dev_tot['fine_sents'],
              dev_corr['bin_sents'] * 100 / dev_tot['bin_sents'],
              dev_corr['bin_phase_v2'] * 100 / dev_tot['bin_phase'])

    # update corresponding test dict cache
    test_correct = {'fine_phase': 0.0, 'fine_sents': 0.0, 'bin_phase': 0.0, 'bin_sents': 0.0,
                         'bin_phase_v2': 0.0, 'bin_sents_v2': 0.0, 'full_bin_phase': 0.0,
                         'full_bin_phase_v2': 0.0}
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

        test_correct['fine_phase'] += p_corr.sum()
        test_correct['fine_sents'] += p_corr[-1]
        test_correct['full_bin_phase'] += bin_corr[0].sum()

        if len(bin_corr) == 2:
            test_correct['full_bin_phase_v2'] += bin_corr[1].sum()
        else:
            test_correct['full_bin_phase_v2'] = test_correct['full_bin_phase']

        if tree.label != int(num_labels / 2):
            test_correct['bin_phase'] += bin_corr[0].sum()
            test_correct['bin_sents'] += bin_corr[0][-1]

            if len(bin_corr) == 2:
                test_correct['bin_phase_v2'] += bin_corr[1].sum()
                test_correct['bin_sents_v2'] += bin_corr[1][-1]
            else:
                test_correct['bin_phase_v2'] = test_correct['bin_phase']
                test_correct['bin_sents_v2'] = test_correct['bin_sents']

        tree.clean()

    time.sleep(1)

    test_total['bin_phase_v2'] = test_total['bin_phase']
    test_total['bin_sents_v2'] = test_total['bin_sents']
    test_total['full_bin_phase_v2'] = test_total['full_bin_phase']

    log_print('test ', test_correct['fine_phase'] * 100 / test_total['fine_phase'],
                  test_correct['fine_sents'] * 100 / test_total['fine_sents'],
                  test_correct['bin_sents'] * 100 / test_total['bin_sents'],
                  test_correct['bin_phase_v2'] * 100 / test_total['bin_phase_v2'])


if __name__ == '__main__':
    main()
