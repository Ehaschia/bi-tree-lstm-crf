import copy
import os

# script paramter
script_name = 'tmp'
model_dir = '/home/liwenzhang/code/sentiment/bi-tree-lstm-crf'
gpu_id = 0
pbs_id = 13
pbs_data = "#PBS -l walltime=1000:00:00 \n#PBS -N node13 \n#PBS -l nodes=sist-gpu" + str(pbs_id) + \
           ":ppn=1 \n#PBS -S /bin/bash \n#PBS -k oe \n#PBS -q sist-tukw \n#PBS -u zhanglw"
# batch script parameter
run_num = 1
run_prefix = 1

# model paramter
leaf_lstm = False  # store_true
bi_leaf_lstm = False  # store true
leaf_rnn_mode = 'LSTM'  # choice
leaf_rnn_num = 1  # choice
tree_mode = 'BUTreeLSTM'  # choice
model_mode = 'TreeLSTM'  # choice
pred_mode = 'avg_h'  # choice
batch_size = 16
epoch = 20  # hard
hidden_size = 100
pred_dense_layer = True
softmax_dim = 64
optim_method = 'Adam'
lr = 0.001  # rule based on optim method
momentum = 0.9
decay_rate = 0.1
gamma = 0.0
schedule = 5
embedding = 'glove'
embedding_path = '/home/liwenzhang/code/sentiment/data/glove.840B.300d.txt'
train = '/home/liwenzhang/code/sentiment/data/trees/train.txt'
dev = '/home/liwenzhang/code/sentiment/data/trees/dev.txt'
test = '/home/liwenzhang/code/sentiment/data/trees/test.txt'
num_labels = 5
p_in = 0.5
p_leaf = 0.0
p_tree = 0.0
p_pred = 0.0
tensorboard = True
td_name = 'tmp'
td_dir = 'log'

attention = True
coattention_dim = 150
elmo = 'none'
elmo_weight = 'elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'
elmo_config = 'elmo_2x4096_512_2048cnn_2xhighway_options.json'
lveg_comp = [1]
gaussian_dim = [1]

base_dict = {}

if leaf_lstm:
    base_dict['leaf_lstm'] = '--leaf_lstm '
    if bi_leaf_lstm:
        base_dict['bi_leaf_lstm'] = '--bi_leaf_lstm '

base_dict['leaf_rnn_mode'] = '--leaf_rnn_mode ' + leaf_rnn_mode + ' '
base_dict['leaf_rnn_num'] = '--leaf_rnn_num ' + str(leaf_rnn_num) + ' '
base_dict['tree_mode'] = '--tree_mode ' + tree_mode + ' '
base_dict['pred_mode'] = '--pred_mode ' + pred_mode + ' '
# alert the model model specialize for different file
base_dict['batch_size'] = '--batch_size ' + str(batch_size) + ' '
base_dict['epoch'] = '--epoch ' + str(epoch) + ' '
base_dict['hidden_size'] = '--hidden_size ' + str(hidden_size) + ' '
base_dict['softmax_dim'] = '--softmax_dim ' + str(softmax_dim) + ' '
if pred_dense_layer:
    base_dict['pred_dense_layer'] = '--pred_dense_layer '
base_dict['optim_method'] = '--optim_method ' + optim_method + ' '
base_dict['learning_rate'] = '--learning_rate ' + str(lr) + ' '
base_dict['momentum'] = '--momentum ' + str(momentum) + ' '
base_dict['decay_rate'] = '--decay_rate ' + str(decay_rate) + ' '
base_dict['gamma'] = '--gamma ' + str(gamma) + ' '
base_dict['schedule'] = '--schedule ' + str(schedule) + ' '
base_dict['embedding'] = '--embedding ' + embedding + ' '
base_dict['embedding_path'] = '--embedding_path ' + embedding_path + ' '
base_dict['train'] = '--train ' + train + ' '
base_dict['dev'] = '--dev ' + dev + ' '
base_dict['test'] = '--test ' + test + ' '
base_dict['num_labels'] = '--num_labels ' + str(num_labels) + ' '
base_dict['p_in'] = '--p_in ' + str(p_in) + ' '
base_dict['p_leaf'] = '--p_leaf ' + str(p_leaf) + ' '
base_dict['p_tree'] = '--p_tree ' + str(p_tree) + ' '
base_dict['p_pred'] = '--p_pred ' + str(p_pred) + ' '

if tensorboard:
    base_dict['tensorboard'] = '--tensorboard '
base_dict['td_dir'] = '--td_dir ' + td_dir + ' '

if attention:
    base_dict['attention'] = '--attention '
base_dict['coattention_dim'] = '--coattention_dim ' + str(coattention_dim) + ' '

base_dict['elmo'] = '--elmo ' + elmo + ' '
base_dict['elmo_weight'] = '--elmo_weight ' + elmo_weight + ' '
base_dict['elmo_config'] = '--elmo_config ' + elmo_config + ' '

crf_dict = copy.copy(base_dict)
bicrf_dict = copy.copy(base_dict)
lveg_dict = copy.copy(base_dict)

base_dict['model_mode'] = '--model_mode ' + model_mode + ' '
crf_dict['model_mode'] = '--model_mode CRF' + model_mode + ' '
bicrf_dict['model_mode'] = '--model_mode BiCRF' + model_mode + ' '
lveg_dict['model_mode'] = '--model_mode LVeG' + model_mode + ' '

base_dict['td_name'] = '--td_name ' + td_name + ' '
crf_dict['td_name'] = '--td_name crf_' + td_name + ' '
bicrf_dict['td_name'] = '--td_name bicrf_' + td_name + ' '
lveg_dict['td_name'] = '--td_name lveg_' + td_name + ' '

lveg_list = []
for comp in lveg_comp:
    for dim in gaussian_dim:
        tmp_script = copy.copy(lveg_dict)
        tmp_script['lveg_comp'] = '--lveg_comp ' + str(comp) + ' '
        tmp_script['gaussian_dim'] = '--gaussian_dim ' + str(dim) + ' '
        tmp_script['td_name'] = tmp_script['td_name'].strip() + '_c' + str(comp) + 'd' + str(dim) + ' '
        lveg_list.append(tmp_script)

prefix = '#!/usr/bin/env bash\nCUDA_VISIBLE_DEVICES=' + str(gpu_id) + ' python example/sentiment.py '

scripts_dict = {}
# encoding base
script = copy.copy(prefix)
for key in base_dict.keys():
    script += base_dict[key]

name = 'base_' + script_name
scripts_dict[name] = script

# encoding crf
script = copy.copy(prefix)
for key in crf_dict.keys():
    script += crf_dict[key]

name = 'crf_' + script_name
scripts_dict[name] = script

# encoding bicrf
script = copy.copy(prefix)
for key in bicrf_dict.keys():
    script += bicrf_dict[key]

name = 'bicrf_' + script_name
scripts_dict[name] = script

# encoding lveg
cnt = 0
for lveg_dict in lveg_list:

    script = copy.copy(prefix)
    for key in lveg_dict.keys():
        script += lveg_dict[key]

    name = 'lveg_' + str(cnt) + '_' + script_name
    scripts_dict[name] = script
    cnt += 1

# write script
for name in scripts_dict.keys():
    with open(name + '.sh', 'w') as f:
        f.write(scripts_dict[name])

# write batch run script
if not os.path.exists('script'):
    os.makedirs('script')

for i in range(run_num):
    with open('script/run' + str(run_prefix + i) + '.sh', 'w') as f:
        f.write('#!/usr/bin/env bash\n')
        f.write(pbs_data + '\n')
        f.write('source activate allen\n')
        f.write('cd ' + model_dir + '\n')

cnt = 0
for key in scripts_dict.keys():
    with open('script/run' + str(run_prefix + cnt % run_num) + '.sh', 'a+') as f:
        f.write('sh example/' + key + '.sh')
        f.write('\n')
        f.write('sleep 5\n')
    cnt += 1
