import copy
import os

# script paramter
script_dir = 'scripts'
model_dir = '/home/liwenzhang/code/sentiment/bi-tree-lstm-crf'
# gpu_id = 0
gpu_prefix = 0
max_gpu_id = 4

# batch script parameter
pbs_id = [13, 14, 15, 16]
run_dir = 'run_files'
run_batch = 8
run_suffix = [0, 1, 2, 3]

# model paramter
leaf_lstm = [False, True]  # store_true
bi_leaf_lstm = True  # store true
leaf_rnn_mode = 'LSTM'  # choice
leaf_rnn_num = 1  # choice
tree_mode = 'BUTreeLSTM'  # choice
model_mode = 'BiTreeLSTM'  # choice
pred_mode = 'td_avg_h'  # choice
batch_size = [8, 16]
epoch = 20  # hard
hidden_size = [100, 200]
pred_dense_layer = [True, False]
softmax_dim = 64
optim_method = 'Adam'
learning_rate = 0.001  # rule based on optim method
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

attention = False
coattention_dim = 150
elmo = 'none'
elmo_weight = '/home/liwenzhang/code/sentiment/data/elmo/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'
elmo_config = '/home/liwenzhang/code/sentiment/data/elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json'
lveg_comp = 1
gaussian_dim = 1

parameter_list = [('leaf_lstm', leaf_lstm),
                  ('bi_leaf_lstm', bi_leaf_lstm),
                  ('leaf_rnn_mode', leaf_rnn_mode),
                  ('tree_mode', tree_mode),
                  ('model_mode', model_mode),
                  ('pred_mode', pred_mode),
                  ('batch_size', batch_size),
                  ('epoch', epoch),
                  ('hidden_size', hidden_size),
                  ('pred_dense_layer', pred_dense_layer),
                  ('softmax_dim', softmax_dim),
                  ('optim_method', optim_method),
                  ('learning_rate', learning_rate),
                  ('momentum', momentum),
                  ('decay_rate', decay_rate),
                  ('gamma', gamma),
                  ('schedule', schedule),
                  ('embedding', embedding),
                  ('embedding_path', embedding_path),
                  ('train', train),
                  ('dev', dev),
                  ('test', test),
                  ('num_labels', num_labels),
                  ('p_in', p_in),
                  ('p_leaf', p_leaf),
                  ('p_tree', p_tree),
                  ('p_pred', p_pred),
                  # temporarily change ugly
                  ('tensorboard', ''),
                  ('td_dir', td_dir),
                  ('td_name', td_name),
                  # ('attention', attention),
                  ('coattention_dim', coattention_dim),
                  ('elmo', elmo),
                  ('elmo_weight', elmo_weight),
                  ('lveg_comp', lveg_comp),
                  ('gaussian_dim', gaussian_dim)]


def dfs(dict, i, dict_holder):
    if i >= len(parameter_list):
        dict_holder.append(dict)
        return

    parameter_pair = parameter_list[i]

    if 'pred_dense_layer' not in dict and parameter_pair[0] == 'softmax_dim':
        parameter_pair = ('parameter_pair', 64)
    # next idx
    i += 1
    if isinstance(parameter_pair[1], list):
        for value in parameter_pair[1]:
            new_dict = copy.copy(dict)
            new_dict[parameter_pair[0]] = value
            dfs(new_dict, i, dict_holder)
    else:
        dict[parameter_pair[0]] = parameter_pair[1]
        dfs(dict, i, dict_holder)
    return


dict_holder = []
dfs({}, 0, dict_holder)


def delete_useless_parameter(d):
    if not d['pred_dense_layer']:
        del d['pred_dense_layer']
    else:
        d['pred_dense_layer'] = ''
    if not d['leaf_lstm']:
        del d['leaf_lstm']
        del d['bi_leaf_lstm']
    else:
        d['leaf_lstm'] = ''
        d['bi_leaf_lstm'] = ''
    return d


for d in dict_holder:
    delete_useless_parameter(d)


def change_model_mode(dir, mode):
    new_dir = copy.copy(dir)
    if mode != '':
        new_dir['model_mode'] = mode + new_dir['model_mode']
        new_dir['td_name'] = str.lower(mode) + '_' + new_dir['td_name']
    else:
        new_dir['td_name'] = 'base_' + new_dir['td_name']
    return new_dir


def change_td_name(dir):
    td_name = dir['td_name']
    hidden_part = 'h' + str(dir['hidden_size'])
    batch_part = 'b' + str(dir['batch_size'])
    if 'leaf_lstm' in dir:
        leaf_part = 'bi' + str.lower(dir['leaf_rnn_mode'])
    else:
        leaf_part = None
    if 'pred_dense_layer' in dir:
        pred_part = 'pred' + str(dir['softmax_dim'])
    else:
        pred_part = 'pred0'
    if dir['p_tree'] != 0.0:
        ptree_part = 'ptree' + str(dir['p_tree'])
    else:
        ptree_part = None
    prefix = hidden_part + '_' + batch_part + '_' + pred_part + '_'
    if leaf_part is not None:
        prefix += leaf_part + '_'
    if ptree_part is not None:
        prefix += ptree_part + '_'
    td_name = prefix + td_name
    dir['td_name'] = td_name


crf_holder = []
bicrf_holder = []
lveg_holder = []
for d in dict_holder:
    change_td_name(d)
    crf_holder.append(change_model_mode(copy.copy(d), 'CRF'))
    bicrf_holder.append(change_model_mode(copy.copy(d), 'BiCRF'))
    lveg_holder.append(change_model_mode(copy.copy(d), 'LVeG'))


def dict2str(dict, gpu_id):
    prefix = '#!/usr/bin/env bash\nCUDA_VISIBLE_DEVICES=' + str(gpu_id) + ' python example/sentiment.py '
    for key in dict.keys():
        prefix += '--' + key + ' ' + str(dict[key]) + ' '
    return prefix


str_dict = {}
crf_dict = {}
bicrf_dict = {}
lveg_dict = {}
cnt = 0
for d in dict_holder:
    str_dict[d['td_name']] = dict2str(d, gpu_prefix + int(cnt % max_gpu_id))
    cnt += 1
cnt = 0
for d in crf_holder:
    crf_dict[d['td_name']] = dict2str(d, gpu_prefix + int(cnt % max_gpu_id))
    cnt += 1
cnt = 0
for d in bicrf_holder:
    bicrf_dict[d['td_name']] = dict2str(d, gpu_prefix + int(cnt % max_gpu_id))
    cnt += 1
cnt = 0
for d in lveg_holder:
    lveg_dict[d['td_name']] = dict2str(d, gpu_prefix + int(cnt % max_gpu_id))
    cnt += 1


def save_sample(dir, key, value):
    if not os.path.exists(dir):
        os.makedirs(dir)

    with open(dir + '/' + key + '.sh', 'w') as f:
        f.write(value)


def save_run_script(node_idx, run_batch, str_dict, self_dir, scrpt_dir, script_name):
    prefix = '#!/usr/bin/env bash\n'
    pbs_data = "#PBS -l walltime=1000:00:00 \n#PBS -N node" + str(node_idx) + " \n#PBS -l nodes=sist-gpu" + str(
        node_idx) + \
               ":ppn=1 \n#PBS -S /bin/bash \n#PBS -k oe \n#PBS -q sist-tukw \n#PBS -u zhanglw\n"
    conda_prefix = 'source activate allen\n'
    cd_prefix = 'cd ' + model_dir + ' \n'

    if not os.path.exists(self_dir):
        os.makedirs(self_dir)

    with open(self_dir + '/' + script_name, 'w') as f:
        f.write(prefix)
        f.write(pbs_data)
        f.write(conda_prefix)
        f.write(cd_prefix)
        f.write('\n')

        cnt = 0
        for d in str_dict.keys():
            cnt += 1
            if cnt % run_batch == 0 or cnt == len(str_dict):
                f.write('sh ' + scrpt_dir + '/' + d + '.sh\n')
                f.write('sleep 10 \n')
            else:
                f.write('sh ' + scrpt_dir + '/' + d + '.sh &\n')


for key in str_dict.keys():
    save_sample(script_dir, key, str_dict[key])

for key in crf_dict.keys():
    save_sample(script_dir, key, crf_dict[key])

for key in bicrf_dict.keys():
    save_sample(script_dir, key, bicrf_dict[key])

for key in lveg_dict.keys():
    save_sample(script_dir, key, lveg_dict[key])

save_run_script(pbs_id[0], run_batch, str_dict, run_dir, script_dir, 'run' + str(run_suffix[0]) + '.sh')
save_run_script(pbs_id[1], run_batch, crf_dict, run_dir, script_dir, 'run' + str(run_suffix[1]) + '.sh')
save_run_script(pbs_id[2], run_batch, bicrf_dict, run_dir, script_dir, 'run' + str(run_suffix[2]) + '.sh')
save_run_script(pbs_id[3], run_batch, lveg_dict, run_dir, script_dir, 'run' + str(run_suffix[3]) + '.sh')
# base_dict = {}
#
# if leaf_lstm:
#     base_dict['leaf_lstm'] = '--leaf_lstm '
#     if bi_leaf_lstm:
#         base_dict['bi_leaf_lstm'] = '--bi_leaf_lstm '
#
# base_dict['leaf_rnn_mode'] = '--leaf_rnn_mode ' + leaf_rnn_mode + ' '
# base_dict['leaf_rnn_num'] = '--leaf_rnn_num ' + str(leaf_rnn_num) + ' '
# base_dict['tree_mode'] = '--tree_mode ' + tree_mode + ' '
# base_dict['pred_mode'] = '--pred_mode ' + pred_mode + ' '
# # alert the model model specialize for different file
# base_dict['batch_size'] = '--batch_size ' + str(batch_size) + ' '
# base_dict['epoch'] = '--epoch ' + str(epoch) + ' '
# base_dict['hidden_size'] = '--hidden_size ' + str(hidden_size) + ' '
# base_dict['softmax_dim'] = '--softmax_dim ' + str(softmax_dim) + ' '
# if pred_dense_layer:
#     base_dict['pred_dense_layer'] = '--pred_dense_layer '
# base_dict['optim_method'] = '--optim_method ' + optim_method + ' '
# base_dict['learning_rate'] = '--learning_rate ' + str(lr) + ' '
# base_dict['momentum'] = '--momentum ' + str(momentum) + ' '
# base_dict['decay_rate'] = '--decay_rate ' + str(decay_rate) + ' '
# base_dict['gamma'] = '--gamma ' + str(gamma) + ' '
# base_dict['schedule'] = '--schedule ' + str(schedule) + ' '
# base_dict['embedding'] = '--embedding ' + embedding + ' '
# base_dict['embedding_path'] = '--embedding_path ' + embedding_path + ' '
# base_dict['train'] = '--train ' + train + ' '
# base_dict['dev'] = '--dev ' + dev + ' '
# base_dict['test'] = '--test ' + test + ' '
# base_dict['num_labels'] = '--num_labels ' + str(num_labels) + ' '
# base_dict['p_in'] = '--p_in ' + str(p_in) + ' '
# base_dict['p_leaf'] = '--p_leaf ' + str(p_leaf) + ' '
# base_dict['p_tree'] = '--p_tree ' + str(p_tree) + ' '
# base_dict['p_pred'] = '--p_pred ' + str(p_pred) + ' '
#
# if tensorboard:
#     base_dict['tensorboard'] = '--tensorboard '
# base_dict['td_dir'] = '--td_dir ' + td_dir + ' '
#
# if attention:
#     base_dict['attention'] = '--attention '
# base_dict['coattention_dim'] = '--coattention_dim ' + str(coattention_dim) + ' '
#
# base_dict['elmo'] = '--elmo ' + elmo + ' '
# base_dict['elmo_weight'] = '--elmo_weight ' + elmo_weight + ' '
# base_dict['elmo_config'] = '--elmo_config ' + elmo_config + ' '
#
# crf_dict = copy.copy(base_dict)
# bicrf_dict = copy.copy(base_dict)
# lveg_dict = copy.copy(base_dict)
#
# base_dict['model_mode'] = '--model_mode ' + model_mode + ' '
# crf_dict['model_mode'] = '--model_mode CRF' + model_mode + ' '
# bicrf_dict['model_mode'] = '--model_mode BiCRF' + model_mode + ' '
# lveg_dict['model_mode'] = '--model_mode LVeG' + model_mode + ' '
#
# base_dict['td_name'] = '--td_name base-' + td_name + ' '
# crf_dict['td_name'] = '--td_name crf-' + td_name + ' '
# bicrf_dict['td_name'] = '--td_name bicrf-' + td_name + ' '
# lveg_dict['td_name'] = '--td_name lveg-' + td_name + ' '
#
# lveg_list = []
# for comp in lveg_comp:
#     for dim in gaussian_dim:
#         tmp_script = copy.copy(lveg_dict)
#         tmp_script['lveg_comp'] = '--lveg_comp ' + str(comp) + ' '
#         tmp_script['gaussian_dim'] = '--gaussian_dim ' + str(dim) + ' '
#         tmp_script['td_name'] = tmp_script['td_name'].strip() + '_c' + str(comp) + 'd' + str(dim) + ' '
#         lveg_list.append(tmp_script)
#
# prefix = '#!/usr/bin/env bash\nCUDA_VISIBLE_DEVICES=' + str(gpu_id) + ' python example/sentiment.py '
#
# scripts_dict = {}
# # encoding base
# script = copy.copy(prefix)
# for key in base_dict.keys():
#     script += base_dict[key]
#
# name = 'base_' + script_name
# scripts_dict[name] = script
#
# # encoding crf
# script = copy.copy(prefix)
# for key in crf_dict.keys():
#     script += crf_dict[key]
#
# name = 'crf_' + script_name
# scripts_dict[name] = script
#
# # encoding bicrf
# script = copy.copy(prefix)
# for key in bicrf_dict.keys():
#     script += bicrf_dict[key]
#
# name = 'bicrf_' + script_name
# scripts_dict[name] = script
#
# # encoding lveg
# cnt = 0
# for lveg_dict in lveg_list:
#
#     script = copy.copy(prefix)
#     for key in lveg_dict.keys():
#         script += lveg_dict[key]
#
#     name = 'lveg_' + str(cnt) + '_' + script_name
#     scripts_dict[name] = script
#     cnt += 1
#
# # write script
# for name in scripts_dict.keys():
#     with open(name + '.sh', 'w') as f:
#         f.write(scripts_dict[name])
#
# # write batch run script
# if not os.path.exists('script'):
#     os.makedirs('script')
#
# for i in range(run_num):
#     with open('script/run' + str(run_prefix + i) + '.sh', 'w') as f:
#         f.write('#!/usr/bin/env bash\n')
#         f.write(pbs_data + '\n')
#         f.write('source activate allen\n')
#         f.write('cd ' + model_dir + '\n')
#
# cnt = 0
# for key in scripts_dict.keys():
#     with open('script/run' + str(run_prefix + cnt % run_num) + '.sh', 'a+') as f:
#         f.write('sh example/' + key + '.sh')
#         f.write('\n')
#         f.write('sleep 5\n')
#     cnt += 1
