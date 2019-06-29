#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python example/sentiment.py --leaf_rnn_mode LSTM --leaf_rnn_num 1 --tree_mode TreeLSTM \
--pred_mode single_h --batch_size 32 --epoch 15 --hidden_size 300 --softmax_dim 64 --optim_method Adam \
--learning_rate 0.001 --momentum 0.9 --decay_rate 0.1 --gamma 0.0 --schedule 5 --embedding glove \
--embedding_path /path/to/glove/glove.840B.300d.txt \
--train /path/to/SST/train.txt \
--dev /path/to/SST/dev.txt \
--test /path/to/SST/test.txt \
--num_labels 5 --p_in 0.5 --p_leaf 0.0 --p_tree 0.0 --p_pred 0.0 \
--td_dir /dir/to/save/log/  --elmo none \
--elmo_weight /path/to/elmo/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5 \
--elmo_config /path/to/elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json \
--model_mode LVeGTreeLSTM --td_name lveg-tlstm-b32 --lveg_comp 1 --gaussian_dim 2
