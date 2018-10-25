#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python example/sentiment.py --leaf_lstm --bi_leaf_lstm --leaf_rnn_mode LSTM \
--tree_mode BUTreeLSTM --pred_mode avg_h --model_mode CRFBiTreeLSTM --batch_size 16 --epoch 50 \
--hidden_size 150 --softmax_dim 64 --optim_method Adam --learning_rate 0.001 --momentum 0.9 \
--decay_rate 0.05 --gamma 0.0 --schedule 5 --embedding glove \
--embedding_path /home/ehaschia/Code/dataset/glove/glove.6B.300d.txt \
--train /home/ehaschia/Code/dataset/sst/trees/train.txt \
--dev /home/ehaschia/Code/dataset/sst/trees/dev.txt \
--test /home/ehaschia/Code/dataset/sst/trees/test.txt \
--num_labels 5 --p_in 0.5 --p_leaf 0.5 --p_tree 0.5 --p_pred 0.5