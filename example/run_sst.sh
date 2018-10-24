#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python examples/sentiment.py --leaf-lstm --bi-leaf-lstm --leaf-rnn-mode LSTM --tree-mode SLSTM \
--pred-mode single_h --batch-size 16 --epoch 50 --hidden-size 150 --softmax-dim 64 --optim-method SGD \
--learning-rate 0.01 --momentum 0.9 --decay-rate 0.05 --gamma 0.0 --schedule 5 --embedding glove \
--train /home/ehaschia/Code/dataset/sst/trees/train.txt --dev /home/ehaschia/Code/dataset/sst/trees/dev.txt \
--test /home/ehaschia/Code/dataset/sst/trees/test.txt --num-labels 5 --p_in 0.5 --p_leaf 0.5 --p_tree 0.5 --p_pred 0.5