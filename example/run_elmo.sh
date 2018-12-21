#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python example/sentiment_be.sh --model_mode elmo_crf --optim_method Adam --learning_rate 0.001 \
--embedding glove --embedding_path /home/ehaschia/Code/for_liwen_zhang/data/glove.sentiment.large.pretrained.vec \
--train /home/ehaschia/Code/dataset/sst/trees/train_part.txt --dev /home/ehaschia/Code/dataset/sst/trees/train_part.txt \
--test /home/ehaschia/Code/dataset/sst/trees/train_part.txt --elmo --elmo_input --bert none --td_dir /home/ehaschia/Code/bi-tree-lstm-crf/tmp