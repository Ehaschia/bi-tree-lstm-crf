#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python example/sentiment_be.py --model_mode elmo_lveg --optim_method Adam
--learning_rate 0.001 --embedding glove \
--embedding_path /home/liwenzhang/code/data/glove.840B.300d.txt \
--train /home/liwenzhang/code/data/trees/train.txt \
--dev /home/liwenzhang/code/data/trees/dev.txt \
--test /home/liwenzhang/code/data/trees/test.txt \
--elmo_input --td_dir /home/ehaschia/Code/bi-tree-lstm-crf/tmp