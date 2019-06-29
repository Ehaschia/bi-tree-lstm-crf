#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python example/evaluate_elmo.py \
--embedding_path /path/to/glove/glove.840B.300d.txt \
--train /path/to/SST/train.txt \
--dev /path/to/SST/dev.txt \
--test /path/to/SST/test.txt \
--elmo_input --td_dir /public/sist/home/zhanglw/code/sentiment/bi-tree-lstm-crf/tmp \
--gaussian_dim 1 --component_num 2 --random_seed 48 \
--elmo_weight /path/to/elmo/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5 \
--elmo_config /path/to/elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json \
--model_name /path/to/model/model_name