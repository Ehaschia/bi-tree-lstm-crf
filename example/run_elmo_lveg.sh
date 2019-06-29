#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python example/sentiment_be.py --model_mode elmo_lveg --optim_method Adam
--learning_rate 0.001 --embedding glove \
--embedding_path /path/to/glove/glove.840B.300d.txt \
--train /path/to/SST/train.txt \
--dev /path/to/SST/dev.txt \
--test /path/to/SST/test.txt \
--elmo_input --td_dir /dir/to/save/log/ \
--gaussian_dim 1 --component_num 2 --random_seed 48 --batch_size 8 \
--elmo_weight /path/to/elmo/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5 \
--elmo_config /path/to/elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json