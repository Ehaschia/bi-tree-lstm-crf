# Latent Variable Sentiment Grammar

This repo contains code implementations for  ACL 2019 paper [*Latent Variable Sentiment Grammar*](https://arxiv.org/pdf/1907.00218.pdf)

## Requirements

* python==3.7.1
* pytorch==0.4.1
* allennlp==0.8.0
* tensorboardX==1.4
* glove and ELMo for pre-trained vector
* Stanford Sentiment Treebank (SST) for training or testing

## Training and testing

Please check the `run_XXX.sh` in the `example` folder, where `XXX` corresponds to the names of the kind of models.

## Pretrained Models

For the model with ELMo achieved the best score on the fine-grained root level sentiment classification, you can download it via the [Google drive link](https://drive.google.com/open?id=1jyoLFs0Ivwn_8yJWQ49eBZDzUn5Rw4aP)
