from allennlp.models import BiattentiveClassificationNetwork

from allennlp.data.dataset_readers import StanfordSentimentTreeBankDatasetReader
from allennlp.modules.token_embedders.embedding import *

from allennlp.common import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules import Elmo, FeedForward, Maxout
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
import torch.optim as optim
from allennlp.data.iterators import BucketIterator
from allennlp.training import Trainer
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer, SingleIdTokenIndexer
from allennlp.models.archival import *
from allennlp.common.util import prepare_environment
from allennlp.commands.evaluate import evaluate

import torch.nn as nn
root_path = '/home/ehaschia/Code/dataset/'
save_dir = '/home/ehaschia/Code/bi-tree-lstm-crf'
cuda_id = 0
token_indexers = {'tokens': SingleIdTokenIndexer(),
                  'elmo': ELMoTokenCharactersIndexer()}

train_reader = StanfordSentimentTreeBankDatasetReader(token_indexers=token_indexers, use_subtrees=False)
dev_reader = StanfordSentimentTreeBankDatasetReader(token_indexers=token_indexers, use_subtrees=False)

train_dataset = train_reader.read(root_path + '/sst/trees/train.txt')
dev_dataset = dev_reader.read(root_path + '/sst/trees/dev.txt')
test_dataset = dev_reader.read(root_path + '/sst/trees/test.txt')

# You can optionally specify the minimum count of tokens/labels.
# `min_count={'tokens':3}` here means that any tokens that appear less than three times
# will be ignored and not included in the vocabulary.
vocab = Vocabulary.from_instances(train_dataset + dev_dataset,
                                  min_count={'tokens': 3})

params = Params({'embedding_dim': 300,
                 'pretrained_file': root_path + '/glove.840B.300d.txt',
                 'trainable': False})

embedding = Embedding.from_params(vocab, params)
embedder = BasicTextFieldEmbedder({'tokens': embedding})

options_file = root_path + "/elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = root_path + "/elmo/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
elmo = Elmo(options_file, weight_file, num_output_representations=1, dropout=0).to(device)
pre_feedforward_layer = FeedForward(input_dim=1324, num_layers=1, hidden_dims=[300], activations=[torch.nn.ReLU()],
                                    dropout=[0.25])

pytorch_encoder = nn.LSTM(input_size=300, hidden_size=300, num_layers=1, bidirectional=True, batch_first=True)
encoder = PytorchSeq2SeqWrapper(pytorch_encoder)
pytorch_integrator = nn.LSTM(input_size=1800, hidden_size=300, num_layers=1, bidirectional=True, batch_first=True)
integrator = PytorchSeq2SeqWrapper(pytorch_integrator)
output_layer = Maxout(input_dim=2400, num_layers=3, output_dims=[1200, 600, 5], pool_sizes=4, dropout=[0.2, 0.3, 0.0])

model = BiattentiveClassificationNetwork(vocab, embedder, embedding_dropout=0.5,
                                         pre_encode_feedforward=pre_feedforward_layer, encoder=encoder,
                                         integrator=integrator, integrator_dropout=0.1,
                                         output_layer=output_layer, elmo=elmo, use_input_elmo=True,
                                         use_integrator_output_elmo=False)
optimizer = optim.Adam(model.parameters(), lr=0.001)
iterator = BucketIterator(batch_size=100, sorting_keys=[("tokens", "num_tokens")])
iterator.index_with(vocab)

trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_dataset,
                  validation_dataset=dev_dataset,
                  patience=5,
                  num_epochs=20,
                  grad_norm=5,
                  validation_metric='+accuracy',
                  cuda_device=cuda_id,
                  serialization_dir=save_dir)

trainer.train()
print('*'*20 + ' EVALUATE with Best Epoch ' + "*"*20)
evaluate(model, test_dataset, iterator, cuda_id)