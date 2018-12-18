import torch
from pytorch_pretrained_bert import BertForSequenceClassification
from pytorch_pretrained_bert import BertAdam, BertTokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer

from allennlp.data.dataset_readers import StanfordSentimentTreeBankDatasetReader

token_indexers = {'tokens': SingleIdTokenIndexer()}
train_reader = StanfordSentimentTreeBankDatasetReader(token_indexers=token_indexers, use_subtrees=False)
dev_reader = StanfordSentimentTreeBankDatasetReader(token_indexers=token_indexers, use_subtrees=False)

allen_train_dataset = train_reader.read('/home/ehaschia/Code/dataset/sst/trees/train.txt')
allen_dev_dataset = dev_reader.read('/home/ehaschia/Code/dataset/sst/trees/dev.txt')
allen_test_dataset = dev_reader.read('/home/ehaschia/Code/dataset/sst/trees/test.txt')

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

bert_tokenizer = BertTokenizer.from_pretrained('/home/ehaschia/Code/dataset/bert-base-uncased.txt')

gradient_accumulation_steps = 1
num_train_epochs = 10
train_batch_size = 32
train_batch_size = int(train_batch_size / gradient_accumulation_steps)


def convert_allen_to_list(allen_dataset):
    dataset = []
    for instance in allen_dataset:
        # fixme may>=2
        int_label = [1] if int(instance['label'].label) > 2 else [0]
        label = torch.tensor(int_label).to(device)
        str_sentence = [token.text for token in instance['tokens'].tokens]
        str_sentence = ['[CLS]'] + bert_tokenizer.tokenize(' '.join(str_sentence)) + ['[SEP]']
        sentence_len = len(str_sentence)
        bert_token = torch.tensor([bert_tokenizer.convert_tokens_to_ids(str_sentence)]).to(device)
        sengment_id = torch.tensor([0]*sentence_len).to(device)
        dataset.append((label, bert_token, sengment_id))
    return dataset


train_dataset = convert_allen_to_list(allen_train_dataset)
dev_dataset = convert_allen_to_list(allen_dev_dataset)
test_dataset = convert_allen_to_list(allen_test_dataset)

model = BertForSequenceClassification.from_pretrained('/home/ehaschia/Code/dataset/bert-base-uncased.tar.gz').to(device)
param_optimizer = list(model.named_parameters())

no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
num_train_steps = int(
    len(train_dataset) / train_batch_size / gradient_accumulation_steps * num_train_epochs)
t_total = num_train_steps
optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=0.0005,
                     warmup=0.1,
                     t_total=t_total)

for idx, i in enumerate(train_dataset):
    label, sent, segment = i
    loss = model(sent, segment, None, label)
    print(loss[0].item())
    loss[0].backward()
    if idx % train_batch_size == 0 and idx != 0:
        optimizer.step()
        optimizer.zero_grad()
        print('BP')