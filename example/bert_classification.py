import torch
from pytorch_pretrained_bert import BertForSequenceClassification
from pytorch_pretrained_bert import BertAdam, BertTokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer

from allennlp.data.dataset_readers import StanfordSentimentTreeBankDatasetReader
from tqdm import tqdm
from random import shuffle
import time

root_dir = '/home/ehaschia/Code/bi-tree-lstm-crf/'
data_dir = '/home/ehaschia/Code/dataset/'
bert_mode = 'bert-base-uncased'
lower = bert_mode.find('uncased') != -1

gradient_accumulation_steps = 1
num_train_epochs = 30
train_batch_size = 8
train_batch_size = int(train_batch_size / gradient_accumulation_steps)

token_indexers = {'tokens': SingleIdTokenIndexer()}
train_reader = StanfordSentimentTreeBankDatasetReader(token_indexers=token_indexers, use_subtrees=False)
dev_reader = StanfordSentimentTreeBankDatasetReader(token_indexers=token_indexers, use_subtrees=False)

allen_train_dataset = train_reader.read(data_dir + '/sst/trees/train.txt')
allen_dev_dataset = dev_reader.read(data_dir + '/sst/trees/dev.txt')
allen_test_dataset = dev_reader.read(data_dir + '/sst/trees/test.txt')
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

bert_tokenizer = BertTokenizer.from_pretrained(data_dir + bert_mode + '.txt', do_lower_case=lower)


def convert_allen_to_list(allen_dataset):
    dataset = []
    for instance in allen_dataset:
        if int(instance['label'].label) == 2:
            continue
        int_label = 1 if int(instance['label'].label) > 2 else 0
        label = int_label
        str_sentence = [token.text for token in instance['tokens'].tokens]
        str_sentence = ['[CLS]'] + bert_tokenizer.tokenize(' '.join(str_sentence)) + ['[SEP]']
        bert_token = bert_tokenizer.convert_tokens_to_ids(str_sentence)
        dataset.append((label, bert_token))
    return dataset


def dataset_to_batch(dataset, batch_size):
    def batch_to_tensor(batch):
        labels = []
        bert_tokens = []
        max_len = 0
        for instance in batch:
            label, bert_token = instance
            labels.append(label)
            bert_tokens.append(bert_token)
            if len(bert_token) > max_len:
                max_len = len(bert_token)
        masks = []
        expand_bert_tokens = []
        for bert_token in bert_tokens:
            bert_token += [0] * (max_len - len(bert_token))
            expand_bert_tokens.append(bert_token)
            mask = [1] * len(bert_token) + [0] * (max_len - len(bert_token))
            masks.append(mask)

        labels = torch.tensor(labels).to(device)
        bert_tokens = torch.tensor(expand_bert_tokens).to(device)
        masks = torch.tensor(masks).to(device)
        sigment_id = torch.zeros_like(masks)
        return labels, bert_tokens, sigment_id, masks

    batch_dataset = []
    batch = []
    for instance in dataset:
        batch.append(instance)
        if len(batch) != batch_size:
            continue
        else:
            batch_dataset.append(batch_to_tensor(batch))
            batch = []
    if len(batch) != 0:
        batch_dataset.append(batch_to_tensor(batch))
    return batch_dataset


train_dataset = convert_allen_to_list(allen_train_dataset)
dev_dataset = convert_allen_to_list(allen_dev_dataset)
test_dataset = convert_allen_to_list(allen_test_dataset)

model = BertForSequenceClassification.from_pretrained(data_dir + bert_mode + '.tar.gz').to(device)
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
dev_acc = 0.0
test_acc = 0.0
for i in range(num_train_epochs):
    shuffle(train_dataset)
    shuffle(dev_dataset)
    shuffle(test_dataset)
    batch_train = dataset_to_batch(train_dataset, train_batch_size)
    batch_dev = dataset_to_batch(dev_dataset, train_batch_size)
    batch_test = dataset_to_batch(test_dataset, train_batch_size)
    print("Epoch " + str(i) + "/" + str(num_train_epochs))
    total_loss = 0.0
    time.sleep(1)
    model.train()
    for train_data in tqdm(batch_train):
        label, sent, segment, mask = train_data
        loss = model(sent, segment, mask, label)
        loss[0].backward()
        total_loss += loss[0].item()
        optimizer.step()
        optimizer.zero_grad()
    print(total_loss)
    model.eval()
    total_corr = 0.0
    for dev_data in tqdm(batch_dev):
        label, sent, segment, mask = dev_data
        loss = model(sent, segment, mask, label)
        pred_label = torch.argmax(loss[1], dim=1)
        corr = torch.eq(pred_label, label).sum().cpu().data.numpy()
        total_corr += corr
    # if total_corr / len(dev_dataset) > dev_acc:
    if True:
        dev_acc = total_corr / len(dev_dataset)
        total_test_corr = 0.0
        for test_data in tqdm(batch_test):
            label, sent, segment, mask = test_data
            loss = model(sent, segment, mask, label)
            pred_label = torch.argmax(loss[1], dim=1)
            corr = torch.eq(pred_label, label).sum().cpu().data.numpy()
            total_test_corr += corr
        # if total_test_corr / len(test_dataset) > test_acc:
        test_acc = total_test_corr / len(test_dataset)
    print("DEV ACC is %.2f at epoch %d" % (dev_acc, i))
    print("TEST ACC is %.2f at epoch %d" % (test_acc, i))