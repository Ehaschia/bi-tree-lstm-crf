from allennlp.models.archival import load_archive
from allennlp.common.util import prepare_environment
from allennlp.commands.evaluate import evaluate
from allennlp.data.dataset_readers import StanfordSentimentTreeBankDatasetReader
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer, SingleIdTokenIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.iterators import BucketIterator
from allennlp.commands.train import train_model_from_file


save_dir = '/home/ehaschia/Code/bi-tree-lstm-crf'
data_path = '/home/ehaschia/Code/dataset/'
cuda_id = 0

token_indexers = {'tokens': SingleIdTokenIndexer(),
                  'elmo': ELMoTokenCharactersIndexer()}
train_reader = StanfordSentimentTreeBankDatasetReader(token_indexers=token_indexers, use_subtrees=False)


train_dataset = train_reader.read(data_path + '/sst/trees/train.txt')
dev_dataset = train_reader.read(data_path + '/sst/trees/dev.txt')
test_dataset = train_reader.read(data_path + '/sst/trees/test.txt')

vocab = Vocabulary.from_instances(train_dataset + dev_dataset,
                                  min_count={'tokens': 3})
iterator = BucketIterator(batch_size=100, sorting_keys=[("tokens", "num_tokens")])

archive_file = save_dir + '/best.th'
archive = load_archive(archive_file, cuda_id)
config = archive.config
prepare_environment(config)
model = archive.model
model.eval()
evaluate(model, test_dataset, iterator, cuda_id)

# train_model_from_file('/home/ehaschia/Code/bi-tree-lstm-crf/biattentive_classification_network_elmo.jsonnet',
#                       '/home/ehaschia/Code/bi-tree-lstm-crf/debug')