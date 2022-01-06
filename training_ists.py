"""
The system trains BERT (or any other transformer model like RoBERTa, DistilBERT etc.) on the SNLI + MultiNLI (AllNLI) dataset
with softmax loss function. At every 1000 training steps, the model is evaluated on the
STS benchmark dataset

Usage:
python training_nli.py

OR
python training_nli.py pretrained_transformer_model_name
"""
from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import LabelAccuracyEvaluator
import logging
from datetime import datetime
import sys
import os
import gzip
import csv
from itertools import islice

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#Check if dataset exsist. If not, download and extract  it
# nli_dataset_path = 'data/AllNLI.tsv.gz'
# sts_dataset_path = 'data/stsbenchmark.tsv.gz'

# if not os.path.exists(nli_dataset_path):
#     util.http_get('https://sbert.net/datasets/AllNLI.tsv.gz', nli_dataset_path)

# if not os.path.exists(sts_dataset_path):
#     util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)


train_dataset_path = 'data/task2/semeval/images/images_phrases_train_data.tsv'
test_dataset_path = 'data/task2/semeval/images/images_phrases_test_data.tsv'


#You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
# model_name = sys.argv[1] if len(sys.argv) > 1 else 'bert-base-uncased'

model_name = 'roberta-base'
# config: ścieżki do danych treningowych i testowych, liczba epok, liczba klas, batch_size, evaluation_steps?, warmup steps, train:dev split
# Read the dataset
train_batch_size = 16
split = 0.8

model_save_path = 'output/training_ists_'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
word_embedding_model = models.Transformer(model_name)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                            #    pooling_mode_mean_tokens=True,
                            #    pooling_mode_cls_token=False,
                            #    pooling_mode_max_tokens=False)
                            )

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


# Read the AllNLI.tsv.gz file and create the training dataset
logging.info("Read training dataset")

# label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
# label2int = {"NOALI": 0, "ALIC": 1, "REL": 2, "SIM": 3, "SPE1": 4, "SPE2": 5, "OPPO": 6, "EQUI": 7}

# do poprawy
label2int = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}

train_samples = []
dev_samples = []

examples_num = 0
with open(train_dataset_path, 'rt', encoding='utf8') as fIn:
    examples_num = sum(1 for line in fIn)

with open(train_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    header = next(reader)
    to_eval_counter = int(examples_num * split)
    print("First example for evaluation: {}".format(int(to_eval_counter * split)))
    for row in reader: 
        
        # if row['split'] == 'train':
        #     label_id = label2int[row['label']]
        #     train_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=label_id))

        print(row)
        label_id = label2int[row['y_score']]
        #score = float(row['y_score']) / 5.0 #Normalize score to range 0 ... 1
        if (to_eval_counter > 0):
            train_samples.append(InputExample(texts=[row['x1'], row['x2']], label=label_id))
            to_eval_counter -= 1
        else: 
            dev_samples.append(InputExample(texts=[row['x1'], row['x2']], label=label_id))


train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
train_loss = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=5) 


#Read STSbenchmark dataset and use it as development set
# logging.info("Read dev dataset")
# dev_samples = []
# with open(train_dataset_path, 'rt', encoding='utf8') as fIn:
#     f_len = sum(1 for line in fIn)
#     reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
#     #header = next(reader)
#     for row in islice(reader, int(f_len * split), None):
#         # if row['split'] == 'dev':
#         #     score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
#         #     dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))
        
#         print(row)
#         label_id = label2int[row['y_score']]
#         dev_samples.append(InputExample(texts=[row['x1'], row['x2']], label=label_id))

dev_dataloader = DataLoader(dev_samples, shuffle=True, batch_size=train_batch_size) # inny batch size?
dev_evaluator = LabelAccuracyEvaluator(dev_dataloader, name='sts-dev', softmax_model=model)

# Configure the training
num_epochs = 1

warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))



# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=dev_evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path
          )



##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################

test_samples = []
with open(test_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        # if row['split'] == 'test':
        #     score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
        #     test_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))
        print(row)
        label_id = label2int[row['y_score']]
        test_samples.append(InputExample(texts=[row['x1'], row['x2']], label=label_id))

model = SentenceTransformer(model_save_path)
test_evaluator = LabelAccuracyEvaluator.from_input_examples(test_samples, batch_size=train_batch_size, name='ists-test')
test_evaluator(model, output_path=model_save_path)
