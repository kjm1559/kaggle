import torch
import pickle
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.nn import CrossEntropyLoss, MSELoss

from tqdm import tqdm, trange
import os
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np

#----------------------------------------------------------------------------------

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'


# The input data dir. Should contain the .tsv files (or other data files) for the task.
DATA_DIR = "data/"

# Bert pre-trained model selected in the list: bert-base-uncased, 
# bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased,
# bert-base-multilingual-cased, bert-base-chinese.
BERT_MODEL = 'bert-base-multilingual-uncased'#'bert-base-cased'

# The name of the task to train.I'm going to name this 'yelp'.
TASK_NAME = 'yelp'

# The output directory where the fine-tuned model and checkpoints will be written.
OUTPUT_DIR = f'outputs/{TASK_NAME}/'

# The directory where the evaluation reports will be written to.
REPORTS_DIR = f'reports/{TASK_NAME}_evaluation_report/'

# This is where BERT will look for pre-trained models to load parameters from.
CACHE_DIR = 'cache/'

# The maximum total input sequence length after WordPiece tokenization.
# Sequences longer than this will be truncated, and sequences shorter than this will be padded.
MAX_SEQ_LENGTH = 128

TRAIN_BATCH_SIZE = 24
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 1
RANDOM_SEED = 42
GRADIENT_ACCUMULATION_STEPS = 1
WARMUP_PROPORTION = 0.1
OUTPUT_MODE = 'classification'

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"

#----------------------------------------------------------------------------------

output_mode = OUTPUT_MODE

cache_dir = CACHE_DIR

if os.path.exists(REPORTS_DIR) and os.listdir(REPORTS_DIR):
        REPORTS_DIR += f'/report_{len(os.listdir(REPORTS_DIR))}'
        os.makedirs(REPORTS_DIR)
if not os.path.exists(REPORTS_DIR):
    os.makedirs(REPORTS_DIR)
    REPORTS_DIR += f'/report_{len(os.listdir(REPORTS_DIR))}'
    os.makedirs(REPORTS_DIR)
    
if os.path.exists(OUTPUT_DIR) and os.listdir(OUTPUT_DIR):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(OUTPUT_DIR))
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

#----------------------------------------------------------------------------------
try:
    train_examples = pd.read_pickle('./outputs/jigsaw-unintended-bias-train-processed-seqlen128.pickle')
except:
    train_examples = pd.read_csv('jigsaw-unintended-bias-train-processed-seqlen128.csv', usecols=['input_word_ids', 'input_mask', 'all_segment_id', 'toxic'])
    train_examples['input_word_ids'] = [eval(dd) for dd in train_examples['input_word_ids'].values]
    train_examples['input_mask'] = [eval(dd) for dd in train_examples['input_mask'].values]
    train_examples['all_segment_id'] = [eval(dd) for dd in train_examples['all_segment_id'].values]
    train_examples.to_pickle('./outputs/jigsaw-unintended-bias-train-processed-seqlen128.pickle')
train_examples_len = len(train_examples)

#----------------------------------------------------------------------------------

# fine tunning
# Load pre-trained model (weights)
num_labels = 2
num_train_optimization_steps = int(
    train_examples_len / TRAIN_BATCH_SIZE / GRADIENT_ACCUMULATION_STEPS) * NUM_TRAIN_EPOCHS
model = BertForSequenceClassification.from_pretrained(BERT_MODEL, cache_dir=CACHE_DIR, num_labels=num_labels)
# model = BertForSequenceClassification.from_pretrained(CACHE_DIR + 'cased_base_bert_pytorch.tar.gz', cache_dir=CACHE_DIR, num_labels=num_labels)

if device != 'cpu':
    model.to(device)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=LEARNING_RATE,
                     warmup=WARMUP_PROPORTION,
                     t_total=num_train_optimization_steps)

global_step = 0
nb_tr_steps = 0
tr_loss = 0

# logger.info("***** Running training *****")
# logger.info("  Num examples = %d", train_examples_len)
# logger.info("  Batch size = %d", TRAIN_BATCH_SIZE)
# logger.info("  Num steps = %d", num_train_optimization_steps)
# all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
# all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
# all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)

all_input_ids = torch.tensor(train_examples['input_word_ids'], dtype=torch.long)
print('input_word_ids complete')
all_input_mask = torch.tensor(train_examples['input_mask'], dtype=torch.long)
print('input_word_ids complete')
all_segment_ids = torch.tensor(train_examples['all_segment_id'], dtype=torch.long)
print('input_word_ids complete')

if device != 'cpu':
    all_segment_ids = all_segment_ids.to(device)
    all_input_mask = all_input_mask.to(device)
    all_input_ids = all_input_ids.to(device)
    print('gpu setting')

if OUTPUT_MODE == "classification":
    all_label_ids = torch.tensor(train_examples.toxic, dtype=torch.long)
elif OUTPUT_MODE == "regression":
    all_label_ids = torch.tensor(train_examples.toxic, dtype=torch.float)

del train_examples
import gc
gc.collect()
#----------------------------------------------------------------------------------

# data loading
train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=TRAIN_BATCH_SIZE)

#----------------------------------------------------------------------------------

# training
model.train()
for _ in trange(int(NUM_TRAIN_EPOCHS), desc="Epoch"):
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        # print(input_ids, input_mask, segment_ids)

        logits = model(input_ids, segment_ids, input_mask, labels=None)

        if OUTPUT_MODE == "classification":
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        elif OUTPUT_MODE == "regression":
            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1), label_ids.view(-1))

        if GRADIENT_ACCUMULATION_STEPS > 1:
            loss = loss / GRADIENT_ACCUMULATION_STEPS

        loss.backward()
        # check accuray
        logits[logits >= 0.5] = 1
        logits[logits < 0.5] = 0
        acc = len(logits[lable_ids == logits]) / len(logits)
        print("\r%f, %f" % [loss, acc], end='')
        
        tr_loss += loss.item()
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1
        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            
model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

# If we save using the predefined names, we can load using `from_pretrained`
output_model_file = os.path.join(OUTPUT_DIR, WEIGHTS_NAME)
output_config_file = os.path.join(OUTPUT_DIR, CONFIG_NAME)

torch.save(model_to_save.state_dict(), output_model_file)
model_to_save.config.to_json_file(output_config_file)
tokenizer.save_vocabulary(OUTPUT_DIR)