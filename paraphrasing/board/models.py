from django.db import models
import gensim.downloader
from gensim.models import Word2Vec
from collections import Counter
from tqdm import tqdm
import numpy as np
import torch
from transformers import BertTokenizer, BertForMaskedLM
import matplotlib.pyplot as plt

# Create your models here.
wget http://www.thespermwhale.com/jaseweston/babi/CBTest.tgz
tar -xzf ./CBTest.tgz
DATA_PATH=./CBTest/data; cat ${DATA_PATH}/cbt_train.txt ${DATA_PATH}/cbt_valid.txt ${DATA_PATH}/cbt_test.txt > ./cbt_all.txt

#@title Build a table of word frequency
def count_lines(path):
  with open(path, 'r') as f:
    return sum([1 for _ in f])

word_frequency = Counter()
filepath = './cbt_all.txt'
n_lines = count_lines(filepath)
with open(filepath, 'r') as f:
  for line in tqdm(f, total=n_lines):
    if line.startswith("_BOOK_TITLE_"):
      continue
    else:
      tokens = tokenizer.tokenize(line.rstrip())
      for token in tokens:
        word_frequency[token] += 1
        
def model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    # Model's input
    text = "[CLS] the cat perched on the mat [SEP] the cat perched on the mat [SEP]"
    masked_idx = 10

    # Tokenize a text
    tokenized_text = tokenizer.tokenize(text)

    # Mask a complex token which should be substituted
    complex_word = tokenized_text[masked_idx]
    tokenized_text[masked_idx] = '[MASK]'

    #Convert inputs to PyTorch tensors
    tokens_ids = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    tokens_tensor = torch.tensor([tokens_ids])
    segments_tensors = torch.tensor([segments_ids])

    with torch.no_grad():
        outputs = model(tokens_tensor, token_type_ids=segments_tensors)
        predictions = outputs[0]

    # Output top 10 of candidates
    topk_score, topk_index = torch.topk(predictions[0, masked_idx], 10)
    topk_tokens = tokenizer.convert_ids_to_tokens(topk_index.tolist())
    print(f'Input: {" ".join(tokenized_text)}')
    print(f'Top10: {topk_tokens}')

model()