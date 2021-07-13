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