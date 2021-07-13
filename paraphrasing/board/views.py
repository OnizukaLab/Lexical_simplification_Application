from django.shortcuts import render
from django.http import HttpResponse
import gensim.downloader
from gensim.models import Word2Vec
from collections import Counter
from tqdm import tqdm
import numpy as np
import torch
from pytorch_transformers import (
    BertTokenizer,
    BertForMaskedLM,
)
import matplotlib.pyplot as plt

# Create your views here.
def index(request):
    params= {
        'msg1':'',
        'msg2':'',
    }
    return render(request, 'board/index.html', params)

def form(request):
    msg = request.POST['msg']
    msg2 = paraphrasing(msg)
    params= {
        'msg1':'Your Input: '+msg,
        'msg2':'Synonyms: '+msg2,
        }
    return render(request, 'board/index.html', params)

def paraphrasing(msg):
    #model = Word2Vec.load("C:\\Users\\jun14\\Paraphrasing\\paraphrasing\\board\\word2vec.gensim.model")
    # Build model
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

    # Output top 10 of candidates
    topk_score, topk_index = torch.topk(predictions[0, masked_idx], 10)
    topk_tokens = tokenizer.convert_ids_to_tokens(topk_index.tolist())
    print(f'Input: {" ".join(tokenized_text)}')
    print(f'Top10: {topk_tokens}')
    # Input: [CLS] the cat perched on the mat [SEP] the cat [MASK] on the mat [SEP]
    # Top10: ['perched', 'sat', 'landed', 'was', 'rested', 'stood', 'settled', 'hovered', 'sitting', 'crouched']
    #model = gensim.downloader.load('glove-twitter-25')
    #sentenses = model.most_similar(msg)
    sentense = text
    return sentense