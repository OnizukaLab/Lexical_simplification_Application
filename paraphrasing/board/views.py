from django.shortcuts import render
from django.http import HttpResponse
import gensim.downloader
from gensim.models import Word2Vec
from collections import Counter
from tqdm import tqdm
from . import paraphrasingModel
import numpy as np
from transformers import BertTokenizer, BertForMaskedLM
import torch
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
    model = gensim.downloader.load('glove-twitter-25')

    word_frequency = Counter()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    word_frequency = paraphrasingModel.wordFrequency(word_frequency=word_frequency,tokenizer=tokenizer)
    # Model's input
    text = "[CLS] the cat perched on the mat [SEP] the cat perched on the mat [SEP]"
    masked_idx = 10

    paraphrasingModel.model(tokenizer=tokenizer,text=text,masked_idx=masked_idx,word_frequency=word_frequency)

    

    sentense = model.most_similar(msg)
    return sentense