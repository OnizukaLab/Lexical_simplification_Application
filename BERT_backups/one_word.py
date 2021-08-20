import argparse
import csv
import os
import random
import math
import sys
import re

from BERT_resources.tokenization import BertTokenizer
from BERT_resources.modeling import BertModel, BertForMaskedLM

from sklearn.metrics.pairwise import cosine_similarity as cosine

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import numpy as np
import torch
import nltk
#from PPDB import Ppdb
from scipy.special import softmax
#from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

max_seq_length = 250
num_selections = 20
ps = PorterStemmer()
window_context = 11

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids

def convert_sentence_to_token(sentence, seq_length, tokenizer):

    tokenized_text = tokenizer.tokenize(sentence.lower())

    assert len(tokenized_text) < seq_length-2

    nltk_sent = nltk.word_tokenize(sentence.lower())

    position2 = []

    token_index = 0

    start_pos =  len(tokenized_text)  + 2

    pre_word = ""

    for i,word in enumerate(nltk_sent):

        if word=="n't" and pre_word[-1]=="n":
            word = "'t"

        if tokenized_text[token_index]=="\"":
            len_token = 2
        else:
            len_token = len(tokenized_text[token_index])

        if tokenized_text[token_index]==word or len_token>=len(word):
            position2.append(start_pos+token_index)
            pre_word = tokenized_text[token_index]

            token_index += 1
        else:
            new_pos = []
            new_pos.append(start_pos+token_index)

            new_word = tokenized_text[token_index]

            while new_word != word:

                token_index += 1

                new_word += tokenized_text[token_index].replace('##','')

                new_pos.append(start_pos+token_index)

                if len(new_word)==len(word):
                    break
            token_index += 1
            pre_word = new_word
           
            position2.append(new_pos)
       
    return tokenized_text, nltk_sent, position2

def extract_context(words, mask_index, window):
    #extract 7 words around the content word

    length = len(words)

    half = int(window/2)

    assert mask_index>=0 and mask_index<length

    context = ""

    if length<=window:
        context = words
    elif mask_index<length-half and mask_index>=half:
        context = words[mask_index-half:mask_index+half+1]
    elif mask_index<half:
        context = words[0:window]
    elif mask_index>=length-half:
        context = words[length-window:length]
    else:
        print("Wrong!")

    return context

def convert_whole_word_to_feature(tokens_a, mask_position, seq_length, tokenizer):
    """Loads a data file into a list of `InputFeature`s."""

    #tokens_a = tokenizer.tokenize(sentence)
    #print(mask_position)

    tokens = []
    input_type_ids = []
    tokens.append("[CLS]")
    input_type_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        input_type_ids.append(0)
    
    tokens.append("[SEP]")
    input_type_ids.append(0)

    for token in tokens_a:
        tokens.append(token)
        input_type_ids.append(1)

    tokens.append("[SEP]")
    input_type_ids.append(1)


    true_word = ''
    index = 0
    count = 0
    mask_position_length = len(mask_position)

    while count in range(mask_position_length):
        index = mask_position_length - 1 - count

        pos = mask_position[index]
        if index == 0:
            tokens[pos] = '[MASK]'
        else:
            del tokens[pos]
            del input_type_ids[pos]

        count += 1

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
    input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
    while len(input_ids) < seq_length:
        input_ids.append(0)
        input_mask.append(0)
        input_type_ids.append(0)

    assert len(input_ids) == seq_length
    assert len(input_mask) == seq_length
    assert len(input_type_ids) == seq_length

      
    return InputFeatures(unique_id=0,  tokens=tokens, input_ids=input_ids,input_mask=input_mask,input_type_ids=input_type_ids)

def convert_token_to_feature(tokens_a, mask_position, seq_length, tokenizer):
    """Loads a data file into a list of `InputFeature`s."""

    #tokens_a = tokenizer.tokenize(sentence)
    #print(mask_position)

    tokens = []
    input_type_ids = []
    tokens.append("[CLS]")
    input_type_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        input_type_ids.append(0)
    
    tokens.append("[SEP]")
    input_type_ids.append(0)

    for token in tokens_a:
        tokens.append(token)
        input_type_ids.append(1)

    tokens.append("[SEP]")
    input_type_ids.append(1)


    true_word = ''
    if isinstance(mask_position,list):
        for pos in  mask_position:
            true_word = true_word + tokens[pos]
            tokens[pos] = '[MASK]'
    else:
        true_word = tokens[mask_position]
        tokens[mask_position] =  '[MASK]'


    input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
    input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
    while len(input_ids) < seq_length:
        input_ids.append(0)
        input_mask.append(0)
        input_type_ids.append(0)

    assert len(input_ids) == seq_length
    assert len(input_mask) == seq_length
    assert len(input_type_ids) == seq_length

      
    return InputFeatures(unique_id=0,  tokens=tokens, input_ids=input_ids,input_mask=input_mask,input_type_ids=input_type_ids)

def substitution_ranking(source_word, source_context, substitution_selection, fasttext_dico, fasttext_emb, word_count, ssPPDB, tokenizer, maskedLM, lables):
    ss,sis_scores,count_scores=preprocess_SR(source_word, substitution_selection, fasttext_dico, fasttext_emb, word_count)
    #print(ss)
    if len(ss)==0:
        return source_word

    if len(sis_scores)>0:
        seq = sorted(sis_scores,reverse = True )
        sis_rank = [seq.index(v)+1 for v in sis_scores]

    rank_count = sorted(count_scores,reverse = True )
    count_rank = [rank_count.index(v)+1 for v in count_scores]
    lm_score,source_lm = LM_score(source_word,source_context,ss,tokenizer,maskedLM)
    rank_lm = sorted(lm_score)
    lm_rank = [rank_lm.index(v)+1 for v in lm_score]

    bert_rank = []
    ppdb_rank =[]
    for i in range(len(ss)):
        bert_rank.append(i+1)
        if ss[i] in ssPPDB:
            ppdb_rank.append(1)
        else:
            ppdb_rank.append(len(ss)/3)

    if len(sis_scores)>0:
        all_ranks = [bert+sis+count+LM+ppdb  for bert,sis,count,LM,ppdb in zip(bert_rank,sis_rank,count_rank,lm_rank,ppdb_rank)]
    else:
        all_ranks = [bert+count+LM+ppdb  for bert,count,LM,ppdb in zip(bert_rank,count_rank,lm_rank,ppdb_rank)]

    pre_index = all_ranks.index(min(all_ranks))
    pre_count = count_scores[pre_index]

    if source_word in word_count:
        source_count = word_count[source_word]
    else:
        source_count = 0

    pre_lm = lm_score[pre_index]

    if source_lm>pre_lm or pre_count>source_count:
        pre_word = ss[pre_index]
    else:
        pre_word = source_word

    return pre_word

def substitution_generation(source_word, pre_tokens, ps, num_selection=10):

    cur_tokens=[]

    source_stem = ps.stem(source_word)

    assert num_selection<=len(pre_tokens)

    for i in range(len(pre_tokens)):
        token = pre_tokens[i]
     
        if token[0:2]=="##":
            continue

        if(token==source_word):
            continue

        token_stem = ps.stem(token)

        if(token_stem == source_stem):
            continue

        if (len(token_stem)>=3) and (token_stem[:3]==source_stem[:3]):
            continue

        cur_tokens.append(token)
        

        if(len(cur_tokens)==num_selection):
            break
    
    if(len(cur_tokens)==0):
        cur_tokens = pre_tokens[0:num_selection+1]

    assert len(cur_tokens)>0       

    return cur_tokens

def produce_alternatives(sentence, word_to_replace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    model.to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    tokens, words, position = convert_sentence_to_token(sentence, max_seq_length, tokenizer)

    assert len(words)==len(position)

    #mask_index = words.index(replacement_index)
    mask_index = words.index(word_to_replace)

    mask_context = extract_context(words,mask_index,window_context)

    len_tokens = len(tokens)

    mask_position = position[mask_index]

    if isinstance(mask_position,list):
        feature = convert_whole_word_to_feature(tokens, mask_position, max_seq_length, tokenizer)
    else:
        feature = convert_token_to_feature(tokens, mask_position, max_seq_length, tokenizer)

    tokens_tensor = torch.tensor([feature.input_ids])
    token_type_ids = torch.tensor([feature.input_type_ids])
    attention_mask = torch.tensor([feature.input_mask])
    tokens_tensor = tokens_tensor.to('cuda')
    token_type_ids = token_type_ids.to('cuda')
    attention_mask = attention_mask.to('cuda')

    with torch.no_grad():
                prediction_scores = model(tokens_tensor, token_type_ids,attention_mask)
            
    if isinstance(mask_position,list):
        predicted_top = prediction_scores[0, mask_position[0]].topk(num_selections*2)
    else:
        predicted_top = prediction_scores[0, mask_position].topk(num_selections*2)
        #print(predicted_top[0].cpu().numpy())
    pre_tokens = tokenizer.convert_ids_to_tokens(predicted_top[1].cpu().numpy())

    pre_tokens = tokenizer.convert_ids_to_tokens(predicted_top[1].cpu().numpy())

    candidate_words = substitution_generation(word_to_replace, pre_tokens, ps, num_selections)
    return candidate_words

if __name__ == "__main__":
    sentence = input("Enter a sentence: ")
    word = input("Enter a word to simplify: ")
    replacements = produce_alternatives(sentence, word)
    print("Possible replacements:")
    for r in replacements:
        print(r)
