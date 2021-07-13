import gensim.downloader
from gensim.models import Word2Vec,fasttext
from collections import Counter
from tqdm import tqdm
import numpy as np
import torch
from transformers import BertTokenizer, BertForMaskedLM
import matplotlib.pyplot as plt

# Create your models here.
def wordFrequency(word_frequency, tokenizer):
    filepath = './board/cbt_all.txt'
    n_lines = count_lines(filepath)
    with open(filepath, 'r', encoding="utf-8") as f:
        for line in tqdm(f, total=n_lines):
            if line.startswith("_BOOK_TITLE_"):
                continue
            else:
                tokens = tokenizer.tokenize(line.rstrip())
                for token in tokens:
                    word_frequency[token] += 1
    return word_frequency
        

#@title Build a table of word frequency
def count_lines(path):
  with open(path, 'r', encoding="utf-8") as f:
    return sum([1 for _ in f])


def model(tokenizer,text,masked_idx,word_frequency):
    similalityModel = fasttext.FastText.load_model('cc.en.300.bin')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
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
    # substitution ranking
    bert_rank = np.array([i for i in range(len(topk_tokens))])
    frequency_rank = np.argsort([-word_frequency[token] for token in topk_tokens])
    avg_rank = np.argsort((bert_rank + frequency_rank) / 2)

    # sort candidates and except a complex word
    candidates = [topk_tokens[i] for i in avg_rank if topk_tokens[i] != complex_word]

    # substitute a complex word
    tokenized_text[masked_idx] = candidates[0]
    print(" ".join(tokenized_text))

if __name__=='__main__':
    word_frequency = Counter()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #word_frequency = wordFrequency(word_frequency=word_frequency,tokenizer=tokenizer)
    # Model's input
    text = "[CLS] the cat perched on the mat [SEP] the cat perched on the mat [SEP]"
    masked_idx = 3

    model(tokenizer=tokenizer,text=text,masked_idx=masked_idx,word_frequency=word_frequency)


