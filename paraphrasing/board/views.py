from django.shortcuts import render
from django.http import JsonResponse
import sys
import os
import torch

# パスを通す
current_dir=os.getcwd()
sys.path.append(f"{current_dir}/Japanese")

# =======================================================
#   Functions for English version
# =======================================================

from run_bert import BERT_LS

bert = BERT_LS()

def bert_ls(input_str) -> str:
    output_str = bert.simplify(input_str)
    return output_str

# =======================================================
#   Functions for Japanese version
# =======================================================

import MeCab
import kenlm
class Args:
    def __init__(self):
        self.candidate          = 'bert' # bert or glavas or glavas+synonym or synonymで動作
        self.simplicity         = 'point-wise'
        self.ranking            = 'bert' # bert or glavasで動作
        self.most_similar       = 10
        self.cos_threshold      = 0.0
        self.embedding          = "Japanese/embeddings/glove.txt"
        self.language_model     = 'Japanese/data/wiki.arpa.bin'
        self.word_to_complexity = 'Japanese/data/word2complexity.tsv'
        self.synonym_dict       = 'Japanese/data/ppdb-10best.tsv'
        self.word_to_freq       = 'Japanese/data/word2freq.tsv'
        self.simple_synonym     = 'Japanese/data/ss.pairwise.ours-B.tsv'
        self.pretrained_bert    = 'cl-tohoku/bert-base-japanese-whole-word-masking'
        self.device = -1

args = Args()

from Japanese.demo import *
import json

word2vec, w2v_vocab, mecab_wakati, mecab, language_model, word2level, word2synonym, word2freq, freq_total, simple_synonym, w2v_vocab, bert = load(args)
if args.device == -1:
    device = torch.device("cpu")
else:
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

def split_sentence(sentence:str):
	sentence_list = mecab_wakati.parse(sentence).rstrip('\n').split()
	select_list = [0] * len(sentence_list)

	for i, word in enumerate(sentence_list):
		if word in word2level:
			replaced_word = simplification_sentence(sentence, word)
			if replaced_word == word:
				select_list[i] = 0
			else:
				select_list[i] = 1
		else:
			select_list[i] = 0
	return json.dumps([sentence_list, select_list], ensure_ascii=False)


# sentence 中の target を簡単な単語に書き換えて、その単語を返す
def simplification_sentence(sentence:str, target:str):
    sentence = morphological_analysis(sentence, mecab)
    candidates, scores = pick_candidates(target, args.most_similar, word2vec, w2v_vocab, word2synonym, args.candidate, args.cos_threshold, bert, sentence, device)
    candidates = pick_simple_before_ranking(target, candidates, word2freq, freq_total, word2level, simple_synonym, args.simplicity)
    candidatelist = ranking(target, candidates, sentence, word2vec, w2v_vocab, word2freq, freq_total, language_model, ('',''), mecab, word2synonym, args.ranking, scores)
    candidatelist = pick_simple(candidatelist, args.simplicity, target, word2level, word2freq, freq_total, simple_synonym)
    rst = ",".join([" ".join([c[1] for c in rank]) for rank in candidatelist])
    
    return rst.replace(',', ' ').split(' ')[0]

# =======================================================
#   Interface
# =======================================================
import re
p = re.compile('[a-zA-Z0-9 -~]+')

def index(request):
    """ページの表示"""
    params= {
        'input_str':'',
        'output_str':'',
        'detected_language':"English",
        'split_sentence':'',
        #'highlight_word':[],
    }
    return render(request, 'board/index.html', params)

def Ajax_form(request):
    """入力を受け取り，平易化処理後の文章を返す"""
    input_str=request.POST.get("input_str")
    output_str=""
    splited=""
    try:
        if p.fullmatch(input_str): # 入力が英語の場合
            detected_language = "English"
            #output_str = input_str # 開発用
            output_str = bert_ls(input_str)
        else: #入力が日本語の場合
            detected_language = "日本語"
            output_str = input_str
            splited = split_sentence(input_str)
            if "sentence" in request.POST:
                sentence=request.POST.get("sentence")
                idx=int(request.POST.get("idx"))
                sentence_list = mecab_wakati.parse(sentence).rstrip('\n').split()
                # このoutput_strは単語（他は文章）
                output_str = simplification_sentence(sentence, sentence_list[idx])
                
    except Exception as e:
        output_str=f"err: {e}"
    
    #highlight_words_list=["perched"] # list of words that can be paraphrased
    params= {
        'input_str':input_str,
        'output_str':output_str,
        'detected_language':detected_language,
        'splited': splited,
        #'highlight_words_list':highlight_words_list,
        }
    return JsonResponse(params)


    