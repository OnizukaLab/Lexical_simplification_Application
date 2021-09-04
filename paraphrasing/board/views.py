from django.shortcuts import render
from django.http import HttpResponse
import gensim.downloader
from gensim.models import Word2Vec
from .models import CurrentInput
from run_bert import BERT_LS

msg_saved= ""
bert = BERT_LS()


# Create your views here.
def index(request):
    params= {
        'input_str':'',
        'output_str':'',
        'highlight_word':[],
    }
    return render(request, 'board/index.html', params)

def form(request):
    input_str=request.POST["input_str"]
    try:
        output_str = bert_ls(input_str)
    except Exception as e:
        output_str=f"err: {e}"
    
    highlight_words_list=["purched"] # list of words that can be paraphrased
    params= {
        'input_str':input_str,
        'output_str':output_str,
        'highlight_words_list':highlight_words_list,
        }
    return render(request, 'board/index.html', params)

def bert_ls(input_str) -> str:
    output_str = bert.simplify(input_str)
    return output_str

    