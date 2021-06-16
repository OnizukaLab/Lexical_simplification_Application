from django.shortcuts import render
from django.http import HttpResponse
import gensim.downloader
from gensim.models import Word2Vec

# Create your views here.
def index(request):
    params= {
        'msg1':'',
        'result':'',
        'highlight_word':[],
    }
    return render(request, 'board/index.html', params)

def form(request):
    msg = request.POST['msg']
    #result = paraphrasing(msg)
    highlight_words_list=["University"] # list of words that can be paraphrased
    params= {
        'msg1':'Your Input: '+msg,
        #'result':'Synonyms: '+result,
        'highlight_words_list':highlight_words_list,
        }
    return render(request, 'board/index.html', params)

def paraphrasing(msg):
    #model = Word2Vec.load("C:\\Users\\jun14\\Paraphrasing\\paraphrasing\\board\\word2vec.gensim.model")
    model = gensim.downloader.load('glove-twitter-25')
    sentenses = model.most_similar(msg)
    sentense = sentenses[0][0]
    return sentense