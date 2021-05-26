from django.shortcuts import render
from django.http import HttpResponse
import gensim.downloader
from gensim.models import Word2Vec

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
    sentenses = model.most_similar(msg)
    sentense = sentenses[0][0]
    return sentense