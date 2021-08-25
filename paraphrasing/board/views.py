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
        'msg1':'',
        'result':'',
        'highlight_word':[],
    }
    return render(request, 'board/index.html', params)

def form(request):
    if "change" in request.POST:
        msg = CurrentInput.objects.all()[0].contents
        #msg = "The cat perched on the mat."
        sample_word = "perched"
        sample_word2 = "sat"
        #sample_word2 = paraphrasing(sample_word)
        sample_word2 = bert_ls(sample_word)
        msg = msg.replace(sample_word, sample_word2)
    else:
        msg = request.POST['msg']
        new_input = CurrentInput(contents=msg)
        new_input.save()

    
    #result = paraphrasing(msg)
    highlight_words_list=["purched"] # list of words that can be paraphrased
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

def bert_ls(msg):
    output = bert.simplify(msg)
    return output

    