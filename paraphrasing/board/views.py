from django.shortcuts import render
from django.http import JsonResponse
#from run_bert import BERT_LS

#bert = BERT_LS()


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
        #output_str = bert_ls(input_str)
        #sample: The cat perched on the mat.
        output_str = input_str.replace("perched", "sat")
    except Exception as e:
        output_str=f"err: {e}"
    
    highlight_words_list=["perched"] # list of words that can be paraphrased
    params= {
        'input_str':input_str,
        'output_str':output_str,
        'highlight_words_list':highlight_words_list,
        }
    return render(request, 'board/index.html', params)

def Ajax_form(request):
    """Ajax処理"""
    """
    TODO: 
    JsonResponse(params)で動作させる（ページの再読み込みをしない）．
    一文字入力すると選択がtextareaから外れてしまう問題を解消．
    """
    #input_str = request.GET.get('input_str')
    input_str=request.POST["input_str"]
    try:
        #output_str = bert_ls(input_str)
        #sample: The cat perched on the mat.
        output_str = input_str.replace("perched", "sat")
    except Exception as e:
        output_str=f"err: {e}"
    
    highlight_words_list=["perched"] # list of words that can be paraphrased
    params= {
        'input_str':input_str,
        'output_str':output_str,
        'highlight_words_list':highlight_words_list,
        }
    return render(request, 'board/index.html', params)
    return JsonResponse(params)
    #return params



def bert_ls(input_str) -> str:
    output_str = bert.simplify(input_str)
    return output_str

    