# Paraphrasing2021
2021年度M1の語彙言い換えチーム．

# Requirement
 
動かすのに必要なライブラリ
 
python 3
django 3.2.3 
 
# Installation
 
Requirementで列挙したライブラリなどのインストール方法
 
```
pip install django
pip install numpy
pip install torch
pip install openpyxl
pip install scipy
pip install nltk
pip install tqdm
pip install pathlib
pip install sklearn
pip install spacy
python -m spacy download en_core_web_sm
pip install tensorflow
pip install inflect
pip install truecase
```

# Additional Files

BERT-LSを使うために、このファイルをダウンロードし、BERT_Resourcesに置く

```
crawl-300d-2M-subword.vec: https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip
ppdb-2.0-tldr from: http://paraphrase.org/#/download
gpu_attention_model from: https://github.com/siangooding/lexical_simplification/blob/master/gpu_attention.model
```
 
# Usage
 
実行方法など、基本的な使い方
 
```
git clone 
cd paraphrasing
python manage.py runserver
```

http://127.0.0.1:8000/

をクリックして、safariなどで開いた後、urlを

http://127.0.0.1:8000/board　

へ変更し、実行する。

#Citations

[BERT-LS](https://arxiv.org/pdf/1907.06226.pdf)

```
@article{qiang2020BERTLS,
  title =  {Lexical Simplification with Pretrained Encoders },
  author = {Qiang, Jipeng and 
            Li, Yun and
            Yi, Zhu and
            Yuan, Yunhao and 
            Wu, Xindong},
  journal={Thirty-Fourth AAAI Conference on Artificial Intelligence},
  pages={8649–8656},
  year  =  {2020}
}
```
