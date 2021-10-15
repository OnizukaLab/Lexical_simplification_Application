# Paraphrasing2021
2021年度M1の語彙言い換えチーム．

# Clone

```
git clone 
```

# Dockerを使用する場合

```
cd Paraphrasing2021
docker build -t <image name> .
docker run -it -p <portA>:<portB> -v $PWD:/usr/src/paraphrasing <image name> sh
python paraphrasing/manage.py runserver <portB>
```
を実行し，

`http://127.0.0.1:<portA>/`

をクリックして、safariなどで開く。

例えば，

```
git clone 
cd Paraphrasing2021
docker build -t paraphrasing .
docker run -it -p 8000:80 -v $PWD:/usr/src/paraphrasing paraphrasing sh
python3 paraphrasing/paraphrasing/manage.py runserver 0.0.0.0:80
```

http://127.0.0.1:8000/

をクリックして、safariなどで開く。

***

# Dockerを使用しない場合

## Requirement
 
動かすのに必要なライブラリ
 
python 3
django 3.2.3 

python -m spacy download en_core_web_sm

NLTK - punkt:
```
import nltk
nltk.download('punkt')
```

BERT-LSを使うために、このファイルをダウンロードし、BERT_Resourcesに置く

crawl-300d-2M-subword.vec: 
https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip

ppdb-2.0-tldr:
http://paraphrase.org/#/download

gpu_attention_model: 
https://github.com/siangooding/lexical_simplification/blob/master/gpu_attention.model

## Installation
 
Requirementで列挙したライブラリなどのインストール方法
 
```
pip install -r requirements.txt
```
 
## Usage
 
実行方法など、基本的な使い方
 
```
git clone 
cd Paraphrasing2021/paraphrasing
python manage.py runserver
```

http://127.0.0.1:8000/

をクリックして、safariなどで開いた後、urlを

http://127.0.0.1:8000/board　

へ変更し、実行する。

# Citations

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
