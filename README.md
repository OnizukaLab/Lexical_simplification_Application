# Paraphrasing2021
2021年度M1の語彙言い換えチーム．

# アプリケーションの概要

![Demo](/images/demo_gif.gif "demo")


英語の語彙を平易に言い換えるWebアプリケーションです。
モデルは文脈に沿った言い換えが可能な**BERT LS**を使用し、ブラウザのシンプルなUIで語彙言い換え操作を行うことができます。
入力は単語/文/文章で、出力はそれに対応する言い換え処理後の単語/文/文章です。

(このアプリケーションはメンテナンスされていないため、脆弱性が存在する可能性があります。
使用する際はtensorflowのバージョンなどをアップデートさせてから使用してください)

# 準備

## Requirements

```
git clone 
```

でクローンした後、
BERT-LSを使うために、以下の3つのファイルをダウンロードし、`paraphrasing/BERT_resources`に置く。

* [crawl-300d-2M-subword.vec](https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip) (解凍時に存在する.binは不要)

* [ppdb-2.0-tldr](http://paraphrase.org/#/download)

* [gpu_attention_model](https://github.com/siangooding/lexical_simplification/blob/master/gpu_attention.model)

BERT-LSがpytorchを使うので、CUDAバージョンと合わせるpytorchをインストールするのも必要。

日本語版を使うために、下の指示に従う。

* [pretrained_embedding](https://github.com/yagays/pretrained_embedding)でJapanese Wikipedia Entity Vectorsを作る

* 作ったentity_vector.model.txtの名をglove.txtにする

* glove.txtを`paraphrasing/Japanese/embeddings`に置く。

* wiki.arpa.binというファイル作るため、[このレポ](https://github.com/yagays/pretrained_embedding)でwiki.arpa.binを探索し、指示に従う。git s

* wiki.arpa.binを`paraphrasing/Japanese/data`に置く。

## ディレクトリ構成

Paraphrasing2021/

┣ paraphrasing/

┃ ┣ BERT_resources/

┃ ┃ ┣ **crawl-300d-2M-subword.vec**

┃ ┃ ┣ **ppdb-2.0-tldr**

┃ ┃ ┣ **gpu_attention.model**

┃ ┃ ┗ other files

┃ ┣ Japanese/

┃ ┃ ┣ data

┃ ┃ ┃ ┣ **wiki.arpa.bin**

┃ ┃ ┃ ┗ other files

┃ ┃ ┣ embeddings

┃ ┃ ┃ ┣ **glove.txt**

┃ ┃ ┗ other files

┃ ┣ paraphrasing/

┃ ┣ board/

┃ ┣ manage.py

┃ ┗ run_bert.py

┣ README.md

┣ Dockerfile

┣ requirements.txt

┗ punkt_download.py


# Dockerを使用する場合


## Dockerのビルドとシステムの起動

`Paraphrasing2021`のディレクトリに入り、

```
docker build -t <イメージ名> .
```

```
docker run -it -d -p <ポートA>:<ポートB> -v $PWD/paraphrasing:/usr/src/ --gpus <GPUの設定> --name <コンテナ名> <イメージ名> sh
```

を実行し，

```
docker exec -it <コンテナ名> sh
```

などでコンテナに入り、

```
python paraphrasing/manage.py runserver 0.0.0.0:<ポートB>
```

を実行し、

http://<IPアドレス>:<ポートA>/

をsafariやchromeなどのブラウザで開く。
（ローカルで動作させているなら、IPアドレスは127.0.0.1）


## 例

IPアドレスが10.0.16.12のサーバで動作させる場合、
`Paraphrasing2021`のディレクトリに入り、

```
docker build -t paraphrasing_img .
```

```
docker run -it -d -p 8101:8101 -v $PWD/paraphrasing:/usr/src/ --gpus "device=0" --name paraphrasing paraphrasing_img sh
```

を実行し、

```
docker exec -it paraphrasing sh
```

などでコンテナに入り、

```
python3 manage.py runserver 0.0.0.0:8101
```

を実行し、

http://10.0.16.12:8101

をブラウザで開く。


***

# Dockerを使用しない場合

## Requirement
 
動かすのに必要なライブラリ
 
python 3
django 3.2.3 

```
python -m spacy download en_core_web_sm
```

NLTK - punkt:
```
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
```

## Installation
 
Requirementで列挙したライブラリなどのインストール方法
 
```
pip install -r requirements.txt
```

```
pip install https://github.com/kpu/kenlm/archive/master.zip
```
 
## システムの起動
 
実行方法など、基本的な使い方：
 
```
cd Paraphrasing2021/paraphrasing
```

```
python manage.py runserver
```

を実行し、

http://127.0.0.1:8000/

をsafariなどのブラウザで開く。

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
