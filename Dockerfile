# Python3のimegeを基にする
FROM python:3.7-buster
ENV PYTHONUNBUFFRED 1

#ビルド時
#RUN mkdir /usr/src/

# ワークディレクトリの設定
WORKDIR /usr/src/

# Copy the current directory.
COPY ./ /usr/src/

# Install libraries
RUN pip install -U pip &&\
  pip install --no-cache-dir -r requirements.txt
#  pip install -r requirements.txt
ADD https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip /usr/src/paraphrasing/BERT_resources/
ADD http://nlpgrid.seas.upenn.edu/PPDB/eng/ppdb-2.0-tldr.gz /usr/src/paraphrasing/BERT_resources/
ADD https://github.com/siangooding/lexical_simplification/raw/master/gpu_attention.model /usr/src/paraphrasing/BERT_resources/
RUN python -m spacy download en_core_web_sm
RUN python punkt_download.py

# Open port
#EXPOSE 80

# Run manage.py
#CMD python3 /usr/src/paraphrasing/paraphrasing/manage.py runserver 0.0.0.0:80