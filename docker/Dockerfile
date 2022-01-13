# Python3のimegeを基にする
FROM python:3.7-buster
#FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
ENV PYTHONUNBUFFRED 1

#ビルド時
#RUN mkdir /usr/src/

# ワークディレクトリの設定
WORKDIR /usr/src/

# Copy the current directory.
COPY requirements.txt /usr/src/
COPY punkt_download.py /usr/src/

# Install libraries
#RUN apt-get update && apt-get -y upgrade && \
#     apt-get install wget -y && \
#     wget -P /opt https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh && \
#     bash /opt/Anaconda3-2021.11-Linux-x86_64.sh -b -p /opt/anaconda3 && \
#     rm /opt/Anaconda3-2021.11-Linux-x86_64.sh && \
#     echo "export PATH=/opt/anaconda3/bin:$PATH" >> ~/.bashrc && \
#     . ~/.bashrc && \
#     conda init
#RUN  curl -kL https://bootstrap.pypa.io/get-pip.py | python
RUN  pip install -U pip &&\
     pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm
RUN python punkt_download.py
#ADD https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip /usr/src/paraphrasing/BERT_resources/
#ADD http://nlpgrid.seas.upenn.edu/PPDB/eng/ppdb-2.0-tldr.gz /usr/src/paraphrasing/BERT_resources/
#ADD https://github.com/siangooding/lexical_simplification/raw/master/gpu_attention.model /usr/src/paraphrasing/BERT_resources/

# Open port
#EXPOSE 80

# Run manage.py
#CMD python3 /usr/src/paraphrasing/paraphrasing/manage.py runserver 0.0.0.0:80