# Python3のimegeを基にする
FROM python:3.7-buster
ENV PYTHONUNBUFFRED 1

#ビルド時
RUN mkdir /usr/src/paraphrasing

# ワークディレクトリの設定
WORKDIR /usr/src/paraphrasing

#requirements.txtをコピー
ADD requirements.txt /usr/src/paraphrasing

#requiremtnts.txtを基にpipインストールをする
RUN pip install -U pip &&\
  pip install --no-cache-dir -r requirements.txt

ADD . /usr/src/paraphrasing/