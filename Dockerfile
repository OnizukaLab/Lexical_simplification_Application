# Python3のimegeを基にする
FROM python:3
ENV PYTHONUNBUFFRED 1

#ビルド時
RUN mkdir /paraphrasing

# ワークディレクトリの設定
WORKDIR /paraphrasing

#requirements.txtをコピー
ADD requirements.txt /paraphrasing

#requiremtnts.txtを基にpipインストールをする
RUN pip install -r requirements.txt
ADD . /paraphrasing/