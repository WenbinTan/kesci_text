import os
import re
import csv
import codecs
import pandas as pd
import numpy as np
import re
from gensim.models import word2vec
import gensim
import logging

path = '../data/'
train = pd.read_csv(path+'train.csv',lineterminator='\n')
test  = pd.read_csv(path+'test.csv',lineterminator='\n')
train['label'] = train['label'].map({'Negative':0,'Positive':1})

full_data = pd.concat([train, test])

full_data['review'].to_csv('dl_text.txt',index=False,header=False)

cut_txt = 'dl_text.txt'  # 须注意文件必须先另存为utf-8编码格式
save_model_name = 'Word300.model'
###w2v的特征维度
max_features = 30000
maxlen = 200
validation_split = 0.1

MAX_SEQUENCE_LENGTH = 150
MAX_NB_WORDS = 30000
EMBEDDING_DIM = 300
embed_size = 300


df_train = train
train_row = train.shape[0]
df_test = test


##word2vec模型训练
def model_train(train_file_name, save_model_file, n_dim):
    # 模型训练，生成词向量
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.Text8Corpus(train_file_name)  # 加载语料
    model = gensim.models.Word2Vec(sentences, size=n_dim)
    model.save(save_model_file)


##对词向量的转换
def gen_vec(text):
    vec = np.zeros(embed_size).reshape((1, embed_size))
    count = 0
    for word in text:
        try:
            vec += w2v_model[word].reshape((1, embed_size))
            count += 1
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


if not os.path.exists(save_model_name):
    model_train(cut_txt, save_model_name, EMBEDDING_DIM)
else:
    print('此训练模型已经存在，不用再次训练')
## 分词
w2v_model = word2vec.Word2Vec.load(save_model_name)
