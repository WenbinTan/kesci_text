import numpy as np
import pandas as pd
import time
import gc
from collections import defaultdict
from sklearn.metrics import mean_squared_error, f1_score
from gensim.models import word2vec
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Activation, Dropout, Embedding, BatchNormalization, Bidirectional, Conv1D, \
    GlobalMaxPooling1D, GlobalAveragePooling1D, Input, Lambda, TimeDistributed, Convolution1D
from keras.layers import LSTM, concatenate, Conv2D,SpatialDropout1D,Bidirectional,CuDNNGRU
from keras.layers import AveragePooling1D, Flatten, GlobalMaxPool1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from Model_dir.attention_model import get_attention_cv_model

t1 = time.time()
###################################################################################################
model = word2vec.Word2Vec.load("Word300.model")

path = '../data/'
train = pd.read_csv(path+'train.csv',lineterminator='\n')
test = pd.read_csv(path+'test.csv',lineterminator='\n')
train['label'] = train['label'].map({'Negative':0,'Positive':1})

MAX_SEQUENCE_LENGTH = 220
MAX_NB_WORDS = 22000
EMBEDDING_DIM = 300

column = "review"
tokenizer = Tokenizer(
    nb_words = MAX_NB_WORDS,
    filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    lower = True,
    split = ' ')
tokenizer.fit_on_texts(list(train[column]) + list(test[column]))


sequences_all = tokenizer.texts_to_sequences(list(train[column]))
sequences_test = tokenizer.texts_to_sequences(list(test[column]))
X_train = pad_sequences(sequences_all, maxlen=MAX_SEQUENCE_LENGTH)
X_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)

word_index = tokenizer.word_index
nb_words = min(MAX_NB_WORDS, len(word_index))
print(nb_words)
ss = 0
embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
print(len(word_index.items()))
for word, i in word_index.items():
    if word in model.wv.vocab:
        ss += 1
        if i >= nb_words:
            break
        embedding_matrix[i] = model.wv[word]
    else:
        # print word
        pass
print(ss)
print(embedding_matrix.shape)
# np.save("embedding_matrix.npy",embedding_matrix)

y_true = train["label"].astype(int)
y = np_utils.to_categorical(y_true)


###################################################################################################################################
from sklearn.model_selection import KFold

folds = 4
seed = 2018
skf = KFold(n_splits=folds, shuffle=True, random_state=seed)

te_pred = np.zeros((X_train.shape[0], 2))
test_pred = np.zeros((X_test.shape[0], 2))
test_pred_cv = np.zeros((5, X_test.shape[0], 2))
cnt = 0
score = 0
score_cv_list = []
for ii, (idx_train, idx_val) in enumerate(skf.split(X_train)):
    X_train_tr = X_train[idx_train]
    X_train_te = X_train[idx_val]
    y_tr = y[idx_train]
    y_te = y[idx_val]

    model = get_attention_cv_model(MAX_SEQUENCE_LENGTH, nb_words, EMBEDDING_DIM, embedding_matrix)
    early_stop = EarlyStopping(patience=5)
    check_point = ModelCheckpoint('cate_model.hdf5', monitor="val_acc", mode="max", save_best_only=True, verbose=1)

    history = model.fit(X_train_tr, y_tr, batch_size=128, epochs=15, verbose=1, validation_data=(X_train_te, y_te),
                        callbacks=[early_stop, check_point]
    )

    model.load_weights('cate_model.hdf5')
    preds_te = model.predict(X_train_te)
    score_cv = f1_score(y_true[idx_val], np.argmax(preds_te, axis=1), labels=range(0, 2), average='macro')
    score_cv_list.append(score_cv)
    print(score_cv_list)
    te_pred[idx_val] = preds_te
    preds = model.predict(X_test)
    test_pred_cv[ii, :] = preds
    # break
    del model
    gc.collect()

test_pred[:] = test_pred_cv.mean(axis=0)
preds = [i[1] for i in test_pred]
res = pd.DataFrame()
res['ID'] = test['ID']
res['Pred'] = preds
res.to_csv('../summit/attention_base.csv', index = False, quoting = 3)

