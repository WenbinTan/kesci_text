from keras.layers import Dense, Activation, Dropout, Embedding, BatchNormalization, Bidirectional, Conv1D, \
    GlobalMaxPooling1D, GlobalAveragePooling1D, Input, Lambda, TimeDistributed, Convolution1D
from keras.layers import LSTM, concatenate, Conv2D,SpatialDropout1D,Bidirectional,CuDNNGRU
from keras.layers import AveragePooling1D, Flatten, GlobalMaxPool1D
from keras.models import Sequential, load_model, Model
from keras.optimizers import Adam

# 建立模型
from model_utils import AttentivePoolingLayer
def get_attention_cv_model(MAX_SEQUENCE_LENGTH, nb_words, EMBEDDING_DIM, embedding_matrix):
    inp = Input(shape=(MAX_SEQUENCE_LENGTH,))
    emb = Embedding(nb_words, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH, weights=[embedding_matrix],
                    trainable=True,  dropout=0.1)(inp)
    att=AttentivePoolingLayer()(emb)

    fc1 = Dense(256, activation='relu')(att)
    fc2 = Dense(64, activation='relu')(fc1)
    fc2 = BatchNormalization()(fc2)
    output = Dense(2, activation="softmax")(fc2)
    model = Model(inputs=inp, outputs=output)
    adam = Adam(lr=1e-3)
    model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=['accuracy'])
    model.summary()
    return model