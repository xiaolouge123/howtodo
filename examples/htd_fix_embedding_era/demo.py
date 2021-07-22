import os
import json
import random
import argparse

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.backend import keras, K
from bert4keras.layers import Layer

from keras.initializers import Constant
from keras.layers import Input, Embedding, Dropout, Dense, Add, Average, Concatenate, Flatten, Lambda, GlobalMaxPooling1D, LSTM, Bidirectional
from keras.models import Model
from keras.utils import to_categorical
from keras.callbacks import Callback
from keras.optimizers import Adagrad
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


parser = argparse.ArgumentParser(description="Run text classifier")
parser.add_argument('--patience', default=5)
parser.add_argument('--learning_rate', default=0.01, type=float)
args = parser.parse_args()
print(args)


snli_data_dir = '/Users/davidzhang/Downloads/snli_1.0'
snli_train_path = os.path.join(snli_data_dir,'snli_1.0_train.jsonl')
snli_dev_path = os.path.join(snli_data_dir,'snli_1.0_dev.jsonl')
snli_test_path = os.path.join(snli_data_dir,'snli_1.0_test.jsonl')
snli_label_number = 3

glove_emb_path = '/Users/davidzhang/Downloads/glove/glove.6B.50d.txt'

EPOCHS = 10
MAX_VOCAB_SIZE = 200000
BATCH_SIZE = 128

SNLI_LABEL2ID = {'entailment': 0, 'neutral': 1, 'contradiction': 2}

def build_tokenizer(data_list):
    raw_texts = []
    for data in data_list:
        raw_texts.extend([d[0] for d in data] + [d[1] for d in data])
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token='[UNK]')
    tokenizer.fit_on_texts(raw_texts)
    print('Found %s unique tokens.' % len(tokenizer.word_index))
    print('[UNK] token id is 1.')
    return tokenizer # tokenizer index the tokens from corpus

def load_snli_data(path):
    data, labels = [], []
    with open(path, 'r') as fr:
        for line in fr.readlines():
            d = json.loads(line)
            data.append([d['sentence1'], d['sentence2']])
            labels.append(d['gold_label'])
    print("Load data number: {} from {}.".format(len(data), path))
    return data, labels
            
def load_pretrain_embeddings(path):
    token2emb = {}
    with open(path, 'r') as fr:
        for line in fr.readlines():
            values = line.strip().split()
            token = values[0]
            emb = np.asarray(values[1:], dtype='float32')
            token2emb[token] = emb
    print('Loading {} tokens.'.format(len(token2emb)))
    return token2emb

def get_embedding_matrix(token2emb, tokenizer):
    embedding_dim = len(token2emb['.'])
    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((len(word_index)+1, len(token2emb['.'])))
    random_100_unk_vector = []
    for i in range(100):
        np.random.seed(i)
        random_100_unk_vector.append(np.random.normal(0, 0.01, embedding_dim)) # every time it called, it is different.
    missing_cnt = 0
    for word, i in word_index.items():
        if word == '[UNK]':
            embedding_matrix[i] = random_100_unk_vector[0]
        v = token2emb.get(word)
        if v is not None:
            embedding_matrix[i] = v
        else:
            embedding_matrix[i] = random.sample(random_100_unk_vector,1)[0]
            missing_cnt += 1
    print("{} tokens are not in pretrained embeddings.".format(missing_cnt))
    return embedding_matrix


class data_generator(DataGenerator):
    def __iter__(self, random=False):
        batch_input_a_token_ids, batch_input_b_token_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            text_a = 'null ' + text[0]
            text_b = 'null ' + text[1]
            if label == '-':
                continue
            else: 
                label = SNLI_LABEL2ID[label]
            batch_input_a_token_ids.append(text_a)
            batch_input_b_token_ids.append(text_b)
            batch_labels.append(label)
            if len(batch_input_a_token_ids) == self.batch_size or is_end:
                batch_input_a_token_ids = sequence_padding(tokenizer.texts_to_sequences(batch_input_a_token_ids))
                batch_input_b_token_ids = sequence_padding(tokenizer.texts_to_sequences(batch_input_b_token_ids))
                batch_labels = to_categorical(batch_labels, num_classes=snli_label_number)
                yield [batch_input_a_token_ids, batch_input_b_token_ids], batch_labels
                batch_input_a_token_ids, batch_input_b_token_ids, batch_labels = [], [], []

class data_generator_one_seq(DataGenerator):
    def __iter__(self, random=False):
        batch_input_a_token_ids, batch_labels = [], []
        for is_end, (text, label) in self.sample(random):
            text_a = text[0] + text[1]
            if label == '-':
                continue
            else: 
                label = SNLI_LABEL2ID[label]
            batch_input_a_token_ids.append(text_a)
            batch_labels.append(label)
            if len(batch_input_a_token_ids) == self.batch_size or is_end:
                batch_input_a_token_ids = sequence_padding(tokenizer.texts_to_sequences(batch_input_a_token_ids))
                batch_labels = to_categorical(batch_labels, num_classes=snli_label_number)
                yield batch_input_a_token_ids, batch_labels
                batch_input_a_token_ids, batch_labels = [], []

class Evaluator(Callback):
    def __init__(self):
        super(Callback, self).__init__()
        self.best_valid_f1 = 0.
        self.best_valid_epoch = 0
        self.metric_history = {}
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        f1, pre, rec, acc = evaluate_categorical(dev_generator)
        self.metric_history[epoch] = {'f1':f1, 'precision':pre, 'recall':rec, 'accuracy':acc}
        if f1 > self.best_valid_f1:
            self.wait = 0
            self.best_valid_f1 = f1
            self.best_valid_epoch = epoch
            self.metric_history['best'] = {'best_valid_f1': self.best_valid_f1, 'best_valid_epoch':self.best_valid_epoch}
            self.model.save_weights('best_model.weights')
        print(
            u'valid_f1: %.5f, 'u'valid_precision: %.5f, 'u'valid_recall: %.5f, 'u'valid_accuracy: %.5f, best_valid_f1: %.5f\n' %
            (f1, pre, rec, acc, self.best_valid_f1)
        )
        self.wait += 1
        if self.wait > args.patience:
            self.model.stop_training = True

def build_model_vallia_lstm(embedding_matrix):
    '''
    valid_f1: 0.65957, valid_precision: 0.66335, valid_recall: 0.66054, valid_accuracy: 0.66084, best_valid_f1: 0.65957
    test_f1: 0.66175, test_precision: 0.66577, test_recall: 0.66301, test_accuracy: 0.66409
    '''
    input_a = Input(shape=(None,), dtype='int64')
    input_b = Input(shape=(None,), dtype='int64')
    token_number = embedding_matrix.shape[0]
    embedding_dim = embedding_matrix.shape[1]
    embedding_layer = Embedding(
        token_number,
        embedding_dim,
        embeddings_initializer=Constant(embedding_matrix),
        trainable=False,
    )
    emb_a = embedding_layer(input_a)
    emb_b = embedding_layer(input_b)
    x = Concatenate(axis=1)([emb_a,emb_b])
    x = LSTM(64, return_sequences=True, return_state=True)(x)
    print(len(x))
    print(x[0].shape)
    print(x[1].shape)
    print(x[2].shape)
    output = Dense(snli_label_number, activation='softmax')(x)
    model = Model([input_a, input_b], output)
    model.compile(
        loss='categorical_crossentropy', 
        optimizer=Adagrad(learning_rate=args.learning_rate), 
        metrics=['categorical_accuracy']
        )
    model.summary()
    return model

def build_model_bilstm(embedding_matrix):
    '''
    valid_f1: 0.68072, valid_precision: 0.68141, valid_recall: 0.68064, valid_accuracy: 0.68076, best_valid_f1: 0.68072
    test_f1: 0.68289, test_precision: 0.68355, test_recall: 0.68284, test_accuracy: 0.68333
    '''
    input_a = Input(shape=(None,), dtype='int64')
    input_b = Input(shape=(None,), dtype='int64')
    token_number = embedding_matrix.shape[0]
    embedding_dim = embedding_matrix.shape[1]
    embedding_layer = Embedding(
        token_number,
        embedding_dim,
        embeddings_initializer=Constant(embedding_matrix),
        trainable=False,
    )
    emb_a = embedding_layer(input_a)
    emb_b = embedding_layer(input_b)
    x = Concatenate(axis=1)([emb_a,emb_b])
    x = Bidirectional(LSTM(64))(x)
    output = Dense(snli_label_number, activation='softmax')(x)
    model = Model([input_a, input_b], output)
    model.compile(
        loss='categorical_crossentropy', 
        optimizer=Adagrad(learning_rate=args.learning_rate), 
        metrics=['categorical_accuracy']
        )
    model.summary()
    return model

def build_model_bilstm_v2(embedding_matrix):
    '''
    valid_f1: 0.70685, valid_precision: 0.70989, valid_recall: 0.70748, valid_accuracy: 0.70809, best_valid_f1: 0.70685
    test_f1: 0.71136, test_precision: 0.71447, test_recall: 0.71189, test_accuracy: 0.71315
    '''
    input_a = Input(shape=(None,), dtype='int64')
    input_b = Input(shape=(None,), dtype='int64')
    token_number = embedding_matrix.shape[0]
    embedding_dim = embedding_matrix.shape[1]
    embedding_layer = Embedding(
        token_number,
        embedding_dim,
        embeddings_initializer=Constant(embedding_matrix),
        trainable=False,
    )
    emb_a = embedding_layer(input_a)
    emb_b = embedding_layer(input_b)
    x = Concatenate(axis=1)([emb_a,emb_b])
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Bidirectional(LSTM(64))(x)
    output = Dense(snli_label_number, activation='softmax')(x)
    model = Model([input_a, input_b], output)
    model.compile(
        loss='categorical_crossentropy', 
        optimizer=Adagrad(learning_rate=args.learning_rate), 
        metrics=['categorical_accuracy']
        )
    model.summary()
    return model

def build_model_bilstm_v3(embedding_matrix):
    '''
    '''
    input_a = Input(shape=(None,), dtype='int64')
    token_number = embedding_matrix.shape[0]
    embedding_dim = embedding_matrix.shape[1]
    embedding_layer = Embedding(
        token_number,
        embedding_dim,
        embeddings_initializer=Constant(embedding_matrix),
        trainable=False,
    )
    emb_a = embedding_layer(input_a)
    x = Bidirectional(LSTM(64, return_sequences=True))(emb_a)
    x = Bidirectional(LSTM(64))(x)
    output = Dense(snli_label_number, activation='softmax')(x)
    model = Model(input_a, output)
    model.compile(
        loss='categorical_crossentropy', 
        optimizer=Adagrad(learning_rate=args.learning_rate), 
        metrics=['categorical_accuracy']
        )
    model.summary()
    return model

def build_model_decomp_att(embedding_matrix):
    '''
    https://arxiv.org/pdf/1606.01933v1.pdf
    '''
    input_a = Input(shape=(None,), dtype='int64')
    input_b = Input(shape=(None,), dtype='int64')
    # token_number = embedding_matrix.shape[0]
    # embedding_dim = embedding_matrix.shape[1]
    # embedding_layer = Embedding(
    #     token_number,
    #     embedding_dim,
    #     embeddings_initializer=Constant(embedding_matrix),
    #     trainable=False,
    # )
    # emb_a = embedding_layer(input_a)
    # emb_b = embedding_layer(input_b)

    # head_size = 128
    # F_layer = Dense(units=head_size, activation='relu')
    # x_a = F_layer(emb_a)
    # x_b = F_layer(emb_b)
    # x_a = K.reshape(x_a, (-1, K.shape(input_a)[1], head_size))
    # x_b = K.reshape(x_b, (-1, K.shape(input_b)[1], head_size))
    # e = tf.einsum("bjh,bkh->bjk", x_a, x_b)
    # beta_att = K.softmax(e, axis=-1)
    # alpha_att = K.softmax(e, axis=-2)
    # emb_a_ = K.reshape(emb_a, (-1, K.shape(input_a)[1], embedding_dim))
    # emb_b_ = K.reshape(emb_b, (-1, K.shape(input_b)[1], embedding_dim))
    # beta = tf.einsum("bjk,bkd->bjd", beta_att, emb_b_)
    # alpha = tf.einsum("bjk,bjd->bkd", alpha_att, emb_a_)
    # v1_ = Concatenate()([emb_a, beta])
    # v2_ = Concatenate()([emb_b, alpha])
    # G_layer = Dense(units=head_size, activation='relu')
    # v1 = K.sum(G_layer(v1_), axis=1)
    # v2 = K.sum(G_layer(v2_), axis=1)
    # x = Concatenate()([v1,v2])
    decomp_att = DecomposeAttentionLayer(embedding_matrix, 128, 3)
    vs = decomp_att([input_a, input_b])
    x = Concatenate()([vs[0],vs[1]])
    output = Dense(3, activation='softmax')(x)
    model = Model([input_a, input_b], output)
    model.compile(
        loss='categorical_crossentropy', 
        optimizer=Adagrad(learning_rate=0.01), 
        metrics=['categorical_accuracy']
        )
    model.summary()
    return model

class DecomposeAttentionLayer(Layer):
    def __init__(self, embedding_matrix, head_size, snli_label_number, **kwargs):
        super(DecomposeAttentionLayer, self).__init__(**kwargs)
        self.head_size = head_size
        self.snli_label_number = snli_label_number
        self.token_number = embedding_matrix.shape[0]
        self.embedding_dim = embedding_matrix.shape[1]
        self.embedding_layer = Embedding(
                                    self.token_number,
                                    self.embedding_dim,
                                    embeddings_initializer=Constant(embedding_matrix),
                                    trainable=False,
                                )
        self.F_layer = Dense(units=head_size, activation='relu')
        self.G_layer = Dense(units=head_size, activation='relu')
        self.H_layer = Dense(self.snli_label_number, activation='softmax')

    def call(self, inputs):
        input_a, input_b = inputs[:2]
        emb_a = self.embedding_layer(input_a)
        emb_b = self.embedding_layer(input_b)
        # x_a = self.F_layer(emb_a)
        # x_b = self.F_layer(emb_b)
        # x_a = K.reshape(x_a, (-1, K.shape(input_a)[1], self.head_size))
        # x_b = K.reshape(x_b, (-1, K.shape(input_b)[1], self.head_size))
        # e = tf.einsum("bjh,bkh->bjk", x_a, x_b)
        # beta_att = K.softmax(e, axis=-1)
        # alpha_att = K.softmax(e, axis=-2)
        # emb_a_ = K.reshape(emb_a, (-1, K.shape(input_a)[1], self.embedding_dim))
        # emb_b_ = K.reshape(emb_b, (-1, K.shape(input_b)[1], self.embedding_dim))
        # beta = tf.einsum("bjk,bkd->bjd", beta_att, emb_b_)
        # alpha = tf.einsum("bjk,bjd->bkd", alpha_att, emb_a_)
        # v1_ = Concatenate()([emb_a, beta])
        # v2_ = Concatenate()([emb_b, alpha])
        # v1 = K.sum(self.G_layer(v1_), axis=1)
        # v2 = K.sum(self.G_layer(v2_), axis=1)
        v1 = tf.reduce_sum(emb_a, axis=1)
        v2 = tf.reduce_sum(emb_b, axis=1)
        
        return [v1,v2]

def build_model_lstmn(embedding_matrix):
    '''
    Long Short-Term Memory-Networks for Machine Reading
    https://www.aclweb.org/anthology/D16-1053.pdf
    '''
    pass


def evaluate_categorical(data):
    f1, pre, rec, acc = 0., 0., 0., 0.,
    y, y_ = [], []
        
    for x_true, y_true in tqdm(data):
        y_p = model.predict(x_true).argmax(axis=1)
        y_.extend([l for l in y_p])
        y.extend([l for l in y_true.argmax(axis=1)])

    f1 = f1_score(y, y_, average='macro')
    pre = precision_score(y, y_, average='macro')
    rec = recall_score(y, y_, average='macro')
    acc = accuracy_score(y, y_)
    return f1, pre, rec, acc

def predict_categorical(data):
    pass

def evaluate(data):
    pass

def predict(data):
    pass

if __name__=="__main__":
    # load data
    train_data, train_labels = load_snli_data(snli_train_path)
    dev_data, dev_labels = load_snli_data(snli_dev_path)
    test_data, test_labels = load_snli_data(snli_test_path)
    # load embeddings
    token2embs = load_pretrain_embeddings(glove_emb_path)
    # get tokenizer
    tokenizer = build_tokenizer([train_data, dev_data, test_data])
    # get embedding matrix 
    embedding_matrix = get_embedding_matrix(token2embs, tokenizer)
    # prepare data generator
    train = [(text, label) for text, label in zip(train_data, train_labels)]
    dev = [(text, label) for text, label in zip(dev_data, dev_labels)]
    test = [(text, label) for text, label in zip(test_data, test_labels)]
    train_generator = data_generator(train, BATCH_SIZE)
    dev_generator = data_generator(dev, BATCH_SIZE)
    test_generator = data_generator(test, BATCH_SIZE)
    # train_generator = data_generator_one_seq(train, BATCH_SIZE)
    # dev_generator = data_generator_one_seq(dev, BATCH_SIZE)
    # test_generator = data_generator_one_seq(test, BATCH_SIZE)

    model = build_model_vallia_lstm(embedding_matrix)
    evaluator = Evaluator()
    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=EPOCHS,
        callbacks=[evaluator]
    )

    f1, pre, rec, acc = evaluate_categorical(test_generator)
    print(
        u'test_f1: %.5f, 'u'test_precision: %.5f, 'u'test_recall: %.5f, 'u'test_accuracy: %.5f\n' %
        (f1, pre, rec, acc)
    )