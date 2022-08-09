import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
plt.style.use("ggplot")

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from helper.data_transformer import *


def input_converter(words, tags, input_dt):

    data_pair = to_token_tag_list(input_dt)

    X, y = input_data(words, tags, data_pair)

    return X, y

def input_data(words, tags, dt_pair):
    # input data using word embeddings only
    max_len = max([len(i) for i in words])

    word_idx = word2idx(words, n=1)
    idx_word = idx2word(word_idx)

    tag_idx = tag2idx(tags, n=1)
    tag_idx["PAD"] = 0
    idx_tag = idx2tag(tag_idx)

    X = [[word_idx[w[0]] for w in s] for s in dt_pair]
    X = pad_sequences(maxlen=max_len, sequences=X, padding='post')

    y = [[tag_idx[t[1]] for t in s] for s in dt_pair]
    y = pad_sequences(maxlen=max_len, sequences=y, padding='post', value=tag_idx["PAD"])

    return X, y, idx_word, idx_tag
'''
def input_data(words, tags, dt_pair):
    # input data using word embeddings only
    max_len = max([len(i) for i in words])

    word_idx = word2idx(words, n=1)
    tag_idx = tag2idx(tags, n=0)

    X = [[word_idx[w[0]] for w in s] for s in dt_pair]
    X = pad_sequences(maxlen=max_len, sequences=X, padding='post')

    y = [[tag_idx[t[1]] for t in s] for s in dt_pair]
    y = pad_sequences(maxlen=max_len, sequences=y, padding='post', value=tag_idx["O"])

    return X, y
'''
def wc_input_converter(merged_data, input_data):
    words = merged_data[1]
    tags = merged_data[2]

    data_pair = to_token_tag_list(input_data)

    X_word, X_char, y_, idx_word, idx_tag = input_data_wc_embd(words, tags, data_pair)

    return X_word, X_char, y_, idx_word, idx_tag


def input_data_wc_embd(words, tags, dt_pair):
    # input data using character and word embedding

    max_len = 100
    max_len_char = 10

    word_idx = word2idx(words, 2)
    word_idx["UNK"] = 1
    word_idx["PAD"] = 0
    idx_word = idx2word(word_idx)

    # word embedding
    X_word = [[word_idx[w[0]] for w in s] for s in dt_pair]
    X_word = pad_sequences(maxlen=max_len, sequences=X_word, value=word_idx["PAD"], padding='post', truncating='post')

    chars = set([w_i for w in words for w_i in w])
    # n_chars = len(chars)
    # print(n_chars)

    # character embedding
    char_idx = char2idx(chars, 2)
    char_idx["UNK"] = 1
    char_idx["PAD"] = 0

    X_char = []
    for sentence in dt_pair:
        sent_seq = []
        for i in range(max_len):
            word_seq = []
            for j in range(max_len_char):
                try:
                    word_seq.append(char_idx.get(sentence[i][0][j]))
                except:
                    word_seq.append(char_idx.get("PAD"))
            sent_seq.append(word_seq)
        X_char.append(np.array(sent_seq))
    # X_char[0]

    tag_idx = tag2idx(tags, 1)
    tag_idx["PAD"] = 0
    idx_tag = idx2tag(tag_idx)

    y = [[tag_idx[t[1]] for t in s] for s in dt_pair]
    y = pad_sequences(maxlen=max_len, sequences=y, padding='post', value=tag_idx["PAD"], truncating = 'post')

    return X_word, X_char, y, idx_word, idx_tag


def char2idx(chars, n):
    return {c: i + n for i, c in enumerate(chars)}


def word2idx(words, n):
    return {w: i + n for i, w in enumerate(words)}


def idx2word(word2idx):
    return {i: w for w, i in word2idx.items()}


def tag2idx(tags, n):
    return {t: i + n for i, t in enumerate(tags)}


def idx2tag(tag2idx):
    return {i: w for w, i in tag2idx.items()}


def get_unique_words(df):
    unique_words = list(set(df['Token'].values))
    unique_words.append('ENDPAD')

    return unique_words


def get_unique_tags(df):
    tags = list(set(df['Label'].values))

    return tags


def getWord(text):
  return [item[0] for row in text for item in row]


def getTag(text):
  return [item[1] for row in text for item in row]

def performance_report(y_actual, y_pred):
    labels = ['B-HERITAGE', 'I-HERITAGE', 'B-NATURAL', 'I-NATURAL', 'B-PURPOSE', 'I-PURPOSE', 'O']

    print(classification_report(y_actual, y_pred, labels=labels, digits=4))

    cm = confusion_matrix(y_actual, y_pred, labels=labels)
    # sns.heatmap(cm, annot=True, fmt='d', ax=ax)
    fig, ax = plt.subplots(figsize=(10, 8))

    # plt.figure(figsize=(12, 10))
    # sns.set(rc={'figure.figsize': (14, 12)})
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, annot_kws={'size': 16})
    # annot=True to annotate cells, ftm='g' to disable scientific notation

    # labels, title and ticks
    ax.set_title('Confusion Matrix', fontsize=16)
    ax.set_xlabel('Predicted labels', fontsize=16)
    ax.set_ylabel('True labels', fontsize=16)
    ax.xaxis.set_ticklabels(labels, rotation=45, fontsize=16)
    ax.yaxis.set_ticklabels(labels, rotation=0, fontsize=16)

    plt.show()