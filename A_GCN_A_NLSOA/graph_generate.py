import random

import pandas as pd
import numpy as np
import re

import torch
from gensim.models import Word2Vec
import time
from scipy import sparse
import json

print('Loading raw file...')
df = pd.read_json('processed_patent_data.json')
print('Finished!')

# df = df.iloc[:1000]

df['text'] = df['title'] + ' ' + df['abstract'] + ' ' + df['claim'].apply(lambda x: ' '.join(x))
# df['text'] = df['title']
df['labels'] = df['cpc'].apply(lambda x: set([hier_label['group_id'] for hier_label in x]))


def text2words(text):
    clean = re.sub('[^\w-]+|\B-+|-+\B', ' ', text)
    words = clean.split()
    return words


df['words'] = df['text'].apply(text2words)
corpus = df['words'].tolist()

print('Training word2vec model...')
# model = Word2Vec(corpus, vector_size=100, min_count=5)
# model.save('vocabulary.model')
model = Word2Vec.load('vocabulary.model')
# model.wv.index_to_key
# model.wv[model.wv.index_to_key]
print('Finished!')

# word_dict = {}
# for words in df['words']:
#     for word in words:
#         if word_dict.get(word):
#             word_dict[word] += 1
#         else:
#             word_dict[word] = 1
#
# word_list = []
# skipword_list = []
# for word in word_dict:
#     if word_dict[word] >= 5:
#         word_list.append(word)
#     else:
#         skipword_list.append(word)
print('Getting number-like labels...')
temp_dict = dict()
patent_count = 0
patent_num = len(df)


def make_label_dict(labels):
    global temp_dict
    global patent_count
    global patent_num
    for label in labels:
        temp_dict[label] = 1
    patent_count += 1
    if patent_count % 1000 == 0:
        print(patent_count, 'out of', patent_num, 'patents\' labels got!\t',
              time.asctime(time.localtime(time.time())))


for i in range(len(df)):
    make_label_dict(df['labels'][i])

def list2vec(index_list, vec_length):
    vector = torch.zeros(vec_length)
    vector[index_list] = 1
    return vector.tolist()


label_list = list(temp_dict)
label_dict = dict(zip(label_list, list(range(len(label_list)))))
df['labels'] = df['labels'].apply(lambda x: [label_dict[label] for label in x])
df['labels'] = df['labels'].apply(lambda x: list2vec(x, 650))
print('Finished!')

word_list = model.wv.index_to_key
word_dict = dict(zip(word_list, range(len(word_list))))


def word2digit(words):
    global patent_count
    global patent_num
    global word_list
    global word_dict
    digits = [word_dict[word] for word in words if word in word_list]
    patent_count += 1
    if patent_count % 1000 == 0:
        print(patent_count, 'out of', patent_num, 'patents\' words transformed into number!\t',
              time.asctime(time.localtime(time.time())))
    return digits


def get_top_n_words(word_list, num=100):
    global patent_count
    global patent_num
    dict_ = {}
    for elem in word_list:
        if dict_.get(elem):
            dict_[elem] += 1
        else:
            dict_[elem] = 1
    unique_elem = list(dict_)
    unique_elem.sort(key=lambda x: dict_[x], reverse=True)
    if len(unique_elem) > num:
        result = unique_elem[:num]
    else:
        result = unique_elem
    result.sort(key=lambda x: word_list.index(x))

    patent_count += 1
    if patent_count % 100 == 0:
        print(patent_count, 'out of', patent_num, 'patents\' top-n words got!\t',
              time.asctime(time.localtime(time.time())))
    return result


def padding(data, num):
    global patent_count
    global patent_num
    if len(data) >= num:
        result = data[:num]
    else:
        result = data + [0] * (num - len(data))
    patent_count += 1
    if patent_count % 100 == 0:
        print(patent_count, 'out of', patent_num, 'patents\' top-n words padded!\t',
              time.asctime(time.localtime(time.time())))
    return result


print('Getting top-n words...')
patent_count = 0
patent_num = len(df)
df['words'] = df['words'].apply(word2digit)
patent_count = 0
patent_num = len(df)
df['top_words'] = df['words'].apply(get_top_n_words)
patent_count = 0
patent_num = len(df)
df['top_words'] = df['top_words'].apply(lambda x: padding(x, 100))
print('Finished!')

unique_word_num = len(word_list)
self_presence = np.zeros(unique_word_num, dtype=int)
co_presence = np.zeros((unique_word_num, unique_word_num), dtype=int)


def self_register(words):
    global self_presence
    for word in words:
        self_presence[word] += 1
    return


def co_register(word_pairs):
    global co_presence
    for word1, word2 in word_pairs:
        co_presence[word1][word2] += 1
    return


def slide_and_count(text, window_size=20):
    global patent_count
    global patent_num
    self_pre = {}
    co_pre = {}
    num_window_ = 0
    window_size = min(len(text), window_size)
    if len(text) == window_size:
        num_window_ += 1
        for i in range(window_size):
            self_pre[text[i]] = 1
            for j in range(i + 1, window_size):
                if text[i] != text[j]:
                    co_pre[(text[i], text[j])] = 1
                    co_pre[(text[j], text[i])] = 1
        self_register(self_pre)
        self_pre.clear()
        co_register(co_pre)
        co_pre.clear()
    else:
        for window_start in range(len(text) - window_size + 1):
            num_window_ += 1
            for i in range(window_start, window_start + window_size):
                self_pre[text[i]] = 1
                for j in range(i + 1, window_start + window_size):
                    if text[i] != text[j]:
                        co_pre[(text[i], text[j])] = 1
                        co_pre[(text[j], text[i])] = 1
            self_register(self_pre)
            self_pre.clear()
            co_register(co_pre)
            co_pre.clear()

    patent_count += 1
    if patent_count % 100 == 0:
        print(patent_count, 'out of', patent_num, 'patents\' words co-appearance got!\t',
              time.asctime(time.localtime(time.time())))

    return num_window_


print('Generating graph...')
patent_count = 0
patent_num = len(df)
df['num_window'] = df['words'].apply(slide_and_count)
num_window = df['num_window'].sum()

presence = self_presence.reshape(1, -1)
adj_matrix = ((co_presence * len(word_list) / (presence.T * presence)) > 1).astype(int)
print('Finished!')
# 邻接矩阵 adj_matrix
# 每篇专利所选words df['top_words']/df.top_words
# 所有words, words:编号 word_list, word_dict

result = df[['patent_id', 'top_words', 'labels']]
result.to_json('id_topwd_label_high.json', orient='table')
# with open('id_topwd_label.json', 'w') as f:
#     json.dump(result.to_json(), f)

# np.save('adj_matrix.npy', adj_matrix)
adj_matrix_sp = sparse.csr_matrix(adj_matrix)
sparse.save_npz('adj_matrix_sp.npz', adj_matrix_sp)
vector_count = 0
adj = []
for adj_vector_sp in adj_matrix_sp:
    adj_vector = adj_vector_sp.toarray()[0]
    adj.append([i for i in range(len(adj_vector)) if adj_vector[i] != 0])
    vector_count += 1
    if vector_count % 100 == 0:
        print(vector_count, 'sparse vectors trasformed!')

with open('adj_lists.json', 'w') as f:
    json.dump(adj, f)
