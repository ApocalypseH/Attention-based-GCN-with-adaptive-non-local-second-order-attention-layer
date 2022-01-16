from model import *
import torch
import numpy as np
import json
import pandas as pd
from gensim.models import Word2Vec
from torch.utils.data import DataLoader
from utils import *
import time
from train_eval import *

print('Loading adjacent lists...', end='')
with open('adj_lists.json', 'r') as file:
    adj_lists = json.load(file)
print('Finished!')

# print('Loading label list...')
# with open('label_list.json', 'r') as file:
#     label_list = json.load(file)
# print('Finished!')

print('Loading word embeddings...', end='')
model = Word2Vec.load('vocabulary.model')
raw_features = torch.tensor(model.wv[[i for i in range(len(model.wv))]])
print('Finished!')

print('Loading data...', end='')
df = pd.read_json('id_topwd_label_zero_pad.json', orient='table')
print('Finished!')
# df[['patent_id', 'top_words', 'labels']]

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True  # 保证每次结果一样

print('Making data_iter...', end='')
# train_file = 'train_data.json'
train_file = 'train_data_minor.json'
train_set = PatentDataset(train_file)
train_iter = DataLoader(train_set, batch_size=128, collate_fn=collate, shuffle=True)
# dev_file = 'dev_data.json'
dev_file = 'dev_data_minor.json'
dev_set = PatentDataset(dev_file)
dev_iter = DataLoader(dev_set, batch_size=128, collate_fn=collate, shuffle=False)
# test_file = 'test_data.json'
test_file = 'test_data_minor.json'
test_set = PatentDataset(test_file)
test_iter = DataLoader(test_set, batch_size=128, collate_fn=collate, shuffle=False)
print('Finished!')

net = MultiLabelPatentCategorization(2, 20, 10, 100, 256, adj_lists, raw_features, 100, 50, 650)
# for words, labels in train_iter:
#     result = net(words)
#     print(len(result), len(result[0]))


print('Training model...')
train(net, train_iter, dev_iter, test_iter, 30, 'cpu', 0.5, 5)


# if __name__ == '__main__':
#     train()
