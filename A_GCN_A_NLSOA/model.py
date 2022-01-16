from typing import List

from a_nlsoa import *
from attention_gcn import *


# from config import *


class MultiLabelPatentCategorization(nn.Module):
    """
    Multi-Label Patent Categorization with Non-Local Attention-Based Graph Convolutional Network
    输入为[batch_size, num_word_each_patent]，list不是tensor
    """

    def __init__(self, num_sage_layers, num_heads, num_sample, input_size, lstm_hidden_size, adj_lists,
                 raw_features,
                 num_input, nlsoa_hidden_size, num_class):
        super(MultiLabelPatentCategorization, self).__init__()
        # AttentionGCN
        self.num_sage_layers = num_sage_layers
        self.num_heads = num_heads
        self.num_sample = num_sample
        self.input_size = input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.adj_lists = adj_lists
        self.raw_features = raw_features
        # A_NLSOA
        self.num_input = num_input
        self.nlsoa_input_size = input_size
        self.nlsoa_hidden_size = nlsoa_hidden_size
        self.num_class = num_class

        self.gcn = AttentionGCN(self.num_sage_layers, self.num_heads, self.num_sample, self.input_size,
                                self.lstm_hidden_size, self.adj_lists, self.raw_features)
        self.nlsoa = A_NLSOA(self.num_input, self.nlsoa_input_size, self.nlsoa_hidden_size, self.num_class)
        self.l1 = nn.Linear(self.num_input, 256)
        self.l2 = nn.Linear(256, 1)

    def forward(self, x):
        # x为tensor，但是需要list
        unique_words_list, index_list = self._make_index_dict_2d(x.tolist())
        hidden_embs = self.gcn(unique_words_list)
        features = torch.stack([hidden_embs[index] for index in index_list], dim=0)
        label_vectors = self.nlsoa(features)
        output = F.relu(self.l1(label_vectors.permute(0, 2, 1)))
        output = F.softmax(self.l2(output), dim=1)
        output = output.squeeze()
        return output

    def _make_index_dict_2d(self, x):
        unique_words_list = list(set.union(*[set(i) for i in x]))
        unique_words_dict = dict(list(zip(unique_words_list, list(range(len(unique_words_list))))))
        index_list = [[unique_words_dict[word] for word in words] for words in x]
        return unique_words_list, index_list


# adj = [[2, 3, 1], [0, 2, 3], [1, 0, 3], [0, 2, 1]]
# raw_feature = torch.randn(4, 6)
# net = MultiLabelPatentCategorization(2, 2, 3, 6, 10, adj, raw_feature, 2, 3, 4)
# x = torch.tensor([[1, 2], [2, 3], [0, 2]])
# print(net(x).shape)