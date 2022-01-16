import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools


class AttentionSAGE(nn.Module):
    """
    第一输入为[num_batch/num_nodes_batch_word, seq_length/num_neighbor_word, feature_dim]
    第二输入为[num_batch/num_nodes_batch_word, feature_dim]
    输出为[num_batch/num_nodes_batch_node, feature_dim]
    """

    def __init__(self, input_size, hidden_size, num_heads):
        super(AttentionSAGE, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert self.input_size % self.num_heads == 0, 'Feature_dim must be divided by num_heads!'
        self.head_output_size = self.input_size / self.num_heads
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, 1, batch_first=True, bidirectional=True)
        self.weight_matrix = nn.Parameter(torch.FloatTensor(self.num_heads, self.input_size + 2 * self.hidden_size),
                                          requires_grad=True)
        nn.init.xavier_uniform_(self.weight_matrix, gain=1)

    def forward(self, neighbor_x, x):
        output_rnn = self.rnn(neighbor_x)[0]
        cat = x.unsqueeze(1).repeat((1, neighbor_x.shape[1], 1))
        input_head = torch.cat((output_rnn, cat), dim=2)
        attention = torch.bmm(self.weight_matrix.unsqueeze(0).repeat((neighbor_x.shape[0], 1, 1)),
                              input_head.permute(0, 2, 1))
        attention = F.softmax(F.leaky_relu(attention, negative_slope=0.1), dim=2)
        output_heads = torch.bmm(attention, neighbor_x)  # [batch, num_heads, feature_dim]
        # 先采取每个head的结果求平均，不然感觉参数很多
        output = output_heads.mean(dim=1)
        return output


class AttentionGCN(nn.Module):

    def __init__(self, num_layers, num_heads, num_sample, input_size, hidden_size, adj_lists, raw_features):
        super(AttentionGCN, self).__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_sample = num_sample
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.adj_lists = adj_lists
        self.raw_features = raw_features

        for index in range(1, num_layers + 1):
            setattr(self, 'sage_layer' + str(index), AttentionSAGE(self.input_size, self.hidden_size, self.num_heads))

    def forward(self, nodes_batch):
        lower_layer_nodes = list(nodes_batch)
        nodes_batch_layers = [(lower_layer_nodes,), ]
        for i in range(self.num_layers):
            lower_samp_neighs, lower_layer_nodes_dict, lower_layer_nodes = self._get_unique_neighs_list(
                lower_layer_nodes)
            nodes_batch_layers.insert(0, (lower_layer_nodes, lower_samp_neighs, lower_layer_nodes_dict))

        nb = nodes_batch_layers[0][0]
        pre_hidden_embs = self.raw_features[nb]
        for index in range(1, self.num_layers + 1):
            nb = nodes_batch_layers[index][0]
            pre_neighs = nodes_batch_layers[index - 1]
            nb = self._nodes_map(nb, pre_neighs)
            neighbor_x, x = self._make_sage_input(nb, pre_hidden_embs, pre_neighs)
            sage_layer = getattr(self, 'sage_layer' + str(index))

            # neighbor_x = neighbor_x.cuda()
            # x = x.cuda()

            cur_hidden_embs = sage_layer(neighbor_x, x)
            pre_hidden_embs = cur_hidden_embs

        return pre_hidden_embs

    def _make_sage_input(self, nb, pre_hidden_embs, pre_neighs):
        unique_nodes_list, samp_neighs, unique_nodes = pre_neighs
        x = pre_hidden_embs[nb]
        neighbor_x = torch.stack(
            [pre_hidden_embs[self._nodes_map(samp_neigh, pre_neighs)] for samp_neigh in samp_neighs], dim=0)
        return neighbor_x, x

    def _nodes_map(self, nodes, neighs):
        layer_nodes, samp_neighs, layer_nodes_dict = neighs
        index = [layer_nodes_dict[x] for x in nodes]
        return index

    def _sample_and_pad(self, data: list, num, pad):
        _sample = random.sample
        # times = num // len(data)
        # sample_num = num % len(data)
        # result = data[:] * times + _sample(data, sample_num)
        num_sample = min(len(data), num)
        num_pad = num - num_sample
        result = _sample(data, num_sample) + [pad] * num_pad
        return result

    def _get_unique_neighs_list(self, nodes, num_sample=10):
        _set = set
        to_neighs = [self.adj_lists[int(node)] for node in nodes]
        samp_neighs = [self._sample_and_pad(to_neighs[i], num_sample, int(nodes[i])) for i in range(len(to_neighs))]
        _unique_nodes_list = list(set(itertools.chain(*(samp_neighs + [nodes]))))
        i = list(range(len(_unique_nodes_list)))
        unique_nodes = dict(list(zip(_unique_nodes_list, i)))
        return samp_neighs, unique_nodes, _unique_nodes_list

# adj = [[2, 3, 1], [0, 2, 3], [1, 0, 3], [0, 2, 1]]
# net = AttentionGCN(2, 2, 3, 6, 10, adj, torch.randn(4, 6))
# nb = [0, 2, 1, 1]
# print(net(nb).shape)
