import torch
import torch.nn as nn
import torch.nn.functional as F


class A_NLSOA(nn.Module):
    """
    Adaptive Non-local Second-order Attention Layer
    the input should be [batch_size, num_input, input_size] batch大小，单词数量，嵌入向量大小
    """

    def __init__(self, num_input, input_size, hidden_size, output_size):
        super(A_NLSOA, self).__init__()
        self.num_input = num_input
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.matrix = nn.Parameter((torch.eye(self.hidden_size) - (1. / self.hidden_size)) / self.hidden_size, requires_grad=False)
        self.normal_factor = nn.Parameter(torch.sqrt(torch.tensor([self.hidden_size])), requires_grad=False)
        # attention theta(x)
        self.l1 = nn.Linear(self.input_size, self.hidden_size)
        self.bn = nn.BatchNorm1d(self.num_input)
        # g(X)
        self.l2 = nn.Linear(self.input_size, self.hidden_size)
        self.l3 = nn.Linear(self.hidden_size, self.output_size)
        # phi(x)
        self.l4 = nn.Linear(self.input_size, self.output_size)

    def forward(self, x):
        theta_x = F.leaky_relu(self.bn(self.l1(x)))
        sigma = torch.bmm(theta_x, self.matrix.unsqueeze(0).repeat((theta_x.shape[0], 1, 1)))
        sigma = torch.bmm(sigma, theta_x.permute(0, 2, 1))
        attention = sigma / self.normal_factor  # softmax对整个矩阵做
        attention = F.softmax(attention.view((x.shape[0], -1)), dim=1).view((-1, self.num_input, self.num_input))
        # 暂未加激活函数
        g = self.l2(x)
        p = self.l3(torch.bmm(attention, g))
        phi = self.l4(x)
        result = phi + p
        return result
