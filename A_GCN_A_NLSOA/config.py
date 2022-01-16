
class Config(object):

    def __init__(self):
        # AttentionGCN
        self.num_sage_layers = 2
        self.num_heads = 20
        self.num_sample = 10
        self.input_size = 100
        self.lstm_hidden_size = 256
        # A_NLSOA
        self.num_input = 100
        self.nlsoa_hidden_size = 50
        self.num_class = 650   # num_class