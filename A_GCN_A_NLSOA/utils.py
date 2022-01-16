from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch


class PatentDataset(Dataset):

    def __init__(self, json_file_path):
        self.df_data = pd.read_json(json_file_path, orient='table')

    def __len__(self):
        return len(self.df_data)

    def __getitem__(self, index):
        words = self.df_data['top_words'][index]
        labels = self.df_data['labels'][index]
        return words, labels


def collate(x):
    words = np.stack([i[0] for i in x]).tolist()
    labels = torch.tensor(np.stack([i[1] for i in x]))
    return [words, labels]