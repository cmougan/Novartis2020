import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from gauss_rank_scaler import GaussRankScaler

from sklearn.externals import joblib
import random
import os

# os.path.isfile(fname)

random.seed(0)


class ReadDataset(Dataset):
    """Read dataset."""

    def __init__(self, csv_file):
        """
        Args:
            csv_file (str): Path to the csv file with the students data.

        """
        self.df = pd.read_csv(csv_file).fillna(0)
        self.df.columns = self.df.columns.str.replace(" ", "")


        self.X = self.df.drop(columns = 'target')
        self.y = self.df.target.values

        self.scaler = GaussRankScaler()
        self.X = pd.DataFrame(self.scaler.fit_transform(self.X),columns=self.X.columns)

    def __len__(self):
        return len(self.X)

    def __shape__(self):
        return self.X.shape[1]

    def __getitem__(self, idx):
        # Convert idx from tensor to list due to pandas bug (that arises when using pytorch's random_split)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        self.X.iloc[idx].values
        self.y[idx]

        return [self.X.iloc[idx].values, self.y[idx]]


class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.relu1 = nn.SELU()
        self.batchnorm1 = nn.BatchNorm1d(input_dim)
        self.drop1 = nn.Dropout(0.05, inplace=False)

        self.fc2 = nn.Linear(input_dim, 2, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        self.batchnorm1(x)
        self.drop1(x)

        x = self.fc2(x)

        return x.squeeze()
