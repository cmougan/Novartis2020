import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from gauss_rank_scaler import GaussRankScaler
from sklearn.model_selection import train_test_split

import random
import os


random.seed(0)


class ReadDataset(Dataset):
    """Read dataset."""

    def __init__(self, csv_file, isTrain=None):
        """
        Args:
            csv_file (str): Path to the csv file with the students data.

        """
        self.df = pd.read_csv(csv_file).fillna(0)
        self.df.columns = self.df.columns.str.replace(" ", "")

        self.X = self.df.drop(
            columns=["target", "Cluster", "brand_group", "cohort", "Country"]
        )
        self.y = self.df.target.values

        self.scaler = GaussRankScaler()
        self.X = pd.DataFrame(self.scaler.fit_transform(self.X), columns=self.X.columns)

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, random_state=42
        )
        if isTrain == True:
            self.X = X_train
            self.y = y_train
        else:
            self.X = X_test
            self.y = y_test

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

        #self.fc2 = nn.Linear(2*input_dim, input_dim)
        #self.relu2 = nn.SELU()
        #self.batchnorm2 = nn.BatchNorm1d(input_dim)
        #self.drop2 = nn.Dropout(0.05, inplace=False)

        self.fc3 = nn.Linear(input_dim, 2, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.batchnorm1(x)
        x = self.drop1(x)
        '''
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.batchnorm2(x)
        x = self.drop2(x)
        '''

        x = self.fc3(x)

        return x.squeeze()
