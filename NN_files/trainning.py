#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from nnet import ReadDataset, Net
import time
from loss_functions import interval_score_loss




tic = time.time()

# Read data
train_file = "data/NHANESI.csv"
trainset = ReadDataset(train_file)


# Data loaders
trainloader = DataLoader(trainset, batch_size=100, shuffle=True)


# Use gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Neural Network
nnet = Net(trainset.__shape__()).to(device)

# Loss function
criterion = nn.MSELoss()

# Optimizer
optimizer = optim.Adam(
    nnet.parameters(), lr=0.1, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.000001
)


# Train the net
loss_per_iter = []
loss_per_batch = []


# Train the net
losses = []
auc_train = []
auc_test = []

# hyperparameteres
n_epochs = 10

for epoch in range(n_epochs):
    print(epoch)

    for i, (inputs, labels) in enumerate(trainloader):
        X = inputs.to(device)
        y = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forwarde
        outputs = nnet(X.float())

        # Compute diff


        loss = interval_score_loss(outputs, y.float())

        # Compute gradient
        loss.backward()

        # update weights
        optimizer.step()

        # Save loss to plot

        losses.append(loss.item())

        if i % 50 == 0:
            auc_train.append(loss.detach().numpy())


            # Figure
            plt.figure()
            plt.plot(auc_train, label="train")
            plt.legend()
            plt.savefig("output/auc_NN.png")
            plt.savefig("output/auc_NN.svg", format="svg")
            plt.close()

print("Elapsed time: ", np.abs(tic-time.time()))
print("done")
