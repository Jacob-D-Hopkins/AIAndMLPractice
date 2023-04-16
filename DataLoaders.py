import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import math

class WineTrainDataset(Dataset):
    def __init__(self):
        xy = np.loadtxt('./data/wine/wine.csv', delimiter=";", dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, :-1])
        self.y = torch.from_numpy(xy[:, [-1]])
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples

class WineTestDataset(Dataset):
    def __init__(self):
        xy = np.loadtxt('./data/wine/winetest.csv', delimiter=";", dtype=np.float32)
        self.x = torch.from_numpy(xy[:, :-1])
        self.y = torch.from_numpy(xy[:, [-1]])
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples

trainDataset = WineTrainDataset()
trainDataLoader = DataLoader(dataset=trainDataset, batch_size=4, shuffle=True)

num_epochs = 100
total_samples = len(trainDataset)
n_iterations = math.ceil(total_samples/4)

n_samples, n_features = trainDataset.x.shape
class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)
    
    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted

model = LogisticRegression(n_features)

learning_rate = .01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    y_predicted = model(trainDataset.x)

    loss = criterion(y_predicted, trainDataset.y)

    loss.backward()

    optimizer.step()

    optimizer.zero_grad()
    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

testDataset = WineTestDataset()
testDataLoader = DataLoader(dataset=testDataset, batch_size=4, shuffle=True)

with torch.no_grad():
    y_predicted = model(testDataset.x)
    y_predicted_cls = 10*y_predicted.round()
    acc = y_predicted_cls.eq(testDataset.y).sum()/float(testDataset.y.shape[0])
    print(f'accuracy = {acc:.4f}')