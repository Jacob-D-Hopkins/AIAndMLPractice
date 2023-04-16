import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np

rawData = np.loadtxt('./data/wine/wine.csv', delimiter=";", dtype=np.float32, skiprows=1)

inputs_array = rawData[: , :-1]
outputs_array = rawData[: , [-1]]

inputs = torch.from_numpy(inputs_array).type(torch.float32)
targets = torch.from_numpy(outputs_array).type(torch.float32)

dataset = TensorDataset(inputs, targets)

train_ds, test_ds = random_split(dataset, [1400, 199])
batch_size = 50
train_loader = DataLoader(train_ds, batch_size, shuffle=True)
test_loader = DataLoader(test_ds, batch_size)

num_features = inputs.size()[1]
num_epochs = 10000
model = nn.Linear(num_features, 1)

learning_rate = 0.001
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for batch in train_loader:
        inputs, targets = batch
        y_predicted = model(inputs)

        loss = criterion(y_predicted, targets)

        loss.backward()

        optimizer.step()

        optimizer.zero_grad()

    if epoch % 100 == 0:
        print(f'epoch: {epoch}, loss = {loss.item():.4f}')
    if epoch % 250 == 0:
        with torch.no_grad():
            for batch in test_loader:
                inputs, targets = batch
                predicted = model(inputs)
                predicted_cls = predicted.round()
                acc = predicted_cls.eq(targets).sum()/float(batch_size)
                print(f'acc: {acc:.4f}')
