import torch
import torch.nn as nn
import torchvision.transforms as T

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

transform = T.Compose([T.ToTensor()])
batch_size = 100
num_epochs = 50
learning_rate = .0005
weight_decay = .0001

train_dataset = ImageFolder(root="./data/FruitDataset/Training", transform=transform)
test_dataset = ImageFolder(root="./data/FruitDataset/Test", transform=transform)

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.Dropout(.25),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(.5)
        self.linear1 = nn.Linear(4608, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 131)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = torch.flatten(x, start_dim=1)
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x

model = Net().to("cuda")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for batch, (x, y) in enumerate(train_dataloader):
        optimizer.zero_grad()
        x, y = x.to("cuda"), y.to("cuda")

        y_hat = model(x)
        loss = criterion(y_hat, y)

        loss.backward()
        optimizer.step()

        if batch%50 == 0:
            print(f'Epoch: {epoch+1}    Batch: {batch} / {len(train_dataloader)}    Loss: {loss.item():.5f}')

    avg_loss = 0
    correct = 0
    seen = 0
    with torch.no_grad():
        for batch, (x, y) in enumerate(test_dataloader):
            x, y = x.to("cuda"), y.to("cuda")
            y_hat = model(x)
            avg_loss += criterion(y_hat, y)
            predictions = torch.max(y_hat, 1)[1]

            correct += (predictions == y).sum()
            seen += len(y)

    print(f'Test:   AVG LOSS: {avg_loss/len(test_dataloader):.5f}    Accuracy: {correct*100/seen:.5f}%')