import torch
import torch.nn as nn
import torchvision.transforms as T

from torch.utils.data import DataLoader
from torchvision import datasets

stats = ((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025))
transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomCrop(32, padding = 4, padding_mode="reflect"),
    T.ToTensor(),
    T.Normalize(*stats)
])


batch_size = 128
num_epochs = 500
learning_rate = .0005
weight_decay = .0001

train_dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(16384, 4096),
            nn.Dropout(.25),
            nn.ReLU(),

            nn.Linear(4096, 1024),
            nn.Dropout(.25),
            nn.ReLU(),

            nn.Linear(1024, 256),
            nn.Dropout(.25),
            nn.ReLU(),

            nn.Linear(256, 100)
        )


    def forward(self, x):
        return self.network(x)

model = Net().to("cuda")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

for epoch in range(num_epochs):
    for batch, (x, y) in enumerate(train_dataloader):
        x, y = x.to("cuda"), y.to("cuda")
        optimizer.zero_grad()
        
        y_hat = model(x)
        loss = criterion(y_hat, y)

        loss.backward()
        optimizer.step()

        if batch%100 == 0:
            print(f'Epoch: {epoch+1} Batch: {batch} / {len(train_dataloader)} Loss: {loss.item()}')

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

    print(f'Test: AVG LOSS: {avg_loss/len(test_dataloader):.5f} Accuracy: {correct*100/seen:.4f}%')