import torch
import torch.nn as nn
import torchvision.transforms as T

from torch.utils.data import DataLoader
from torchvision import datasets

transform = T.Compose([T.ToTensor()])
batch_size = 100
num_epochs = 10
learning_rate = .01

train_dataset = datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = datasets.FashionMNIST(root="./data", train=False, transform=transform, download=True)

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = torch.flatten(out, start_dim=1)
        out = self.fc(out)
        return out

model = Net()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    for batch, (x, y) in enumerate(train_dataloader):
        y_hat = model(x)

        loss = criterion(y_hat, y)

        optimizer.zero_grad()
        loss.backward()

        if batch%100 == 0:
            print(f'Epoch: {epoch+1} Batch: {batch} / {len(train_dataloader)} Loss: {loss.item()}')

        optimizer.step()

    model.eval()
    avg_loss = 0
    correct = 0
    seen = 0
    for batch, (x, y) in enumerate(test_dataloader):
        y_hat = model(x)
        avg_loss += criterion(y_hat, y)
        predictions = torch.max(y_hat, 1)[1]

        correct += (predictions == y).sum()
        seen += len(y)

    print(f'Test: AVG LOSS: {avg_loss/len(test_dataloader):.5f} Accuracy: {correct*100/seen:.4f}%')