import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as T

device = torch.device("cuda")

# Hyper-params
num_epochs = 5
batch_size = 16
learning_rate = .01

# Load MNIST
train_dataset = datasets.MNIST(root="./data",train=True,download=True, transform=T.ToTensor())
test_dataset = datasets.MNIST(root="./data",train=False,download=True, transform=T.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

class Net(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=(5,5), stride=(1,1), padding=(0,0))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5,5), stride=(1,1), padding=(0,0))
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(3,3), stride=(1,1), padding=(0,0))
        
        self.linear1 = nn.Linear(1080, 84)
        self.linear2 = nn.Linear(84, out_channels)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

        self.dropout = nn.Dropout(.25)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = torch.flatten(x, start_dim=1)

        x = self.dropout(x)
        x = self.linear1(x)
        x = self.relu(x)

        x = self.dropout(x)
        x = self.linear2(x)
        return x

model = Net(1, 10)

criterion = nn.CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    for batch, (x,y) in enumerate(train_loader):
        y_predicted = nn.Softmax()(model(nn.functional.pad(x, (2,2,2,2))))

        loss = criterion(y_predicted, y)

        if batch % 100 == 0:
            print(f'Epoch: {epoch+1}, Batch: {batch}, Loss: {loss:.5f}')

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    avg_loss = 0
    correct = 0
    for batch, (x,y) in enumerate(test_loader):
        y_predicted = nn.Softmax()(model(nn.functional.pad(x, (2,2,2,2))))

        avg_loss += criterion(y_predicted, y)

        for vector in range(len(y_predicted)):
            if torch.argmax(y_predicted[vector]) == y[vector]:
                correct += 1

    print(f'Test:    |   AVG LOSS: {avg_loss / len(train_loader)}   | Accuracy: {correct / (len(train_loader) * batch_size)}')
