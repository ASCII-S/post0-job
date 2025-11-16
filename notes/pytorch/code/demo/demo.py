import torch
from torchvision import datasets,transforms

# data preprocess
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

# load data
train_data = datasets.MNIST(root='./data/',train=True,download=True,transform=transform)
test_data = datasets.MNIST(root='./data/',train=False,download=True,transform=transform)

# data loader
train_loader = torch.utils.data.DataLoader(train_data,batch_size=64,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data,batch_size=64,shuffle=True)

import torch.nn as nn

class demoNN(nn.Module):
    def __init__(self):
        super(demoNN,self).__init__()
        self.fc1 = nn.Linear(28*28,128)
        self.fc2 = nn.Linear(128,10)

    def forward(self,x):
        x = x.view(-1,28*28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# loss function
loss_fn = nn.CrossEntropyLoss()

# optimizer
import torch.optim as optim

model = demoNN()
optimizer = optim.SGD(model.parameters(),lr=0.001)

# train
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images,labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs,labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}")


# test
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images,labels in test_loader:
        outputs = model(images)
        _,predicted = torch.max(outputs,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total * 100.0
print(f"Accuracy: {accuracy:.2f}%")