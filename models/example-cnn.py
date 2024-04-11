# Example CNN below -> starting with pytorch's example here https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# CNN structure has been changed to account for 96x96 images in dataset

import torch
from torch.utils.data import Dataset
import os
from torchvision.io import read_image
import torchvision.transforms as transforms
import pandas as pd
import sklearn.model_selection
import torch.nn as nn
import torch.nn.functional as F
pd.options.mode.chained_assignment = None

random_seed = 517
test_percent = 0.2
labels_file = "..\\data\\labels.csv"
save_path = ".\\trained_models\\example-cnn.pt"

# Separate dataset into train, validate, test
# Pull in labels CSV
labels = pd.read_csv(os.path.join(os.path.dirname(__file__), labels_file), index_col=0)
classes = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
X = labels['pth']
Y = labels[["pth", "label"]]
Y['label'] = Y['label'].apply(lambda x: classes.index(x))

_, _, train, test = sklearn.model_selection.train_test_split(X, Y, test_size = test_percent, random_state = random_seed)

class FacesDataset(Dataset):
    def __init__(self, img_labels, img_dir, transform=None, target_transform=None):
        self.img_labels = img_labels
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
transform = transforms.Compose([transforms.ConvertImageDtype(torch.float), transforms.Grayscale(), transforms.Normalize(0.5, 0.5)])

train_set = torch.utils.data.DataLoader(FacesDataset(train, os.path.join(os.path.dirname(__file__), "../data"), transform=transform), batch_size=4, shuffle=True)
test_set = torch.utils.data.DataLoader(FacesDataset(test, os.path.join(os.path.dirname(__file__), "../data"), transform=transform), batch_size=4, shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=2)
        self.fc1 = nn.Linear(16 * 24 * 24, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 8)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

net.to(device)

for epoch in range(11):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_set, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 1000 == 999:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 1000:.3f}')
            running_loss = 0.0

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_set:
        images, labels = data[0].to(device), data[1].to(device)
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the test images: {100 * correct // total} %')

print("Saving model to " + save_path)
torch.save(net, os.path.join(os.path.dirname(__file__), save_path))