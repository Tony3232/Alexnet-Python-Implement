import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3,96,11,stride=4)
        self.conv2 = nn.Conv2d(96,256,5,stride=1,padding=2)
        self.conv3 = nn.Conv2d(256,384,3,stride=1,padding=1)
        self.conv4 = nn.Conv2d(384,384,3,stride=1,padding=1)
        self.conv5 = nn.Conv2d(384,256,3,stride=1,padding=1)
        self.pool = nn.MaxPool2d(3,2)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(256*6*6,4096)
        self.fc2 = nn.Linear(4096,4096)
        self.fc3 = nn.Linear(4096,10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(F.relu(self.conv5(x)))
        x = x.view(-1, 256*6*6)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x
