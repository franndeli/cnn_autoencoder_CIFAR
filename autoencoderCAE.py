import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from constants import BATCH_SIZE

class AutoencoderCAE(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1) # input channels: 3 (RGB), output channels: 8, kernel size: 3x3
        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0) # kernel size: 2x2, stride: defaults to kernel size
        self.conv2 = nn.Conv2d(8, 12, 3, padding=1) # input channels: 8, output channels: 12, kernel size: 3x3
        self.conv3 = nn.Conv2d(12, 16, 3, padding=1) # input channels: 12, output channels: 16, kernel size: 3x3

        # Decoder
        self.upsample = nn.Upsample(scale_factor=2) # scale factor: 2
        self.conv4 = nn.Conv2d(16, 12, 3, padding=1) # input channels: 16, output channels: 12, kernel size: 3x3
        self.conv5 = nn.Conv2d(12, 3, 3, padding=1) # input channels: 12, output channels: 3, kernel size: 3x3

    def forward(self, input):

        #Encoder
        x = F.relu(self.conv1(input))   # 32x32x3 -> 32x32x8
        x = self.maxpool(x)             # 32x32x8 -> 16x16x8
        x = F.relu(self.conv2(x))       # 16x16x8 -> 16x16x12
        x = self.maxpool(x)             # 16x16x12 -> 8x8x12
        x = F.relu(self.conv3(x))       # 8x8x12 -> 8x8x16 (latent space)

        #Decoder
        x = self.upsample(x)            # 8x8x16 -> 16x16x16
        x = F.relu(self.conv4(x))       # 16x16x16 -> 16x16x12
        x = self.upsample(x)            # 16x16x12 -> 32x32x12
        x = self.conv5(x)               # 32x32x12 -> 32x32x3
        x = torch.sigmoid(x)            # [0,1]

        return x
    

def prepare_dataloaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    total_trainset = len(dataset)
    train_size = int(0.8 * total_trainset)
    valid_size = int(0.1 * total_trainset)
    test_size = total_trainset - train_size - valid_size

    trainset, validset, testset = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size])

    trainset = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    validset = torch.utils.data.DataLoader(validset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    testset = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return trainset, validset, testset