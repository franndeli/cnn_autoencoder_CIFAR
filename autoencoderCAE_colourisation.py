import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
from skimage.color import rgb2lab, lab2rgb
import numpy as np

from constants import BATCH_SIZE

class AutoencoderCAE(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        
        # Decoder
        self.conv4 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv6 = nn.Conv2d(64, 2, 3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x):
        # Encoder
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        
        # Decoder
        x = self.upsample(x)
        x = F.relu(self.conv4(x))
        x = self.upsample(x)
        x = F.relu(self.conv5(x))
        x = torch.tanh(self.conv6(x))
        
        return x


class ColorizationDataset(Dataset):
    def __init__(self, train=True):
        self.data = torchvision.datasets.CIFAR10(
            root='./data', train=train, download=True,
            transform=transforms.ToTensor()
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rgb_img, label = self.data[idx]

        rgb = rgb_img.permute(1, 2, 0).numpy()
        lab = rgb2lab(rgb).astype("float32")

        L  = lab[:, :, 0:1]
        ab = lab[:, :, 1:]

        L = L / 100.0              # [0,100] -> [0,1]
        ab = ab / 128.0            # [-128,127] -> [-1,1]

        L  = torch.from_numpy(L).permute(2, 0, 1)
        ab = torch.from_numpy(ab).permute(2, 0, 1)

        return L, ab


def prepare_dataloaders():    
    train_dataset = ColorizationDataset(train=True)
    
    total_size = len(train_dataset)
    train_size = int(0.8 * total_size)
    valid_size = int(0.1 * total_size)
    test_size = total_size - train_size - valid_size
    
    trainset, validset, testset = torch.utils.data.random_split(
        train_dataset, [train_size, valid_size, test_size]
    )
    
    trainloader = DataLoader(
        trainset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0
    )
    validloader = DataLoader(
        validset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=0
    )
    testloader = DataLoader(
        testset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=0
    )
    
    return trainloader, validloader, testloader

def lab_to_rgb(L, ab):
    L = L * 100.0
    ab = ab * 128.0

    L = L.cpu().numpy()
    ab = ab.cpu().numpy()

    B, _, H, W = L.shape
    rgbs = []

    for i in range(B):
        lab = np.zeros((H, W, 3), dtype="float32")
        lab[:, :, 0] = L[i, 0]
        lab[:, :, 1:] = np.transpose(ab[i], (1, 2, 0))
        rgb = lab2rgb(lab)
        rgbs.append(rgb)

    return np.stack(rgbs, axis=0)