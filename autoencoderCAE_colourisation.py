import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
# from skimage.color import rgb2lab, lab2rgb
import numpy as np

from constants import BATCH_SIZE

class AutoencoderCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder: acepta L channel (1 canal)
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 24, 3, padding=1)
        self.conv3 = nn.Conv2d(24, 32, 3, padding=1)
        
        # Decoder: predice ab channels (2 canales)
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv4 = nn.Conv2d(32, 24, 3, padding=1)
        self.conv5 = nn.Conv2d(24, 3, 3, padding=1)


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
        x = self.conv5(x)
        x = torch.sigmoid(x)
        
        return x


class ColorizationDataset(Dataset):
    def __init__(self, train=True):
        transform = transforms.Compose([transforms.ToTensor()])
        self.data = torchvision.datasets.CIFAR10(
            root='./data', train=train, download=True, transform=transform
        )
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        rgb_img, label = self.data[idx]
        
        # Convertir a grayscale
        grayscale = 0.299 * rgb_img[0] + 0.587 * rgb_img[1] + 0.114 * rgb_img[2]
        grayscale = grayscale.unsqueeze(0)
        
        return grayscale, rgb_img


def prepare_dataloaders():
    """Prepara dataloaders con LAB color space"""
    
    # Crear datasets
    train_dataset = ColorizationDataset(train=True)
    
    # Split 80/10/10
    total_size = len(train_dataset)
    train_size = int(0.8 * total_size)
    valid_size = int(0.1 * total_size)
    test_size = total_size - train_size - valid_size
    
    trainset, validset, testset = torch.utils.data.random_split(
        train_dataset, [train_size, valid_size, test_size]
    )
    
    # Create dataloaders
    trainloader = DataLoader(
        trainset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0  # Cambiar a 2-4 si tienes CPU potente
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


# def lab_to_rgb(L, ab):
#     """
#     Convierte L y ab de vuelta a RGB
    
#     Args:
#         L: tensor (batch, 1, H, W) normalizado [-1, 1]
#         ab: tensor (batch, 2, H, W) normalizado [-1, 1]
    
#     Returns:
#         rgb: numpy array (batch, H, W, 3) en rango [0, 1]
#     """
#     # Desnormalizar
#     L = (L + 1) * 50          # [-1, 1] → [0, 100]
#     ab = ab * 110             # [-1, 1] → [-110, 110]
    
#     batch_size = L.shape[0]
#     H, W = L.shape[2], L.shape[3]
    
#     rgb_images = []
    
#     for i in range(batch_size):
#         # Reconstruir LAB
#         lab_img = np.zeros((H, W, 3))
#         lab_img[:, :, 0] = L[i, 0].cpu().numpy()
#         lab_img[:, :, 1:] = ab[i].cpu().numpy().transpose(1, 2, 0)
        
#         # Convertir a RGB
#         rgb_img = lab2rgb(lab_img)
#         rgb_images.append(rgb_img)
    
#     return np.array(rgb_images)