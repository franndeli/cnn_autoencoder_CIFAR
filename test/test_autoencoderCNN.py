import torch
import torch.nn as nn
import torch.nn.functional as F

class FlexibleAutoencoder(nn.Module):    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.name = config['name']
        self.arch_type = config.get('type', 'standard')  # standard, shallow, deep
        
        if self.arch_type == 'standard':
            self._build_standard_arch()
        elif self.arch_type == 'shallow':
            self._build_shallow_arch()
        elif self.arch_type == 'deep':
            self._build_deep_arch()
        else:
            raise ValueError(f"Unknown architecture type: {self.arch_type}")
    
    def _build_standard_arch(self):
        cfg = self.config
        
        # Encoder
        self.conv1 = nn.Conv2d(
            cfg['conv1']['in'], cfg['conv1']['out'], 
            cfg['conv1']['kernel'], padding=cfg['conv1']['padding']
        )
        self.pool1 = nn.MaxPool2d(2, stride=2)
        
        self.conv2 = nn.Conv2d(
            cfg['conv2']['in'], cfg['conv2']['out'],
            cfg['conv2']['kernel'], padding=cfg['conv2']['padding']
        )
        self.pool2 = nn.MaxPool2d(2, stride=2)
        
        self.conv3 = nn.Conv2d(
            cfg['conv3']['in'], cfg['conv3']['out'],
            cfg['conv3']['kernel'], padding=cfg['conv3']['padding']
        )
        
        # Decoder
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.conv4 = nn.Conv2d(
            cfg['conv4']['in'], cfg['conv4']['out'],
            cfg['conv4']['kernel'], padding=cfg['conv4']['padding']
        )
        
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.conv5 = nn.Conv2d(
            cfg['conv5']['in'], cfg['conv5']['out'],
            cfg['conv5']['kernel'], padding=cfg['conv5']['padding']
        )
    
    def _build_shallow_arch(self):
        cfg = self.config
        
        # Encoder
        self.conv1 = nn.Conv2d(
            cfg['conv1']['in'], cfg['conv1']['out'],
            cfg['conv1']['kernel'], padding=cfg['conv1']['padding']
        )
        self.pool1 = nn.MaxPool2d(2, stride=2)
        
        self.conv2 = nn.Conv2d(
            cfg['conv2']['in'], cfg['conv2']['out'],
            cfg['conv2']['kernel'], padding=cfg['conv2']['padding']
        )
        
        # Decoder
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.conv3 = nn.Conv2d(
            cfg['conv3']['in'], cfg['conv3']['out'],
            cfg['conv3']['kernel'], padding=cfg['conv3']['padding']
        )
    
    def _build_deep_arch(self):
        cfg = self.config
        
        # Encoder
        self.conv1 = nn.Conv2d(
            cfg['conv1']['in'], cfg['conv1']['out'],
            cfg['conv1']['kernel'], padding=cfg['conv1']['padding']
        )
        self.pool1 = nn.MaxPool2d(2, stride=2)
        
        self.conv2 = nn.Conv2d(
            cfg['conv2']['in'], cfg['conv2']['out'],
            cfg['conv2']['kernel'], padding=cfg['conv2']['padding']
        )
        self.pool2 = nn.MaxPool2d(2, stride=2)
        
        self.conv3 = nn.Conv2d(
            cfg['conv3']['in'], cfg['conv3']['out'],
            cfg['conv3']['kernel'], padding=cfg['conv3']['padding']
        )
        self.pool3 = nn.MaxPool2d(2, stride=2)
        
        self.conv4 = nn.Conv2d(
            cfg['conv4']['in'], cfg['conv4']['out'],
            cfg['conv4']['kernel'], padding=cfg['conv4']['padding']
        )
        
        # Decoder
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.conv5 = nn.Conv2d(
            cfg['conv5']['in'], cfg['conv5']['out'],
            cfg['conv5']['kernel'], padding=cfg['conv5']['padding']
        )
        
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.conv6 = nn.Conv2d(
            cfg['conv6']['in'], cfg['conv6']['out'],
            cfg['conv6']['kernel'], padding=cfg['conv6']['padding']
        )
        
        self.upsample3 = nn.Upsample(scale_factor=2)
        self.conv7 = nn.Conv2d(
            cfg['conv7']['in'], cfg['conv7']['out'],
            cfg['conv7']['kernel'], padding=cfg['conv7']['padding']
        )
    
    def forward(self, x):
        if self.arch_type == 'standard':
            return self._forward_standard(x)
        elif self.arch_type == 'shallow':
            return self._forward_shallow(x)
        elif self.arch_type == 'deep':
            return self._forward_deep(x)
    
    def _forward_standard(self, x):
        # Encoder
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        
        # Decoder
        x = self.upsample1(x)
        x = F.relu(self.conv4(x))
        x = self.upsample2(x)
        x = self.conv5(x)
        x = torch.sigmoid(x)
        
        return x
    
    def _forward_shallow(self, x):
        # Encoder
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        
        # Decoder
        x = self.upsample1(x)
        x = self.conv3(x)
        x = torch.sigmoid(x)
        
        return x
    
    def _forward_deep(self, x):
        # Encoder
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        
        # Decoder
        x = self.upsample1(x)
        x = F.relu(self.conv5(x))
        x = self.upsample2(x)
        x = F.relu(self.conv6(x))
        x = self.upsample3(x)
        x = self.conv7(x)
        x = torch.sigmoid(x)
        
        return x
    
    def calculate_latent_size(self):
        spatial = 0
        if self.arch_type == 'standard':
            # 2 poolings: 32 -> 16 -> 8
            spatial = 8
            channels = self.config['conv3']['out']
        elif self.arch_type == 'shallow':
            # 1 pooling: 32 -> 16
            spatial = 16
            channels = self.config['conv2']['out']
        elif self.arch_type == 'deep':
            # 3 poolings: 32 -> 16 -> 8 -> 4
            spatial = 4
            channels = self.config['conv4']['out']
        
        return spatial * spatial * channels