import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from test.test_autoencoderCNN import FlexibleAutoencoder
from autoencoderCNN import prepare_dataloaders
from test.experiments import EXP_1C_TINY_LATENT # ! CHANGE

def load_flexible_model(model_path, config):
    model = FlexibleAutoencoder(config)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"Model loaded: {config['name']}")
    print(f"Latent size: {model.calculate_latent_size()}")
    return model

def show_reconstructions(model, testloader, n_images=8, save_name='reconstructions.png'):
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    
    with torch.no_grad():
        reconstructed = model(images)
    
    fig, axes = plt.subplots(2, n_images, figsize=(16, 4))
    
    for i in range(min(n_images, len(images))):
        orig_img = images[i].numpy()
        axes[0, i].imshow(np.transpose(orig_img, (1, 2, 0)))
        axes[0, i].set_title(f'Original\n{classes[labels[i]]}')
        axes[0, i].axis('off')
        
        recon_img = reconstructed[i].numpy()
        axes[1, i].imshow(np.transpose(recon_img, (1, 2, 0)))
        axes[1, i].set_title('Reconstructed')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_name}")

def calculate_test_loss(model, testloader):
    criterion = nn.MSELoss()
    test_loss = 0.0
    
    with torch.no_grad():
        for data in testloader:
            images, _ = data
            outputs = model(images)
            loss = criterion(outputs, images)
            test_loss += loss.item()
    
    avg_loss = test_loss / len(testloader)
    print(f"Test loss: {avg_loss:.6f}")
    return avg_loss

if __name__ == '__main__':
    print("=" * 60)
    print("COMPARING EXPERIMENT RESULTS")
    print("=" * 60)
    
    _, _, testloader = prepare_dataloaders()
    
    model_1a = load_flexible_model(
        'models/exp_1c_tiny_latent.pth', # ! CHANGE
        EXP_1C_TINY_LATENT # ! CHANGE
    )
    
    show_reconstructions(model_1a, testloader, n_images=8, 
                        save_name='reconstructions_1b_tiny.png') # ! CHANGE
    loss_1a = calculate_test_loss(model_1a, testloader)
    
    print("\n" + "=" * 60)