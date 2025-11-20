import torch
import matplotlib.pyplot as plt
import numpy as np

from test.test_autoencoderCAE import FlexibleAutoencoder
from autoencoderCAE import prepare_dataloaders
from test.experiments import (
    BASELINE,
    EXP_1A_SMALL_LATENT,
    EXP_1B_LARGE_LATENT,
    EXP_1C_TINY_LATENT,
    EXP_2A_SHALLOW,
    EXP_2B_DEEPER,
    EXP_3A_NARROW,
    EXP_3B_WIDE,
    EXP_4A_LARGE_KERNELS,
)

def load_model(model_path, config):
    try:
        model = FlexibleAutoencoder(config)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    except Exception as e:
        print(f"⚠️  Failed to load {config['name']}: {e}")
        return None

def reconstruct_image(model, image):
    with torch.no_grad():
        if image.dim() == 3:
            image = image.unsqueeze(0)
        reconstructed = model(image)
        return reconstructed.squeeze(0)

def compare_all_models(image, label, experiments_list): 
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    models = []
    names = []
    
    print("\nLoading models...")
    for model_path, config, display_name in experiments_list:
        model = load_model(model_path, config)
        if model is not None:
            models.append(model)
            names.append(display_name)
            latent_size = model.calculate_latent_size()
            print(f"  ✓ {display_name:<25} (latent: {latent_size})")
    
    if len(models) == 0:
        print("❌ No models loaded successfully!")
        return
    
    n_models = len(models) + 1 
    fig, axes = plt.subplots(1, n_models, figsize=(3*n_models, 4))
    
    img_np = image.numpy()
    img_display = np.transpose(img_np, (1, 2, 0))
    
    axes[0].imshow(img_display)
    axes[0].set_title('Original\n' + classes[label], 
                     fontsize=12, fontweight='bold', color='blue')
    axes[0].axis('off')
    
    for idx, (model, name) in enumerate(zip(models, names), start=1):
        reconstructed = reconstruct_image(model, image)
        recon_np = reconstructed.numpy()
        recon_display = np.transpose(recon_np, (1, 2, 0))
        
        axes[idx].imshow(recon_display)
        axes[idx].set_title(name, fontsize=11)
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'single_image_comparison_{classes[label]}.png', 
                dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: single_image_comparison_{classes[label]}.png")
    plt.show()

def compare_multiple_images(images, labels, experiments_list, n_images=3):
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    models = []
    names = []
    
    print("\nLoading models...")
    for model_path, config, display_name in experiments_list:
        model = load_model(model_path, config)
        if model is not None:
            models.append(model)
            names.append(display_name)
            print(f"  ✓ {display_name}")
    
    if len(models) == 0:
        print("❌ No models loaded!")
        return
    
    n_models = len(models) + 1
    fig, axes = plt.subplots(n_images, n_models, 
                            figsize=(3*n_models, 3*n_images))
    
    for row in range(n_images):
        image = images[row]
        label = labels[row]
        
        img_np = image.numpy()
        img_display = np.transpose(img_np, (1, 2, 0))
        
        axes[row, 0].imshow(img_display)
        if row == 0:
            axes[row, 0].set_title('Original', fontsize=12, fontweight='bold')
        axes[row, 0].set_ylabel(f'{classes[label]}', 
                               fontsize=11, fontweight='bold')
        axes[row, 0].axis('off')
        
        for col, (model, name) in enumerate(zip(models, names), start=1):
            reconstructed = reconstruct_image(model, image)
            recon_np = reconstructed.numpy()
            recon_display = np.transpose(recon_np, (1, 2, 0))
            
            axes[row, col].imshow(recon_display)
            if row == 0:
                axes[row, col].set_title(name, fontsize=11)
            axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('multiple_images_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: multiple_images_comparison.png")
    plt.show()

if __name__ == '__main__':
    print("=" * 70)
    print("SINGLE IMAGE RECONSTRUCTION COMPARISON")
    print("=" * 70)
    
    _, _, testloader = prepare_dataloaders()
    
    experiments = [
        ('models/exp_1c_tiny_latent.pth', EXP_1C_TINY_LATENT, 'Tiny\n(256)'),
        ('models/exp_1a_small_latent.pth', EXP_1A_SMALL_LATENT, 'Small\n(512)'),
        ('models/baseline.pth', BASELINE, 'Baseline\n(1024)'),
        ('models/exp_1b_large_latent.pth', EXP_1B_LARGE_LATENT, 'Large\n(2048)'),
        ('models/exp_2a_shallow.pth', EXP_2A_SHALLOW, 'Shallow'),
        ('models/exp_2b_deeper.pth', EXP_2B_DEEPER, 'Deeper'),
        ('models/exp_3a_narrow.pth', EXP_3A_NARROW, 'Narrow'),
        ('models/exp_3b_wide.pth', EXP_3B_WIDE, 'Wide'),
        ('models/exp_4a_large_kernels.pth', EXP_4A_LARGE_KERNELS, 'Large Kernels'),
    ]
    
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    
    print("\n--- Option 1: Single Image Comparison ---")
    single_image = images[0]
    single_label = labels[0]
    compare_all_models(single_image, single_label, experiments)
    
    print("\n--- Option 2: Multiple Images Comparison ---")
    compare_multiple_images(images[:3], labels[:3], experiments, n_images=3)
    
    print("\n" + "=" * 70)
    print("✓ Comparison complete!")
    print("=" * 70)