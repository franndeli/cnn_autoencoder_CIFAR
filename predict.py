import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


# Exercise 1: Basic RGB Autoencoder (Reconstruction)
EXERCISE = 1

# Exercise 2: Architecture Experiments (uses same code as Exercise 1)
# EXERCISE = 2

# Exercise 3: Colorization (Grayscale to Color)
# EXERCISE = 3

if EXERCISE in [1, 2]:
    from autoencoderCAE import AutoencoderCAE, prepare_dataloaders
    MODEL_LOAD_PATH = 'cifar_model_reconstruction.pth'
    RESULTS_SAVE_PATH = 'reconstruction_results.png'
    print("=" * 60)
    print("EVALUATING RGB RECONSTRUCTION MODEL (Exercise 1/2)")
    print("=" * 60)
    
elif EXERCISE == 3:
    from autoencoderCAE_colourisation import AutoencoderCAE, prepare_dataloaders, lab_to_rgb
    MODEL_LOAD_PATH = 'cifar_model_colorization.pth'
    RESULTS_SAVE_PATH = 'colorization_results.png'
    print("=" * 60)
    print("EVALUATING COLORIZATION MODEL (Exercise 3)")
    print("=" * 60)

else:
    raise ValueError("EXERCISE must be 1, 2, or 3")

def load_model(model_path):
    model = AutoencoderCAE()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    print(f"✓ Model loaded from {model_path}")
    return model


def show_reconstruction_results(model, testloader, n_images=8, save_name='reconstruction_results.png'):
    images, labels = next(iter(testloader))
    
    with torch.no_grad():
        reconstructed = model(images)
    
    # CIFAR-10 class names
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    fig, axes = plt.subplots(2, n_images, figsize=(16, 4))
    
    for i in range(min(n_images, len(images))):
        orig_img = images[i].cpu().numpy().transpose(1, 2, 0)
        axes[0, i].imshow(orig_img)
        axes[0, i].set_title(f'Original\n{classes[labels[i]]}', fontsize=10)
        axes[0, i].axis('off')
        
        recon_img = reconstructed[i].cpu().numpy().transpose(1, 2, 0)
        recon_img = np.clip(recon_img, 0, 1)
        axes[1, i].imshow(recon_img)
        axes[1, i].set_title('Reconstructed', fontsize=10)
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_name}")
    plt.show()


def show_colorization_results(model, testloader, n_images=8, save_name='colorization_results.png'):
    grayscale_batch, ab_target_batch = next(iter(testloader))

    with torch.no_grad():
        ab_predicted_batch = model(grayscale_batch)

    rgb_true_batch = lab_to_rgb(grayscale_batch, ab_target_batch)
    rgb_pred_batch = lab_to_rgb(grayscale_batch, ab_predicted_batch)

    fig, axes = plt.subplots(3, n_images, figsize=(16, 6))

    for i in range(min(n_images, len(grayscale_batch))):
        gray_img = grayscale_batch[i, 0].cpu().numpy()
        axes[0, i].imshow(gray_img, cmap='gray')
        axes[0, i].set_title('Grayscale Input', fontsize=10)
        axes[0, i].axis('off')

        true_img = rgb_true_batch[i]                    
        axes[1, i].imshow(true_img)
        axes[1, i].set_title('Ground Truth', fontsize=10)
        axes[1, i].axis('off')

        pred_img = rgb_pred_batch[i]
        pred_img = np.clip(pred_img, 0, 1)
        axes[2, i].imshow(pred_img)
        axes[2, i].set_title('Colorized', fontsize=10)
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_name}")
    plt.show()


def calculate_test_loss(model, testloader, exercise):
    criterion = nn.MSELoss()
    test_loss = 0.0
    
    with torch.no_grad():
        for batch_data in testloader:
            if exercise in [1, 2]:
                images, _ = batch_data
                reconstructed = model(images)
                loss = criterion(reconstructed, images)
            
            elif exercise == 3:
                grayscale, ab_target = batch_data
                ab_predicted = model(grayscale)
                loss = criterion(ab_predicted, ab_target)
            
            test_loss += loss.item()
    
    avg_loss = test_loss / len(testloader)
    
    if exercise in [1, 2]:
        print(f"Test loss (MSE on RGB): {avg_loss:.6f}")
    elif exercise == 3:
        print(f"Test loss (MSE on ab channels): {avg_loss:.6f}")
    
    return avg_loss

if __name__ == '__main__':
    _, _, testloader = prepare_dataloaders()
    
    model = load_model(MODEL_LOAD_PATH)
    
    if EXERCISE in [1, 2]:
        show_reconstruction_results(model, testloader, n_images=8, 
                                   save_name=RESULTS_SAVE_PATH)
    elif EXERCISE == 3:
        show_colorization_results(model, testloader, n_images=8, 
                                 save_name=RESULTS_SAVE_PATH)
    
    test_loss = calculate_test_loss(model, testloader, EXERCISE)
    
    print("\n" + "=" * 60)
    print("✓ Evaluation complete!")
    print("=" * 60)