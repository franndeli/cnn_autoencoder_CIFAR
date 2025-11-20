import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from autoencoderCAE_colourisation import AutoencoderCAE, prepare_dataloaders, lab_to_rgb

# Exercise 1
# from autoencoderCAE import prepare_dataloaders

def load_model(model_path):
    model = AutoencoderCAE()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"✓ Model loaded from {model_path}")
    return model


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
        axes[0, i].set_title('Grayscale Input')
        axes[0, i].axis('off')

        true_img = rgb_true_batch[i]                    
        axes[1, i].imshow(true_img)
        axes[1, i].set_title('Ground Truth')
        axes[1, i].axis('off')

        pred_img = rgb_pred_batch[i]
        pred_img = np.clip(pred_img, 0, 1)
        axes[2, i].imshow(pred_img)
        axes[2, i].set_title('Colorized')
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_name}")
    plt.show()

def calculate_test_loss(model, testloader):
    criterion = nn.MSELoss()
    test_loss = 0.0
    
    with torch.no_grad():
        for grayscale, rgb_target in testloader:
            rgb_predicted = model(grayscale)
            loss = criterion(rgb_predicted, rgb_target)
            test_loss += loss.item()
    
    avg_loss = test_loss / len(testloader)
    print(f"Test loss (MSE on RGB): {avg_loss:.6f}")
    return avg_loss


if __name__ == '__main__':
    print("=" * 60)
    print("COLORIZATION RESULTS (RGB Direct)")
    print("=" * 60)
    
    _, _, testloader = prepare_dataloaders()
    
    model = load_model('cifar_model_colorization.pth')

    show_colorization_results(model, testloader, n_images=8, 
                            save_name='colorization_results.png')
    

    test_loss = calculate_test_loss(model, testloader)
    
    print("\n" + "=" * 60)
    print("✓ Evaluation complete!")
    print("=" * 60)