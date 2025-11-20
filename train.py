import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from constants import NUM_EPOCHS, LEARNING_RATE


# Exercise 1: Basic RGB Autoencoder (Reconstruction)
EXERCISE = 1

# Exercise 2: Architecture Experiments (uses same code as Exercise 1)
# EXERCISE = 2

# Exercise 3: Colorization (Grayscale to Color)
# EXERCISE = 3

if EXERCISE in [1, 2]:
    from autoencoderCAE import AutoencoderCAE, prepare_dataloaders
    MODEL_SAVE_PATH = 'cifar_model_reconstruction.pth'
    LOSS_PLOT_PATH = 'loss_plot_reconstruction.png'
    print("=" * 60)
    print("TRAINING RGB RECONSTRUCTION MODEL (Exercise 1/2)")
    print("=" * 60)
    
elif EXERCISE == 3:
    from autoencoderCAE_colourisation import AutoencoderCAE, prepare_dataloaders
    MODEL_SAVE_PATH = 'cifar_model_colorization.pth'
    LOSS_PLOT_PATH = 'loss_plot_colorization.png'
    print("=" * 60)
    print("TRAINING COLORIZATION MODEL (Exercise 3)")
    print("=" * 60)

else:
    raise ValueError("EXERCISE must be 1, 2, or 3")


if __name__ == '__main__':
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")

    model = AutoencoderCAE().to(device)
    
    trainloader, validloader, testloader = prepare_dataloaders()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    epochs_list = []

    for epoch in range(NUM_EPOCHS):
        print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}')

        # Train
        model.train()
        train_loss = 0.0

        for batch_data in trainloader:
            if EXERCISE in [1, 2]:
                images, _ = batch_data
                images = images.to(device)
                
                optimizer.zero_grad()
                reconstructed = model(images)
                loss = criterion(reconstructed, images)
                
            elif EXERCISE == 3:
                L, ab_target = batch_data
                L = L.to(device)
                ab_target = ab_target.to(device)

                optimizer.zero_grad()
                ab_predicted = model(L)
                loss = criterion(ab_predicted, ab_target)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(trainloader)

        #Val
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_data in validloader:
                if EXERCISE in [1, 2]:
                    images, _ = batch_data
                    images = images.to(device)
                    reconstructed = model(images)
                    loss = criterion(reconstructed, images)
                    
                elif EXERCISE == 3:
                    L, ab_target = batch_data
                    L = L.to(device)
                    ab_target = ab_target.to(device)
                    ab_predicted = model(L)
                    loss = criterion(ab_predicted, ab_target)
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(validloader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        epochs_list.append(epoch + 1)

        print(f'  Train Loss: {avg_train_loss:.6f}')
        print(f'  Val Loss:   {avg_val_loss:.6f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f'  ✓ Best model saved!')
        
        print("-" * 60)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print("=" * 60)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs_list, train_losses, label='Training Loss', marker='o', linewidth=2)
    plt.plot(epochs_list, val_losses, label='Validation Loss', marker='s', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    
    if EXERCISE in [1, 2]:
        plt.ylabel('Loss (MSE on RGB)', fontsize=12)
        plt.title('RGB Reconstruction Training Progress', fontsize=14, fontweight='bold')
    elif EXERCISE == 3:
        plt.ylabel('Loss (MSE on ab channels)', fontsize=12)
        plt.title('Colorization Training Progress', fontsize=14, fontweight='bold')
    
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(LOSS_PLOT_PATH, dpi=300, bbox_inches='tight')
    print(f"Loss plot saved as '{LOSS_PLOT_PATH}'")