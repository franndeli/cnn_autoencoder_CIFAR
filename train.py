import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from autoencoderCAE_colourisation import AutoencoderCAE, prepare_dataloaders

# Exercise 1
# from autoencoderCAE import AutoencoderCAE, prepare_dataloaders

from constants import NUM_EPOCHS, LEARNING_RATE

if __name__ == '__main__':
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    model = AutoencoderCAE().to(device)
    
    trainloader, validloader, testloader = prepare_dataloaders()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    epochs_list = []

    print("=" * 60)
    print("TRAINING COLORIZATION MODEL (LAB COLOR SPACE)")
    print("=" * 60)

    for epoch in range(NUM_EPOCHS):
        print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}')

        # Train
        model.train()
        train_loss = 0.0

        for L, ab_target in trainloader:
            L = L.to(device)
            ab_target = ab_target.to(device)

            optimizer.zero_grad()
            ab_predicted = model(L)
            
            loss = criterion(ab_predicted, ab_target)
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(trainloader)

        # Val
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for L, ab_target in validloader:
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
            torch.save(model.state_dict(), './cifar_model_colorization.pth')
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
    plt.ylabel('Loss (MSE on ab channels)', fontsize=12)
    plt.title('Colorization Training Progress', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('loss_plot_colorization.png', dpi=300, bbox_inches='tight')
    print("Loss plot saved as 'loss_plot_colorization.png'")