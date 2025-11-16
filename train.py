import torch
import torch.nn as nn
import torch.optim as optim

from autoencoderCNN import AutoencoderCNN, prepare_dataloaders

from constants import NUM_EPOCHS, LEARNING_RATE
    
if __name__ == '__main__':
    model = AutoencoderCNN()

    trainset, validset, testloader = prepare_dataloaders()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        print(f'Epoch {epoch}/{NUM_EPOCHS}')

        model.train()
        train_loss = 0.0

        for i, data in enumerate(trainset, 0):
            inputs, _  = data

            optimizer.zero_grad()

            # forward + backward
            outputs = model.forward(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(trainset)
        print(f'  Train Loss: {avg_train_loss:.4f}')

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for data in validset:
                images, _ = data
                outputs = model(images)
                loss = criterion(outputs, images)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(validset)

        print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}]')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Loss:   {avg_val_loss:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), './cifar_model.pth')
            print(f'  ✓ Best model saved!')
        
        print("-" * 60)

    print("\n" + "=" * 60)