import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from autoencoderCNN import AutoencoderCNN

transform = transforms.Compose([
    transforms.ToTensor(),
])

batch_size = 4
num_epochs = 10

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    
if __name__ == '__main__':
    model = AutoencoderCNN()
    print(model)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/2')

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, _  = data

            optimizer.zero_grad()

            # forward + backward
            outputs = model.forward(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for data in testloader:
                images, _ = data
                outputs = model(images)
                loss = criterion(outputs, images)
                test_loss += loss.item()
        
        avg_test_loss = test_loss / len(testloader)
        print(f'Epoch {epoch + 1}/{num_epochs} completed - Test Loss: {avg_test_loss:.4f}')
        print("-" * 60)

    print("=" * 60)
    print('¡Finished training!')

    # Guardar el modelo
    PATH = './cifar_autoencoder.pth'
    torch.save(model.state_dict(), PATH)
    print(f'Modelo guardado en: {PATH}')