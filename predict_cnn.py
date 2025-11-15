import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

from autoencoderCNN import AutoencoderCNN

def load_model(model_path='./cifar_autoencoder.pth'):
    model = AutoencoderCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Modo evaluación
    print(f"Modelo cargado desde: {model_path}")
    return model

def unnormalize(img):
    return img / 2 + 0.5

def show_reconstructions(model, testloader, n_images=8):
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    
    # Hacer predicciones
    with torch.no_grad():
        reconstructed = model(images)
    
    fig, axes = plt.subplots(2, n_images, figsize=(16, 4))
    
    for i in range(min(n_images, len(images))):
        orig_img = unnormalize(images[i]).numpy()
        axes[0, i].imshow(np.transpose(orig_img, (1, 2, 0)))
        axes[0, i].set_title(f'Original\n{classes[labels[i]]}')
        axes[0, i].axis('off')
        
        recon_img = unnormalize(reconstructed[i]).numpy()
        axes[1, i].imshow(np.transpose(recon_img, (1, 2, 0)))
        axes[1, i].set_title('Reconstruida')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('reconstructions.png', dpi=150, bbox_inches='tight')
    print("Visualización guardada en: reconstructions.png")
    plt.show()

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
    print(f"Pérdida promedio en test: {avg_loss:.6f}")
    return avg_loss

def predict_single_image(model, image_tensor):
    with torch.no_grad():
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)  # Añadir dimensión batch
        reconstructed = model(image_tensor)
    return reconstructed.squeeze(0)

if __name__ == '__main__':
    print("=" * 60)
    print("PREDICCIÓN CON AUTOENCODER")
    print("=" * 60)
    
    # Cargar modelo
    model = load_model('./cifar_autoencoder.pth')
    
    # Preparar datos de test
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, 
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=8, 
                                             shuffle=True, num_workers=0)
    
    # 1. Mostrar reconstrucciones
    print("\n1. Generando visualizaciones...")
    show_reconstructions(model, testloader, n_images=8)
    
    # 2. Calcular pérdida en test
    print("\n2. Calculando pérdida en conjunto de test...")
    calculate_test_loss(model, testloader)
    
    # 3. Ejemplo de predicción individual
    print("\n3. Ejemplo de predicción individual...")
    dataiter = iter(testloader)
    images, _ = next(dataiter)
    single_image = images[0]
    
    reconstructed = predict_single_image(model, single_image)
    print(f"   Entrada shape: {single_image.shape}")
    print(f"   Salida shape: {reconstructed.shape}")
    
    print("\n" + "=" * 60)
    print("Predicción completada!")
    print("=" * 60)