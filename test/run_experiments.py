import torch
import torch.nn as nn
import torch.optim as optim
import json
from datetime import datetime

from test_autoencoderCNN import FlexibleAutoencoder
from autoencoderCNN import prepare_dataloaders
from constants import NUM_EPOCHS, LEARNING_RATE
from experiments import ALL_EXPERIMENTS

def train_model(model, trainloader, validloader, num_epochs=10):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for data in trainloader:
            inputs, _ = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(trainloader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in validloader:
                images, _ = data
                outputs = model(images)
                loss = criterion(outputs, images)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(validloader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
        
        print(f'  Epoch {epoch+1}/{num_epochs}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}')
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1]
    }

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    print("=" * 70)
    print("RUNNING ARCHITECTURE EXPERIMENTS")
    print("=" * 70)
    
    trainloader, validloader, testloader = prepare_dataloaders()
    
    results = []
    
    for exp_config in ALL_EXPERIMENTS:
        print(f"\n{'='*70}")
        print(f"Experiment: {exp_config['name']}")
        print(f"{'='*70}")
        
        model = FlexibleAutoencoder(exp_config)
        latent_size = model.calculate_latent_size()
        num_params = count_parameters(model)
        
        print(f"Latent space size: {latent_size}")
        print(f"Number of parameters: {num_params:,}")
        print(f"Compression ratio: {(32*32*3)/latent_size:.2f}:1")
        
        metrics = train_model(model, trainloader, validloader, NUM_EPOCHS)
        
        result = {
            'name': exp_config['name'],
            'latent_size': latent_size,
            'num_parameters': num_params,
            'compression_ratio': (32*32*3)/latent_size,
            **metrics
        }
        results.append(result)
        
        torch.save(model.state_dict(), f"models/{exp_config['name']}.pth")
        print(f"✓ Model saved: {exp_config['name']}.pth")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'experiment_results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETED")
    print(f"Results saved to: experiment_results_{timestamp}.json")
    print("="*70)
    
    print("\nSUMMARY TABLE:")
    print(f"{'Experiment':<25} {'Latent':<10} {'Best Val Loss':<15} {'Params':<12}")
    print("-"*70)
    for r in results:
        print(f"{r['name']:<25} {r['latent_size']:<10} {r['best_val_loss']:<15.6f} {r['num_parameters']:<12,}")