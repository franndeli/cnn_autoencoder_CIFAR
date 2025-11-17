# import json
# import matplotlib.pyplot as plt
# import numpy as np

# def load_results(filename):
#     """Carga resultados de experimentos"""
#     with open(filename, 'r') as f:
#         return json.load(f)

# def plot_latent_vs_loss(results):
#     """Plot: Latent size vs reconstruction error"""
#     latent_sizes = [r['latent_size'] for r in results]
#     val_losses = [r['best_val_loss'] for r in results]
#     names = [r['name'] for r in results]
    
#     plt.figure(figsize=(10, 6))
#     plt.scatter(latent_sizes, val_losses, s=100, alpha=0.6)
    
#     for i, name in enumerate(names):
#         plt.annotate(name, (latent_sizes[i], val_losses[i]), 
#                     fontsize=8, ha='right')
    
#     plt.xlabel('Latent Space Size', fontsize=12)
#     plt.ylabel('Best Validation Loss', fontsize=12)
#     plt.title('Impact of Latent Space Size on Reconstruction Error', 
#               fontsize=14, fontweight='bold')
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig('latent_vs_loss.png', dpi=300)
#     plt.show()

# def plot_compression_vs_loss(results):
#     """Plot: Compression ratio vs loss"""
#     compression = [r['compression_ratio'] for r in results]
#     val_losses = [r['best_val_loss'] for r in results]
    
#     plt.figure(figsize=(10, 6))
#     plt.scatter(compression, val_losses, s=100, alpha=0.6, color='coral')
#     plt.xlabel('Compression Ratio', fontsize=12)
#     plt.ylabel('Best Validation Loss', fontsize=12)
#     plt.title('Trade-off: Compression vs Reconstruction Quality', 
#               fontsize=14, fontweight='bold')
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig('compression_vs_loss.png', dpi=300)
#     plt.show()

# def create_comparison_table(results):
#     """Crea tabla de comparación"""
#     print("\n" + "="*90)
#     print("EXPERIMENT COMPARISON TABLE")
#     print("="*90)
#     print(f"{'Name':<25} {'Latent':<10} {'Compress':<12} {'Val Loss':<12} {'Params':<12}")
#     print("-"*90)
    
#     for r in sorted(results, key=lambda x: x['best_val_loss']):
#         print(f"{r['name']:<25} {r['latent_size']:<10} "
#               f"{r['compression_ratio']:<12.2f} {r['best_val_loss']:<12.6f} "
#               f"{r['num_parameters']:<12,}")

# if __name__ == '__main__':
#     # Cargar resultados (ajusta el nombre del archivo)
#     results = load_results('experiment_results_20251117_194802.json')
    
#     # Generar análisis
#     create_comparison_table(results)
#     plot_latent_vs_loss(results)
#     plot_compression_vs_loss(results)
    
#     print("\n✓ Analysis complete! Check the generated plots.")