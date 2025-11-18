import json
import matplotlib.pyplot as plt
import numpy as np
import os

def load_results(filename='experiment_results_20251117_194802.json'): # ! CHANGE
    with open(filename, 'r') as f:
        return json.load(f)

def plot_latent_vs_loss(results):
    latent_experiments = [r for r in results if 'exp_1' in r['name'] or r['name'] == 'baseline']
    
    latent_sizes = [r['latent_size'] for r in latent_experiments]
    best_val_losses = [r['best_val_loss'] for r in latent_experiments]
    names = [r['name'] for r in latent_experiments]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scatter = ax.scatter(latent_sizes, best_val_losses, s=200, alpha=0.6, c=latent_sizes, 
                        cmap='viridis', edgecolors='black', linewidth=2)
    
    z = np.polyfit(latent_sizes, best_val_losses, 2)
    p = np.poly1d(z)
    x_smooth = np.linspace(min(latent_sizes), max(latent_sizes), 100)
    ax.plot(x_smooth, p(x_smooth), "r--", alpha=0.8, linewidth=2, label='Trend')
    
    for i, name in enumerate(names):
        ax.annotate(f'{latent_sizes[i]}', 
                   (latent_sizes[i], best_val_losses[i]),
                   textcoords="offset points", xytext=(0,10), 
                   ha='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Latent Space Size', fontsize=13, fontweight='bold')
    ax.set_ylabel('Best Validation Loss', fontsize=13, fontweight='bold')
    ax.set_title('Impact of Latent Space Size on Reconstruction Quality', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig('plot_1_latent_vs_loss.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: plot_1_latent_vs_loss.png")
    plt.show()

def plot_compression_tradeoff(results):
    latent_experiments = [r for r in results if 'exp_1' in r['name'] or r['name'] == 'baseline']
    
    compression_ratios = [r['compression_ratio'] for r in latent_experiments]
    best_val_losses = [r['best_val_loss'] for r in latent_experiments]
    names = [r['name'].replace('exp_1', '').replace('_', ' ').title() 
             for r in latent_experiments]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    cmap = plt.cm.get_cmap('RdYlGn_r')
    colors = cmap(np.linspace(0.2, 0.8, len(compression_ratios)))
    bars = ax.barh(names, best_val_losses, color=colors, edgecolor='black', linewidth=1.5)
    
    for i, (bar, ratio, loss) in enumerate(zip(bars, compression_ratios, best_val_losses)):
        width = bar.get_width()
        ax.text(width + 0.0002, bar.get_y() + bar.get_height()/2, 
               f'{loss:.6f}\n({ratio:.1f}:1)', 
               ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Best Validation Loss', fontsize=13, fontweight='bold')
    ax.set_title('Compression vs Quality Trade-off', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plot_2_compression_tradeoff.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: plot_2_compression_tradeoff.png")
    plt.show()

def plot_training_curves(results):
    key_experiments = ['baseline', 'exp_1c_tiny_latent', 'exp_1b_large_latent', 
                      'exp_2a_shallow', 'exp_3b_wide']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, exp_name in enumerate(key_experiments):
        exp = next((r for r in results if r['name'] == exp_name), None)
        if exp:
            epochs = range(1, len(exp['train_losses']) + 1)
            
            ax1.plot(epochs, exp['train_losses'], marker='o', color=colors[i], 
                    linewidth=2, label=exp_name.replace('_', ' ').title())
            
            ax2.plot(epochs, exp['val_losses'], marker='s', color=colors[i], 
                    linewidth=2, label=exp_name.replace('_', ' ').title())
    
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Training Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training Loss Evolution', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Validation Loss Evolution', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plot_3_training_curves.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: plot_3_training_curves.png")
    plt.show()

def plot_parameters_vs_performance(results):
    fig, ax = plt.subplots(figsize=(12, 7))
    
    params = [r['num_parameters'] for r in results]
    losses = [r['best_val_loss'] for r in results]
    names = [r['name'].replace('exp_', '').replace('_', '\n') for r in results]
    latent_sizes = [r['latent_size'] for r in results]
    
    sizes = [l/5 for l in latent_sizes]
    
    scatter = ax.scatter(params, losses, s=sizes, alpha=0.6, 
                        c=latent_sizes, cmap='plasma', 
                        edgecolors='black', linewidth=2)
    
    for i, name in enumerate(names):
        ax.annotate(name, (params[i], losses[i]),
                   textcoords="offset points", xytext=(5,5), 
                   ha='left', fontsize=8)
    
    ax.set_xlabel('Number of Parameters', fontsize=13, fontweight='bold')
    ax.set_ylabel('Best Validation Loss', fontsize=13, fontweight='bold')
    ax.set_title('Model Complexity vs Performance', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Latent Size', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('plot_4_parameters_vs_performance.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: plot_4_parameters_vs_performance.png")
    plt.show()

def plot_experiment_categories(results):
    categories = {
        'Latent Size': ['exp_1c_tiny_latent', 'exp_1a_small_latent', 'baseline', 'exp_1b_large_latent'],
        'Depth': ['exp_2a_shallow', 'baseline', 'exp_2b_deeper'],
        'Width': ['exp_3a_narrow', 'baseline', 'exp_3b_wide'],
        'Kernel': ['baseline', 'exp_4a_large_kernels']
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    for idx, (category, exp_names) in enumerate(categories.items()):
        ax = axes[idx]
        
        cat_results = [r for r in results if r['name'] in exp_names]
        cat_results = sorted(cat_results, key=lambda x: x['latent_size'])
        
        names = [r['name'].replace('exp_', '').replace('_', '\n') for r in cat_results]
        train_losses = [r['final_train_loss'] for r in cat_results]
        val_losses = [r['final_val_loss'] for r in cat_results]
        
        x = np.arange(len(names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, train_losses, width, label='Train Loss', 
                      color='steelblue', edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x + width/2, val_losses, width, label='Val Loss', 
                      color='coral', edgecolor='black', linewidth=1.5)
        
        ax.set_ylabel('Loss', fontsize=11, fontweight='bold')
        ax.set_title(f'{category} Experiments', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plot_5_experiment_categories.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: plot_5_experiment_categories.png")
    plt.show()

def plot_overfitting_analysis(results):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    names = [r['name'].replace('exp_', '').replace('_', ' ').title() for r in results]
    train_losses = [r['final_train_loss'] for r in results]
    val_losses = [r['final_val_loss'] for r in results]
    gaps = [val - train for val, train in zip(val_losses, train_losses)]
    
    sorted_indices = np.argsort(gaps)
    names = [names[i] for i in sorted_indices]
    gaps = [gaps[i] for i in sorted_indices]
    
    colors = ['green' if g < 0.0005 else 'orange' if g < 0.001 else 'red' for g in gaps]
    
    bars = ax.barh(names, gaps, color=colors, edgecolor='black', linewidth=1.5)
    
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    
    ax.set_xlabel('Overfitting Gap (Val Loss - Train Loss)', fontsize=12, fontweight='bold')
    ax.set_title('Overfitting Analysis: Generalization Gap', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plot_6_overfitting_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: plot_6_overfitting_analysis.png")
    plt.show()

def create_summary_table(results):
    print("\n" + "=" * 100)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 100)
    print(f"{'Experiment':<25} {'Latent':<10} {'Params':<10} {'Best Val':<12} {'Final Train':<12} {'Gap':<10}")
    print("-" * 100)
    
    sorted_results = sorted(results, key=lambda x: x['best_val_loss'])
    
    for r in sorted_results:
        gap = r['final_val_loss'] - r['final_train_loss']
        print(f"{r['name']:<25} {r['latent_size']:<10} {r['num_parameters']:<10} "
              f"{r['best_val_loss']:<12.6f} {r['final_train_loss']:<12.6f} {gap:<10.6f}")
    
    print("=" * 100)
    
    print("\n🏆 BEST MODELS:")
    print(f"  Lowest Loss:        {sorted_results[0]['name']} (Loss: {sorted_results[0]['best_val_loss']:.6f})")
    
    best_compression = max(results, key=lambda x: x['compression_ratio'])
    print(f"  Best Compression:   {best_compression['name']} ({best_compression['compression_ratio']:.1f}:1)")
    
    smallest_gap = min(results, key=lambda x: abs(x['final_val_loss'] - x['final_train_loss']))
    gap_val = abs(smallest_gap['final_val_loss'] - smallest_gap['final_train_loss'])
    print(f"  Best Generalization: {smallest_gap['name']} (Gap: {gap_val:.6f})")
    
    print("=" * 100)

if __name__ == '__main__':
    print("Current working directory:", os.getcwd())
    print("=" * 70)
    print("ANALYZING EXPERIMENT RESULTS")
    print("=" * 70)
    
    results = load_results("./test/experiment_results_20251117_194802.json") # ! CHANGE
    
    print("\nGenerating plots...\n")
    
    plot_latent_vs_loss(results)
    plot_compression_tradeoff(results)
    plot_training_curves(results)
    plot_parameters_vs_performance(results)
    plot_experiment_categories(results)
    plot_overfitting_analysis(results)
    
    create_summary_table(results)
    
    print("\n" + "=" * 70)
    print("✓ Analysis complete! All plots saved.")
    print("=" * 70)