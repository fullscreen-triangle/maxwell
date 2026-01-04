"""
Thought Library NPZ Analysis: Multi-Thought Comparison
4-panel visualization comparing multiple thoughts
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 9,
    'axes.linewidth': 0.8,
    'figure.dpi': 300,
})

def load_thought_library(npz_file):
    """Load NPZ thought library"""
    print(f"Loading: {npz_file}")
    data = np.load(npz_file)
    
    thoughts = {}
    thought_indices = set()
    
    for key in data.files:
        if 'thought_' in key:
            idx = int(key.split('_')[1])
            thought_indices.add(idx)
    
    for idx in sorted(thought_indices):
        thought = {
            'o2_positions': data[f'thought_{idx}_o2_positions'],
            'hole_center': data[f'thought_{idx}_hole_center'],
            'electron_position': data[f'thought_{idx}_electron_position'],
            'signature': data[f'thought_{idx}_signature'],
            'energy': data[f'thought_{idx}_energy'],
        }
        thoughts[idx] = thought
    
    print(f"  Loaded {len(thoughts)} thoughts")
    return thoughts

def panel_a_energy_comparison(ax, thoughts):
    """Panel A: Energy comparison across thoughts"""
    indices = sorted(thoughts.keys())
    energies = [thoughts[i]['energy'] for i in indices]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(indices)))
    bars = ax.bar(indices, energies, color=colors, alpha=0.7,
                 edgecolor='black', linewidth=1.5)
    
    # Highlight min energy
    min_idx = indices[np.argmin(energies)]
    bars[min_idx].set_color('#e74c3c')
    bars[min_idx].set_linewidth(2.5)
    
    ax.set_xlabel('Thought Index', fontsize=10, fontweight='bold')
    ax.set_ylabel('Energy', fontsize=10, fontweight='bold')
    ax.set_title('A. Energy Comparison', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Statistics
    info_text = f'Min: {np.min(energies):.3e}\nMax: {np.max(energies):.3e}\nMean: {np.mean(energies):.3e}'
    ax.text(0.97, 0.97, info_text, transform=ax.transAxes,
           fontsize=8, va='top', ha='right',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8))
    
    return ax

def panel_b_signature_heatmap(ax, thoughts):
    """Panel B: Signature heatmap across thoughts"""
    indices = sorted(thoughts.keys())
    signatures = np.array([thoughts[i]['signature'] for i in indices])
    
    # Plot heatmap
    im = ax.imshow(signatures, cmap='RdBu_r', aspect='auto', alpha=0.8)
    
    ax.set_xlabel('Signature Component', fontsize=10, fontweight='bold')
    ax.set_ylabel('Thought Index', fontsize=10, fontweight='bold')
    ax.set_title('B. Signature Heatmap', fontsize=11, fontweight='bold')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels(indices)
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Value', fontsize=8)
    
    return ax

def panel_c_signature_pca(ax, thoughts):
    """Panel C: PCA of signatures"""
    indices = sorted(thoughts.keys())
    signatures = np.array([thoughts[i]['signature'] for i in indices])
    energies = np.array([thoughts[i]['energy'] for i in indices])
    
    # PCA
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(signatures)
    
    # Normalize energies for coloring
    energies_norm = (energies - energies.min()) / (energies.max() - energies.min())
    
    scatter = ax.scatter(coords_2d[:, 0], coords_2d[:, 1],
                        c=energies, cmap='plasma', s=200, alpha=0.7,
                        edgecolors='black', linewidth=2)
    
    # Annotate points
    for i, idx in enumerate(indices):
        ax.annotate(f'{idx}', xy=(coords_2d[i, 0], coords_2d[i, 1]),
                   fontsize=9, fontweight='bold', ha='center', va='center')
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', 
                 fontsize=10, fontweight='bold')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', 
                 fontsize=10, fontweight='bold')
    ax.set_title('C. Signature Space (PCA)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.5, alpha=0.5)
    ax.axvline(0, color='black', linewidth=0.5, alpha=0.5)
    
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Energy', fontsize=8)
    
    # Variance explained
    info_text = f'Total variance:\n{pca.explained_variance_ratio_.sum()*100:.1f}%'
    ax.text(0.03, 0.97, info_text, transform=ax.transAxes,
           fontsize=8, va='top', ha='left',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    return ax

def panel_d_spatial_statistics(ax, thoughts):
    """Panel D: Spatial statistics comparison"""
    indices = sorted(thoughts.keys())
    
    # Calculate statistics for each thought
    mean_dists = []
    std_dists = []
    n_molecules = []
    
    for idx in indices:
        o2_pos = thoughts[idx]['o2_positions']
        hole_center = thoughts[idx]['hole_center']
        
        distances = np.linalg.norm(o2_pos - hole_center, axis=1)
        mean_dists.append(distances.mean())
        std_dists.append(distances.std())
        n_molecules.append(len(o2_pos))
    
    # Plot
    x = np.arange(len(indices))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, mean_dists, width, label='Mean distance',
                  color='#3498db', alpha=0.7, edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, std_dists, width, label='Std distance',
                  color='#e74c3c', alpha=0.7, edgecolor='black', linewidth=1.2)
    
    ax.set_xlabel('Thought Index', fontsize=10, fontweight='bold')
    ax.set_ylabel('Distance (Å)', fontsize=10, fontweight='bold')
    ax.set_title('D. Spatial Statistics', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(indices)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Info box
    info_text = f'All thoughts:\nN = {n_molecules[0]} O₂'
    ax.text(0.97, 0.97, info_text, transform=ax.transAxes,
           fontsize=8, va='top', ha='right',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
    
    return ax

def main():
    """Main function for multi-thought comparison"""
    
    # Load thought library
    npz_file = 'thought_library_20251026_051925.npz'
    thoughts = load_thought_library(npz_file)
    
    print(f"\n{'='*70}")
    print(f"COMPARING {len(thoughts)} THOUGHTS")
    print(f"{'='*70}\n")
    
    # Create figure
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.35)
    
    # Panel A: Energy comparison
    ax_a = fig.add_subplot(gs[0, 0])
    panel_a_energy_comparison(ax_a, thoughts)
    
    # Panel B: Signature heatmap
    ax_b = fig.add_subplot(gs[0, 1])
    panel_b_signature_heatmap(ax_b, thoughts)
    
    # Panel C: Signature PCA
    ax_c = fig.add_subplot(gs[1, 0])
    panel_c_signature_pca(ax_c, thoughts)
    
    # Panel D: Spatial statistics
    ax_d = fig.add_subplot(gs[1, 1])
    panel_d_spatial_statistics(ax_d, thoughts)
    
    # Overall title
    fig.suptitle(f'Thought Library Comparison: {len(thoughts)} Thoughts', 
                fontsize=13, fontweight='bold', y=0.98)
    
    # Save
    plt.savefig('thought_comparison_analysis.pdf', bbox_inches='tight')
    plt.savefig('thought_comparison_analysis.png', bbox_inches='tight', dpi=300)
    print("\n✓ Saved: thought_comparison_analysis.pdf/.png")
    
    plt.show()
    
    print(f"\n{'='*70}")
    print("MULTI-THOUGHT COMPARISON COMPLETE")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
