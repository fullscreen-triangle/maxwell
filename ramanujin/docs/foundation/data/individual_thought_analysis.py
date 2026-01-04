"""
Thought Library NPZ Analysis: Individual Thoughts
4-panel visualization of single thought library structure
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, squareform
from scipy.signal import find_peaks

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
    
    print(f"  Arrays: {list(data.files)}")
    
    # Extract thoughts
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
        
        print(f"\n  Thought {idx}:")
        print(f"    O₂ molecules: {len(thought['o2_positions'])}")
        print(f"    Hole center: {thought['hole_center']}")
        print(f"    Electron: {thought['electron_position']}")
        print(f"    Signature dim: {len(thought['signature'])}")
        print(f"    Energy: {thought['energy']:.6e}")
    
    return thoughts

def panel_a_3d_configuration(ax, thought):
    """Panel A: 3D O₂ configuration with hole and electron"""
    o2_pos = thought['o2_positions']
    hole_center = thought['hole_center']
    electron_pos = thought['electron_position']
    
    # Calculate distances from hole center
    distances = np.linalg.norm(o2_pos - hole_center, axis=1)
    
    # Plot O₂ molecules
    scatter = ax.scatter(o2_pos[:, 0], o2_pos[:, 1], o2_pos[:, 2],
                        c=distances, cmap='viridis', s=40, alpha=0.7,
                        edgecolors='black', linewidth=0.5)
    
    # Plot hole center
    ax.scatter([hole_center[0]], [hole_center[1]], [hole_center[2]],
              c='red', s=300, marker='*', edgecolors='black', linewidth=2,
              label='Hole Center', zorder=10)
    
    # Plot electron
    ax.scatter([electron_pos[0]], [electron_pos[1]], [electron_pos[2]],
              c='blue', s=200, marker='D', edgecolors='black', linewidth=2,
              label='Electron', zorder=10)
    
    ax.set_xlabel('X (Å)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Y (Å)', fontsize=10, fontweight='bold')
    ax.set_zlabel('Z (Å)', fontsize=10, fontweight='bold')
    ax.set_title('A. 3D Configuration', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.view_init(elev=20, azim=45)
    
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.03, pad=0.1, shrink=0.8)
    cbar.set_label('Distance from hole (Å)', fontsize=8)
    
    return ax

def panel_b_radial_distribution(ax, thought):
    """Panel B: Radial distribution from hole center"""
    o2_pos = thought['o2_positions']
    hole_center = thought['hole_center']
    
    distances = np.linalg.norm(o2_pos - hole_center, axis=1)
    
    # Histogram
    counts, bins = np.histogram(distances, bins=20)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    ax.bar(bin_centers, counts, width=bins[1]-bins[0],
          color='#3498db', alpha=0.7, edgecolor='black', linewidth=1.2)
    
    # Statistics
    mean_dist = distances.mean()
    median_dist = np.median(distances)
    
    ax.axvline(mean_dist, color='red', linestyle='--', linewidth=2,
              label=f'Mean: {mean_dist:.2f} Å')
    ax.axvline(median_dist, color='orange', linestyle='--', linewidth=2,
              label=f'Median: {median_dist:.2f} Å')
    
    ax.set_xlabel('Distance from Hole (Å)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Count', fontsize=10, fontweight='bold')
    ax.set_title('B. Radial Distribution', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Info box
    info_text = f'N = {len(o2_pos)}\nRange: {distances.min():.1f}-{distances.max():.1f} Å'
    ax.text(0.97, 0.97, info_text, transform=ax.transAxes,
           fontsize=8, va='top', ha='right',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8))
    
    return ax

def panel_c_signature_spectrum(ax, thought):
    """Panel C: Oscillatory signature spectrum"""
    signature = thought['signature']
    
    # Plot signature as bar chart
    indices = np.arange(len(signature))
    colors = plt.cm.viridis(np.linspace(0, 1, len(signature)))
    
    bars = ax.bar(indices, signature, color=colors, alpha=0.7,
                 edgecolor='black', linewidth=0.8)
    
    # Highlight max and min
    max_idx = np.argmax(signature)
    min_idx = np.argmin(signature)
    
    bars[max_idx].set_color('#e74c3c')
    bars[max_idx].set_linewidth(2)
    bars[min_idx].set_color('#2ecc71')
    bars[min_idx].set_linewidth(2)
    
    ax.set_xlabel('Signature Component', fontsize=10, fontweight='bold')
    ax.set_ylabel('Value', fontsize=10, fontweight='bold')
    ax.set_title('C. Oscillatory Signature', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Statistics
    info_text = f'Dim: {len(signature)}\nMean: {signature.mean():.3f}\nStd: {signature.std():.3f}'
    ax.text(0.03, 0.97, info_text, transform=ax.transAxes,
           fontsize=8, va='top', ha='left',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    return ax

def panel_d_pairwise_distances(ax, thought):
    """Panel D: Pairwise O₂ distance matrix"""
    o2_pos = thought['o2_positions']
    
    # Calculate pairwise distances
    dist_matrix = squareform(pdist(o2_pos))
    
    # Plot heatmap
    im = ax.imshow(dist_matrix, cmap='viridis', aspect='auto', alpha=0.8)
    
    ax.set_xlabel('O₂ Index', fontsize=10, fontweight='bold')
    ax.set_ylabel('O₂ Index', fontsize=10, fontweight='bold')
    ax.set_title('D. Pairwise Distance Matrix', fontsize=11, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Distance (Å)', fontsize=8)
    
    # Statistics
    upper_tri = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
    mean_pairwise = upper_tri.mean()
    
    info_text = f'Mean: {mean_pairwise:.2f} Å\nMin: {upper_tri.min():.2f} Å\nMax: {upper_tri.max():.2f} Å'
    ax.text(0.03, 0.97, info_text, transform=ax.transAxes,
           fontsize=7, va='top', ha='left',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))
    
    return ax

def main():
    """Main function for individual thought analysis"""
    
    # Load thought library
    npz_file = 'thought_library_20251026_051925.npz'
    thoughts = load_thought_library(npz_file)
    
    # Analyze first thought
    thought_idx = 0
    thought = thoughts[thought_idx]
    
    print(f"\n{'='*70}")
    print(f"VISUALIZING THOUGHT {thought_idx}")
    print(f"{'='*70}\n")
    
    # Create figure
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.35)
    
    # Panel A: 3D configuration
    ax_a = fig.add_subplot(gs[0, 0], projection='3d')
    panel_a_3d_configuration(ax_a, thought)
    
    # Panel B: Radial distribution
    ax_b = fig.add_subplot(gs[0, 1])
    panel_b_radial_distribution(ax_b, thought)
    
    # Panel C: Signature spectrum
    ax_c = fig.add_subplot(gs[1, 0])
    panel_c_signature_spectrum(ax_c, thought)
    
    # Panel D: Pairwise distances
    ax_d = fig.add_subplot(gs[1, 1])
    panel_d_pairwise_distances(ax_d, thought)
    
    # Overall title
    fig.suptitle(f'Thought Library Analysis: Thought {thought_idx} (Energy: {thought["energy"]:.6e})', 
                fontsize=13, fontweight='bold', y=0.98)
    
    # Save
    plt.savefig('thought_individual_analysis.pdf', bbox_inches='tight')
    plt.savefig('thought_individual_analysis.png', bbox_inches='tight', dpi=300)
    print("\n✓ Saved: thought_individual_analysis.pdf/.png")
    
    plt.show()
    
    print(f"\n{'='*70}")
    print("INDIVIDUAL THOUGHT ANALYSIS COMPLETE")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
