"""
Similarity Matrix Analysis
2-panel visualization of molecular similarity relationships
"""

import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.cluster.hierarchy import dendrogram, linkage

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 8,
    'axes.linewidth': 0.5,
    'figure.dpi': 300,
})

if __name__ == "__main__":
    # Load data
    print("Loading similarity_matrix.json...")
    with open('similarity_matrix.json', 'r') as f:
        data = json.load(f)

    print(f"Timestamp: {data['timestamp']}")
    print(f"Keys: {list(data.keys())}")

    # Extract matrix
    if 'matrix' in data:
        matrix = np.array(data['matrix'])
        print(f"Matrix shape: {matrix.shape}")
        print(f"Similarity range: {matrix.min():.4f} - {matrix.max():.4f}")
    else:
        print("No matrix found in data")
        matrix = None

    # Create figure
    fig = plt.figure(figsize=(12, 5))
    gs = GridSpec(1, 2, figure=fig, hspace=0.3, wspace=0.3)

    # ============================================================================
    # PANEL A: Similarity Matrix Heatmap
    # ============================================================================
    ax_a = fig.add_subplot(gs[0, 0])

    if matrix is not None:
        im = ax_a.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1, 
                        aspect='auto', interpolation='nearest')
        
        # Add text annotations
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                text_color = 'white' if matrix[i, j] < 0.5 else 'black'
                ax_a.text(j, i, f'{matrix[i, j]:.2f}',
                        ha="center", va="center", color=text_color,
                        fontsize=8, fontweight='bold')
        
        # Labels if available
        if 'labels' in data:
            labels = data['labels']
            ax_a.set_xticks(range(len(labels)))
            ax_a.set_yticks(range(len(labels)))
            ax_a.set_xticklabels(labels, fontsize=9, rotation=45, ha='right')
            ax_a.set_yticklabels(labels, fontsize=9)
        else:
            ax_a.set_xticks(range(matrix.shape[1]))
            ax_a.set_yticks(range(matrix.shape[0]))
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax_a, fraction=0.046, pad=0.04)
        cbar.set_label('Similarity', fontsize=9, fontweight='bold')
        
        # Statistics
        mask = ~np.eye(matrix.shape[0], dtype=bool)
        off_diag = matrix[mask]
        
        stats_text = f'Off-diagonal stats:\n'
        stats_text += f'Mean: {off_diag.mean():.3f}\n'
        stats_text += f'Std: {off_diag.std():.3f}\n'
        stats_text += f'Range: {off_diag.min():.3f} - {off_diag.max():.3f}'
        
        ax_a.text(0.02, 0.98, stats_text, 
                transform=ax_a.transAxes, fontsize=7, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    else:
        ax_a.text(0.5, 0.5, 'No matrix data available', 
                ha='center', va='center', transform=ax_a.transAxes,
                fontsize=12)

    ax_a.set_title('A. Similarity Matrix Heatmap', fontsize=12, fontweight='bold')

    # ============================================================================
    # PANEL B: Hierarchical Clustering
    # ============================================================================
    ax_b = fig.add_subplot(gs[0, 1])

    if matrix is not None and matrix.shape[0] > 1:
        # Convert similarity to distance
        distance_matrix = 1 - matrix
        
        # Perform hierarchical clustering
        # Convert to condensed form
        from scipy.spatial.distance import squareform
        condensed = squareform(distance_matrix, checks=False)
        
        linkage_matrix = linkage(condensed, method='average')
        
        # Create dendrogram
        if 'labels' in data:
            labels = data['labels']
        else:
            labels = [f'Item {i}' for i in range(matrix.shape[0])]
        
        dendro = dendrogram(linkage_matrix, labels=labels, 
                        ax=ax_b, orientation='right',
                        color_threshold=0.5,
                        above_threshold_color='gray')
        
        ax_b.set_xlabel('Distance (1 - Similarity)', fontsize=10, fontweight='bold')
        ax_b.set_title('B. Hierarchical Clustering', fontsize=12, fontweight='bold')
        ax_b.grid(True, alpha=0.3, axis='x')
        
        # Add clustering info
        n_clusters = len(set([c for c in dendro['color_list'] if c != 'gray']))
        cluster_text = f'Clusters detected: {n_clusters}\n'
        cluster_text += f'Linkage method: average\n'
        cluster_text += f'Distance metric: 1 - similarity'
        
        ax_b.text(0.98, 0.02, cluster_text, 
                transform=ax_b.transAxes, fontsize=7, va='bottom', ha='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    else:
        ax_b.text(0.5, 0.5, 'Insufficient data for clustering', 
                ha='center', va='center', transform=ax_b.transAxes,
                fontsize=12)
        ax_b.set_title('B. Hierarchical Clustering', fontsize=12, fontweight='bold')

    # Overall title
    fig.suptitle('Similarity Matrix Analysis', 
                fontsize=14, fontweight='bold', y=0.98)

    # Save
    plt.savefig('similarity_matrix_analysis.pdf', bbox_inches='tight')
    plt.savefig('similarity_matrix_analysis.png', bbox_inches='tight', dpi=300)
    print("\n✓ Saved: similarity_matrix_analysis.pdf/.png")

    plt.show()
