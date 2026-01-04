"""
Molecular Signatures Analysis
2-panel visualization of molecular signature properties
"""

import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 8,
    'axes.linewidth': 0.5,
    'figure.dpi': 300,
})

if __name__ == "__main__":
    # Load data
    print("Loading molecular_signatures.json...")
    with open('molecular_signatures.json', 'r') as f:
        data = json.load(f)

    print(f"Timestamp: {data['timestamp']}")
    print(f"Keys: {list(data.keys())}")

    # Extract signatures
    if 'signatures' in data:
        sigs = data['signatures']
        names = list(sigs.keys())
        print(f"Molecules: {names}")
    else:
        print("No signatures found")
        sigs = {}
        names = []

    # Create figure
    fig = plt.figure(figsize=(12, 5))
    gs = GridSpec(1, 2, figure=fig, hspace=0.3, wspace=0.3)

    # ============================================================================
    # PANEL A: Molecular Property Comparison
    # ============================================================================
    ax_a = fig.add_subplot(gs[0, 0])

    if names:
        # Extract numerical properties
        properties = {}
        for name in names:
            sig_data = sigs[name]
            for key, value in sig_data.items():
                if isinstance(value, (int, float)):
                    if key not in properties:
                        properties[key] = []
                    properties[key].append(value)
        
        # Plot first property as bar chart
        if properties:
            first_prop = list(properties.keys())[0]
            values = properties[first_prop]
            
            colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
            bars = ax_a.bar(range(len(names)), values,
                        color=[colors[i % len(colors)] for i in range(len(names))],
                        alpha=0.7, edgecolor='black', linewidth=1.5)
            
            ax_a.set_xticks(range(len(names)))
            ax_a.set_xticklabels(names, fontsize=9, rotation=45, ha='right')
            ax_a.set_ylabel(first_prop, fontsize=10, fontweight='bold')
            ax_a.grid(True, alpha=0.3, axis='y')
            
            # Add values on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax_a.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.2f}',
                        ha='center', va='bottom', fontsize=7)
            
            # Statistics
            stats_text = f'Mean: {np.mean(values):.3f}\n'
            stats_text += f'Std: {np.std(values):.3f}\n'
            stats_text += f'Range: {np.min(values):.3f} - {np.max(values):.3f}'
            
            ax_a.text(0.02, 0.98, stats_text, 
                    transform=ax_a.transAxes, fontsize=7, va='top', ha='left',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    else:
        ax_a.text(0.5, 0.5, 'No molecular data available', 
                ha='center', va='center', transform=ax_a.transAxes,
                fontsize=12)

    ax_a.set_title('A. Molecular Property Comparison', fontsize=12, fontweight='bold')

    # ============================================================================
    # PANEL B: Signature Space (PCA)
    # ============================================================================
    ax_b = fig.add_subplot(gs[0, 1])

    if names and len(names) > 1:
        # Create feature vectors
        vectors = []
        for name in names:
            sig_data = sigs[name]
            vec = []
            for key, value in sig_data.items():
                if isinstance(value, (int, float)):
                    vec.append(value)
            vectors.append(vec)
        
        vectors = np.array(vectors)
        
        if vectors.shape[1] >= 2:
            # PCA
            pca = PCA(n_components=min(2, vectors.shape[1]))
            coords_2d = pca.fit_transform(vectors)
            
            # Plot
            colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
            for i, name in enumerate(names):
                ax_b.scatter(coords_2d[i, 0], coords_2d[i, 1],
                            s=300, c=colors[i % len(colors)], alpha=0.7,
                            edgecolors='black', linewidth=2, label=name, zorder=10)
                
                ax_b.annotate(name, xy=(coords_2d[i, 0], coords_2d[i, 1]),
                            xytext=(10, 10), textcoords='offset points',
                            fontsize=9, fontweight='bold',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            ax_b.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', 
                        fontsize=10, fontweight='bold')
            if coords_2d.shape[1] > 1:
                ax_b.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', 
                            fontsize=10, fontweight='bold')
            ax_b.legend(fontsize=8, loc='best')
            ax_b.grid(True, alpha=0.3)
            ax_b.axhline(0, color='black', linewidth=0.5, alpha=0.5)
            ax_b.axvline(0, color='black', linewidth=0.5, alpha=0.5)
            
            # Variance explained
            var_text = f'Total variance explained:\n'
            var_text += f'{pca.explained_variance_ratio_.sum()*100:.1f}%'
            
            ax_b.text(0.02, 0.98, var_text, 
                    transform=ax_b.transAxes, fontsize=8, va='top', ha='left',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        else:
            ax_b.text(0.5, 0.5, 'Insufficient features for PCA', 
                    ha='center', va='center', transform=ax_b.transAxes,
                    fontsize=12)
    else:
        ax_b.text(0.5, 0.5, 'Need at least 2 molecules for PCA', 
                ha='center', va='center', transform=ax_b.transAxes,
                fontsize=12)

    ax_b.set_title('B. Signature Space (PCA)', fontsize=12, fontweight='bold')

    # Overall title
    fig.suptitle(f'Molecular Signatures Analysis ({len(names)} molecules)', 
                fontsize=14, fontweight='bold', y=0.98)

    # Save
    plt.savefig('molecular_signatures_analysis.pdf', bbox_inches='tight')
    plt.savefig('molecular_signatures_analysis.png', bbox_inches='tight', dpi=300)
    print("\n✓ Saved: molecular_signatures_analysis.pdf/.png")

    plt.show()
