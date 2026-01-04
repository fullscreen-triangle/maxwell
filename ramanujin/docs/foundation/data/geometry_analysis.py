"""
Olfactory Ensemble: Geometry Analysis
4-panel visualization of hole geometries and molecular signatures
"""

import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle
from scipy.stats import pearsonr

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 9,
    'axes.linewidth': 0.8,
    'figure.dpi': 300,
})

def load_data(filename='olfactory_ensemble.json'):
    """Load olfactory ensemble data"""
    print(f"Loading {filename}...")
    with open(filename, 'r') as f:
        data = json.load(f)
    
    print(f"Timestamp: {data['timestamp']}")
    print(f"N_BMDs: {data['n_bmds']}")
    print(f"Geometries: {len(data['geometries'])}")
    
    return data

def extract_geometry_properties(data):
    """Extract properties from all geometries"""
    n_geom = len(data['geometries'])
    
    hole_radii = np.zeros(n_geom)
    molecular_densities = np.zeros(n_geom)
    hole_centers = np.zeros((n_geom, 3))
    
    for i, geom in enumerate(data['geometries']):
        hole_radii[i] = geom['hole_radius']
        molecular_densities[i] = geom['molecular_density']
        hole_centers[i] = geom['hole_center']
    
    properties = {
        'hole_radii': hole_radii,
        'molecular_densities': molecular_densities,
        'hole_centers': hole_centers,
    }
    
    print(f"\nGeometry Properties:")
    print(f"  Hole radius range: {hole_radii.min():.3f} - {hole_radii.max():.3f} Å")
    print(f"  Mean hole radius: {hole_radii.mean():.3f} ± {hole_radii.std():.3f} Å")
    print(f"  Density range: {molecular_densities.min():.4f} - {molecular_densities.max():.4f}")
    print(f"  Mean density: {molecular_densities.mean():.4f} ± {molecular_densities.std():.4f}")
    
    return properties

def panel_a_hole_radius_distribution(ax, properties):
    """Panel A: Distribution of hole radii"""
    radii = properties['hole_radii']
    
    counts, bins, patches = ax.hist(radii, bins=15, color='#3498db', alpha=0.7,
                                    edgecolor='black', linewidth=1.2)
    
    # Color bars by value
    cm = plt.cm.viridis
    bin_centers = (bins[:-1] + bins[1:]) / 2
    col = bin_centers - min(bin_centers)
    col /= max(col)
    
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(c))
    
    # Mark mean
    mean_radius = radii.mean()
    ax.axvline(mean_radius, color='red', linestyle='--', linewidth=2.5,
              label=f'Mean: {mean_radius:.2f} Å')
    
    ax.set_xlabel('Hole Radius (Å)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Count', fontsize=10, fontweight='bold')
    ax.set_title('A. Hole Radius Distribution', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    return ax

def panel_b_density_vs_radius(ax, properties):
    """Panel B: Molecular density vs hole radius"""
    radii = properties['hole_radii']
    densities = properties['molecular_densities']
    
    # Scatter plot
    scatter = ax.scatter(radii, densities, s=100, c=densities,
                        cmap='plasma', alpha=0.7, edgecolors='black', linewidth=1)
    
    # Fit line
    z = np.polyfit(radii, densities, 1)
    p = np.poly1d(z)
    x_fit = np.linspace(radii.min(), radii.max(), 100)
    ax.plot(x_fit, p(x_fit), 'r--', linewidth=2, alpha=0.8, label='Linear fit')
    
    # Calculate correlation
    corr, pval = pearsonr(radii, densities)
    
    ax.set_xlabel('Hole Radius (Å)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Molecular Density', fontsize=10, fontweight='bold')
    ax.set_title('B. Density vs Radius', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    
    # Add correlation info
    info_text = f'r = {corr:.3f}\np = {pval:.3e}'
    ax.text(0.05, 0.95, info_text, transform=ax.transAxes,
           fontsize=8, va='top', ha='left',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Density', fontsize=8)
    
    return ax

def panel_c_hole_centers_3d(ax, properties, data):
    """Panel C: 3D visualization of hole centers"""
    centers = properties['hole_centers']
    radii = properties['hole_radii']
    
    # Normalize radii for marker size
    sizes = (radii - radii.min()) / (radii.max() - radii.min()) * 500 + 100
    
    scatter = ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
                        s=sizes, c=radii, cmap='coolwarm', alpha=0.7,
                        edgecolors='black', linewidth=1.5)
    
    # Mark overall center
    overall_center = centers.mean(axis=0)
    ax.scatter([overall_center[0]], [overall_center[1]], [overall_center[2]],
              c='red', s=300, marker='*', edgecolors='black', linewidth=2,
              zorder=10)
    
    ax.set_xlabel('X (Å)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Y (Å)', fontsize=10, fontweight='bold')
    ax.set_zlabel('Z (Å)', fontsize=10, fontweight='bold')
    ax.set_title('C. Hole Center Positions', fontsize=11, fontweight='bold')
    ax.view_init(elev=20, azim=45)
    
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.03, pad=0.1, shrink=0.8)
    cbar.set_label('Radius (Å)', fontsize=8)
    
    return ax

def panel_d_geometry_evolution(ax, properties):
    """Panel D: Evolution of geometry properties across BMDs"""
    radii = properties['hole_radii']
    densities = properties['molecular_densities']
    
    bmd_indices = np.arange(len(radii))
    
    # Twin axes for two y-scales
    ax2 = ax.twinx()
    
    # Plot radius
    line1 = ax.plot(bmd_indices, radii, 'o-', color='#3498db', linewidth=2,
                   markersize=6, markeredgecolor='black', markeredgewidth=0.5,
                   label='Hole Radius', alpha=0.8)
    
    # Plot density
    line2 = ax2.plot(bmd_indices, densities, 's-', color='#e74c3c', linewidth=2,
                    markersize=6, markeredgecolor='black', markeredgewidth=0.5,
                    label='Molecular Density', alpha=0.8)
    
    ax.set_xlabel('Geometry Index', fontsize=10, fontweight='bold')
    ax.set_ylabel('Hole Radius (Å)', fontsize=10, fontweight='bold', color='#3498db')
    ax2.set_ylabel('Molecular Density', fontsize=10, fontweight='bold', color='#e74c3c')
    ax.set_title('D. Geometry Evolution', fontsize=11, fontweight='bold')
    
    ax.tick_params(axis='y', labelcolor='#3498db')
    ax2.tick_params(axis='y', labelcolor='#e74c3c')
    
    ax.grid(True, alpha=0.3)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, fontsize=8, loc='upper right')
    
    return ax

def main():
    """Main function to create geometry analysis visualization"""
    
    # Load data
    data = load_data('olfactory_ensemble.json')
    
    # Extract geometry properties
    properties = extract_geometry_properties(data)
    
    # Create figure
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.35)
    
    # Panel A: Hole radius distribution
    ax_a = fig.add_subplot(gs[0, 0])
    panel_a_hole_radius_distribution(ax_a, properties)
    
    # Panel B: Density vs radius
    ax_b = fig.add_subplot(gs[0, 1])
    panel_b_density_vs_radius(ax_b, properties)
    
    # Panel C: Hole centers 3D
    ax_c = fig.add_subplot(gs[1, 0], projection='3d')
    panel_c_hole_centers_3d(ax_c, properties, data)
    
    # Panel D: Geometry evolution
    ax_d = fig.add_subplot(gs[1, 1])
    panel_d_geometry_evolution(ax_d, properties)
    
    # Overall title
    fig.suptitle(f'Olfactory Ensemble Geometry Analysis: {data["n_bmds"]} BMDs', 
                fontsize=13, fontweight='bold', y=0.98)
    
    # Save
    plt.savefig('olfactory_geometry_analysis.pdf', bbox_inches='tight')
    plt.savefig('olfactory_geometry_analysis.png', bbox_inches='tight', dpi=300)
    print("\n✓ Saved: olfactory_geometry_analysis.pdf/.png")
    
    plt.show()
    
    print("\n" + "="*80)
    print("GEOMETRY ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
