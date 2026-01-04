"""
Olfactory Ensemble: Spatial Analysis
4-panel visualization of O₂ spatial distributions and geometries
"""

import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, squareform

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
    print(f"Name: {data['name']}")
    print(f"N_BMDs: {data['n_bmds']}")
    print(f"O₂ concentration: {data['o2_concentration']}")
    print(f"Geometries: {len(data['geometries'])}")
    
    return data

def extract_all_positions(data):
    """Extract all O₂ positions from all geometries"""
    all_positions = []
    for geom in data['geometries']:
        positions = np.array(geom['o2_positions'])
        all_positions.append(positions)
    
    all_positions = np.vstack(all_positions)
    print(f"\nTotal O₂ molecules: {len(all_positions)}")
    
    return all_positions

def calculate_statistics(positions):
    """Calculate spatial statistics"""
    com = positions.mean(axis=0)
    distances = np.linalg.norm(positions - com, axis=1)
    rg = np.sqrt(np.mean(distances**2))
    
    stats = {
        'com': com,
        'distances': distances,
        'rg': rg,
        'range_x': (positions[:, 0].min(), positions[:, 0].max()),
        'range_y': (positions[:, 1].min(), positions[:, 1].max()),
        'range_z': (positions[:, 2].min(), positions[:, 2].max()),
    }
    
    return stats

def panel_a_3d_scatter(ax, positions, stats):
    """Panel A: 3D scatter plot of O₂ positions"""
    com = stats['com']
    distances = stats['distances']
    
    scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                        c=distances, cmap='viridis', s=30, alpha=0.6,
                        edgecolors='black', linewidth=0.3)
    
    # Mark center of mass
    ax.scatter([com[0]], [com[1]], [com[2]], c='red', s=200, marker='*',
              edgecolors='black', linewidth=2, zorder=10)
    
    ax.set_xlabel('X (Å)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Y (Å)', fontsize=10, fontweight='bold')
    ax.set_zlabel('Z (Å)', fontsize=10, fontweight='bold')
    ax.set_title('A. 3D O₂ Distribution', fontsize=11, fontweight='bold')
    ax.view_init(elev=20, azim=45)
    
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.03, pad=0.1, shrink=0.8)
    cbar.set_label('Distance from COM (Å)', fontsize=8)
    
    return ax

def panel_b_xy_projection(ax, positions, stats):
    """Panel B: XY projection with density"""
    com = stats['com']
    distances = stats['distances']
    
    scatter = ax.scatter(positions[:, 0], positions[:, 1],
                        c=distances, cmap='plasma', s=50, alpha=0.6,
                        edgecolors='black', linewidth=0.5)
    
    ax.scatter([com[0]], [com[1]], c='red', s=150, marker='*',
              edgecolors='black', linewidth=2, zorder=10)
    
    ax.set_xlabel('X (Å)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Y (Å)', fontsize=10, fontweight='bold')
    ax.set_title('B. XY Projection', fontsize=11, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Distance (Å)', fontsize=8)
    
    return ax

def panel_c_radial_distribution(ax, stats):
    """Panel C: Radial distribution from COM"""
    distances = stats['distances']
    
    counts, bins = np.histogram(distances, bins=30)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    ax.bar(bin_centers, counts, width=bins[1]-bins[0], 
          color='#3498db', alpha=0.7, edgecolor='black', linewidth=1)
    
    # Mark mean and median
    mean_dist = distances.mean()
    median_dist = np.median(distances)
    
    ax.axvline(mean_dist, color='red', linestyle='--', linewidth=2, 
              label=f'Mean: {mean_dist:.1f} Å')
    ax.axvline(median_dist, color='orange', linestyle='--', linewidth=2,
              label=f'Median: {median_dist:.1f} Å')
    
    ax.set_xlabel('Distance from COM (Å)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Count', fontsize=10, fontweight='bold')
    ax.set_title('C. Radial Distribution', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    return ax

def panel_d_coordinate_distributions(ax, positions):
    """Panel D: Distribution of X, Y, Z coordinates"""
    
    ax.hist(positions[:, 0], bins=25, alpha=0.6, color='#e74c3c', 
           edgecolor='black', linewidth=0.8, label='X')
    ax.hist(positions[:, 1], bins=25, alpha=0.6, color='#2ecc71',
           edgecolor='black', linewidth=0.8, label='Y')
    ax.hist(positions[:, 2], bins=25, alpha=0.6, color='#3498db',
           edgecolor='black', linewidth=0.8, label='Z')
    
    ax.set_xlabel('Position (Å)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Count', fontsize=10, fontweight='bold')
    ax.set_title('D. Coordinate Distributions', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    return ax

def main():
    """Main function to create spatial analysis visualization"""
    
    # Load data
    data = load_data('olfactory_ensemble.json')
    
    # Extract positions
    all_positions = extract_all_positions(data)
    
    # Calculate statistics
    stats = calculate_statistics(all_positions)
    
    print(f"\nSpatial Statistics:")
    print(f"  Center of mass: [{stats['com'][0]:.2f}, {stats['com'][1]:.2f}, {stats['com'][2]:.2f}]")
    print(f"  Radius of gyration: {stats['rg']:.2f} Å")
    print(f"  X range: {stats['range_x'][0]:.2f} to {stats['range_x'][1]:.2f} Å")
    print(f"  Y range: {stats['range_y'][0]:.2f} to {stats['range_y'][1]:.2f} Å")
    print(f"  Z range: {stats['range_z'][0]:.2f} to {stats['range_z'][1]:.2f} Å")
    
    # Create figure
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.35)
    
    # Panel A: 3D scatter
    ax_a = fig.add_subplot(gs[0, 0], projection='3d')
    panel_a_3d_scatter(ax_a, all_positions, stats)
    
    # Panel B: XY projection
    ax_b = fig.add_subplot(gs[0, 1])
    panel_b_xy_projection(ax_b, all_positions, stats)
    
    # Panel C: Radial distribution
    ax_c = fig.add_subplot(gs[1, 0])
    panel_c_radial_distribution(ax_c, stats)
    
    # Panel D: Coordinate distributions
    ax_d = fig.add_subplot(gs[1, 1])
    panel_d_coordinate_distributions(ax_d, all_positions)
    
    # Overall title
    fig.suptitle(f'Olfactory Ensemble Spatial Analysis: {data["n_bmds"]} BMDs, [O₂]={data["o2_concentration"]}', 
                fontsize=13, fontweight='bold', y=0.98)
    
    # Save
    plt.savefig('olfactory_spatial_analysis.pdf', bbox_inches='tight')
    plt.savefig('olfactory_spatial_analysis.png', bbox_inches='tight', dpi=300)
    print("\n✓ Saved: olfactory_spatial_analysis.pdf/.png")
    
    plt.show()
    
    print("\n" + "="*80)
    print("SPATIAL ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
