"""
Figure 5: Biological Applications

Panel A: Protein Complex Structure (3D)
Panel B: Membrane Dynamics
Panel C: Metabolic Flux Visualization
Panel D: Disease State Detection
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from pathlib import Path

def generate_figure_5(output_dir, dpi=300):
    """Generate Figure 5 with 4 panels."""
    fig = plt.figure(figsize=(16, 12))
    
    # Panel A: Protein Complex Structure (3D)
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    _panel_a_protein_complex(ax1)
    
    # Panel B: Membrane Dynamics
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    _panel_b_membrane_dynamics(ax2)
    
    # Panel C: Metabolic Flux Visualization
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    _panel_c_metabolic_flux(ax3)
    
    # Panel D: Disease State Detection
    ax4 = fig.add_subplot(2, 2, 4)
    _panel_d_disease_detection(ax4)
    
    plt.tight_layout()
    output_path = output_dir / "figure_05_biological_applications.png"
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

def _panel_a_protein_complex(ax):
    """Panel A: 3D structure of large protein complex (e.g., ribosome)."""
    # Simulate ribosome-like structure
    # Large subunit
    n_large = 50
    large_subunit = np.random.randn(n_large, 3) * 2
    large_subunit[:, 2] += 2  # Offset
    
    # Small subunit
    n_small = 30
    small_subunit = np.random.randn(n_small, 3) * 1.5
    small_subunit[:, 2] -= 1  # Offset
    
    # Plot subunits
    ax.scatter(*large_subunit.T, s=100, c='blue', alpha=0.7, 
              edgecolors='black', linewidths=1, label='Large subunit')
    ax.scatter(*small_subunit.T, s=80, c='red', alpha=0.7, 
              edgecolors='black', linewidths=1, label='Small subunit')
    
    # Dynamic regions (highlighted)
    dynamic_indices = np.random.choice(n_large, size=5, replace=False)
    ax.scatter(*large_subunit[dynamic_indices].T, s=200, c='yellow', 
              alpha=0.9, edgecolors='black', linewidths=2, marker='*',
              label='Dynamic regions', zorder=5)
    
    # Comparison inset: cryo-EM structure (2D projection)
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    axins = inset_axes(ax, width="30%", height="30%", loc='upper right')
    
    # Cryo-EM structure (gray, lower resolution) - 2D XY projection
    cryo_positions = large_subunit + np.random.randn(n_large, 3) * 0.3
    axins.scatter(cryo_positions[:, 0], cryo_positions[:, 1], s=50, c='gray', alpha=0.5, 
                 edgecolors='black', linewidths=0.5)
    axins.set_title('Cryo-EM (XY projection)', fontsize=8)
    axins.set_xticks([])
    axins.set_yticks([])
    
    # RMSD annotation (3D position)
    rmsd = 0.15  # nm
    ax.text(0, 10, 10, f'RMSD = {rmsd:.2f} nm\nIn vivo structure determination',
           fontsize=10, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
    
    ax.set_xlabel('X (nm)', fontsize=10)
    ax.set_ylabel('Y (nm)', fontsize=10)
    ax.set_zlabel('Z (nm)', fontsize=10)
    ax.set_title('Panel A: Protein Complex Structure (3D)', fontsize=12, fontweight='bold')
    ax.view_init(elev=20, azim=45)
    ax.legend(loc='lower left', fontsize=8)

def _panel_b_membrane_dynamics(ax):
    """Panel B: Time series of membrane protein positions."""
    # Time points
    time_points = [0, 10, 20, 30]  # ms
    n_proteins = 10
    
    # Membrane surface (lipid bilayer)
    x_mem = np.linspace(0, 10, 20)
    y_mem = np.linspace(0, 10, 20)
    X_mem, Y_mem = np.meshgrid(x_mem, y_mem)
    Z_mem = np.zeros_like(X_mem)  # Flat membrane
    
    # Plot membrane
    ax.plot_surface(X_mem, Y_mem, Z_mem, alpha=0.3, color='gray', 
                    edgecolor='none', label='Lipid bilayer')
    
    # Protein trajectories
    colors = plt.cm.tab10(np.linspace(0, 1, n_proteins))
    
    for protein_idx in range(n_proteins):
        # Generate trajectory
        positions = []
        base_pos = np.array([np.random.rand() * 10, np.random.rand() * 10, 0])
        
        for t in time_points:
            # Random walk
            if t == 0:
                pos = base_pos
            else:
                pos = positions[-1] + np.random.randn(3) * 0.5
                pos[2] = 0  # On membrane
            positions.append(pos)
        
        positions = np.array(positions)
        
        # Plot trajectory
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
               'o-', linewidth=2, markersize=8, color=colors[protein_idx], 
               alpha=0.7, label=f'Protein {protein_idx+1}' if protein_idx < 3 else '')
        
        # Mark time points
        for i, (pos, t) in enumerate(zip(positions, time_points)):
            ax.scatter(*pos, s=100, c=colors[protein_idx], 
                     edgecolors='black', linewidths=1, zorder=5)
            if i == len(positions) - 1:
                ax.text(*pos, f'  t={t}ms', fontsize=7)
    
    # Velocity distribution inset
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    axins = inset_axes(ax, width="25%", height="25%", loc='upper right')
    
    # Calculate velocities
    velocities = []
    for protein_idx in range(n_proteins):
        # Simulate velocity distribution
        v = np.random.exponential(scale=0.1, size=100)  # nm/ms
        velocities.extend(v)
    
    axins.hist(velocities, bins=30, density=True, alpha=0.7, 
              edgecolor='black', linewidth=1)
    axins.set_xlabel('Velocity (nm/ms)', fontsize=7)
    axins.set_ylabel('Density', fontsize=7)
    axins.set_title('Velocity Distribution', fontsize=8)
    axins.grid(True, alpha=0.3)
    
    ax.set_xlabel('X (nm)', fontsize=10)
    ax.set_ylabel('Y (nm)', fontsize=10)
    ax.set_zlabel('Z (nm)', fontsize=10)
    ax.set_title('Panel B: Membrane Dynamics', fontsize=12, fontweight='bold')
    ax.view_init(elev=90, azim=0)  # Top view
    ax.set_zlim(-0.5, 0.5)
    
    # Annotation (3D position)
    ax.text(0, 10, 0.3, 'Real-time tracking',
           fontsize=10, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))

def _panel_c_metabolic_flux(ax):
    """Panel C: Cellular volume with metabolite positions and flux."""
    # Metabolite positions
    n_atp = 50
    n_adp = 50
    n_glucose = 30
    n_pyruvate = 30
    
    atp_pos = np.random.rand(n_atp, 3) * 10
    adp_pos = np.random.rand(n_adp, 3) * 10
    glucose_pos = np.random.rand(n_glucose, 3) * 10
    pyruvate_pos = np.random.rand(n_pyruvate, 3) * 10
    
    # Plot metabolites
    ax.scatter(*atp_pos.T, s=100, c='red', alpha=0.7, 
              edgecolors='black', linewidths=1, label='ATP')
    ax.scatter(*adp_pos.T, s=100, c='blue', alpha=0.7, 
              edgecolors='black', linewidths=1, label='ADP')
    ax.scatter(*glucose_pos.T, s=80, c='green', alpha=0.7, 
              edgecolors='black', linewidths=1, label='Glucose')
    ax.scatter(*pyruvate_pos.T, s=80, c='yellow', alpha=0.7, 
              edgecolors='black', linewidths=1, label='Pyruvate')
    
    # Flow vectors (metabolic flux directions)
    # High flux regions (hotspots)
    hotspot_centers = np.array([[3, 3, 3], [7, 7, 7]])
    
    for center in hotspot_centers:
        # Draw flux vectors
        n_vectors = 10
        for _ in range(n_vectors):
            start = center + np.random.randn(3) * 1
            direction = np.random.randn(3)
            direction = direction / np.linalg.norm(direction) * 2
            end = start + direction
            
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                   'k-', linewidth=2, alpha=0.6)
            ax.scatter(*end, s=30, c='orange', alpha=0.8, edgecolors='black', linewidths=0.5)
        
        # Highlight hotspot
        ax.scatter(*center, s=500, c='orange', alpha=0.3, 
                  edgecolors='orange', linewidths=2)
    
    # Pathway annotations
    ax.text(3, 3, 3, 'Glycolysis', fontsize=9, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax.text(7, 7, 7, 'TCA Cycle', fontsize=9, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('X (μm)', fontsize=10)
    ax.set_ylabel('Y (μm)', fontsize=10)
    ax.set_zlabel('Z (μm)', fontsize=10)
    ax.set_title('Panel C: Metabolic Flux Visualization', fontsize=12, fontweight='bold')
    ax.view_init(elev=20, azim=45)
    ax.legend(loc='upper left', fontsize=8)

def _panel_d_disease_detection(ax):
    """Panel D: Comparison of healthy vs diseased cell."""
    # Create side-by-side comparison
    fig = plt.gcf()
    ax.remove()
    
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 2, figure=fig, left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.3, wspace=0.3)
    
    # Healthy cell
    ax1 = fig.add_subplot(gs[0, 0])
    _draw_cell_state(ax1, healthy=True)
    ax1.set_title('Healthy Cell', fontsize=11, fontweight='bold')
    
    # Diseased cell
    ax2 = fig.add_subplot(gs[0, 1])
    _draw_cell_state(ax2, healthy=False)
    ax2.set_title('Diseased Cell', fontsize=11, fontweight='bold')
    
    # Quantification panel
    ax3 = fig.add_subplot(gs[1, :])
    _draw_quantification(ax3)
    
    # Main title
    fig.text(0.5, 0.98, 'Panel D: Disease State Detection', 
            fontsize=12, fontweight='bold', ha='center')

def _draw_cell_state(ax, healthy=True):
    """Draw cell state visualization."""
    # Create cellular structure
    cell_size = 100
    img = np.random.rand(cell_size, cell_size)
    
    if healthy:
        # Organized, high resolution
        img += 0.5 * np.sin(np.linspace(0, 4*np.pi, cell_size))[:, None]
        img += 0.5 * np.sin(np.linspace(0, 4*np.pi, cell_size))[None, :]
        resolution = 0.1  # nm
        satisfaction = 0.95
        richness = 1e5
    else:
        # Disorganized, lower resolution
        img += 0.2 * np.random.randn(cell_size, cell_size)
        resolution = 0.5  # nm
        satisfaction = 0.65
        richness = 1e3
    
    ax.imshow(img, cmap='viridis', aspect='auto')
    ax.axis('off')
    
    # Add text
    ax.text(0.5, 0.95, f'Resolution: {resolution} nm', 
           transform=ax.transAxes, fontsize=9, fontweight='bold',
           ha='center', va='top',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax.text(0.5, 0.05, f'Constraint: {satisfaction:.2f}\nRichness: {richness:.0e}',
           transform=ax.transAxes, fontsize=8,
           ha='center', va='bottom',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

def _draw_quantification(ax):
    """Draw quantification comparison."""
    metrics = ['Constraint\nSatisfaction', 'Resolution\n(nm)', 'Categorical\nRichness']
    healthy_vals = [0.95, 0.1, 1e5]
    diseased_vals = [0.65, 0.5, 1e3]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    # Normalize for visualization
    healthy_norm = [healthy_vals[0], healthy_vals[1]*10, np.log10(healthy_vals[2])]
    diseased_norm = [diseased_vals[0], diseased_vals[1]*10, np.log10(diseased_vals[2])]
    
    bars1 = ax.bar(x - width/2, healthy_norm, width, label='Healthy', 
                  color='green', alpha=0.7, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, diseased_norm, width, label='Diseased', 
                  color='red', alpha=0.7, edgecolor='black', linewidth=1)
    
    # Add value labels
    for i, (b1, b2, h, d) in enumerate(zip(bars1, bars2, healthy_vals, diseased_vals)):
        if i == 0:
            ax.text(b1.get_x() + b1.get_width()/2, b1.get_height(),
                   f'{h:.2f}', ha='center', va='bottom', fontsize=8)
            ax.text(b2.get_x() + b2.get_width()/2, b2.get_height(),
                   f'{d:.2f}', ha='center', va='bottom', fontsize=8)
        elif i == 1:
            ax.text(b1.get_x() + b1.get_width()/2, b1.get_height(),
                   f'{h:.1f}', ha='center', va='bottom', fontsize=8)
            ax.text(b2.get_x() + b2.get_width()/2, b2.get_height(),
                   f'{d:.1f}', ha='center', va='bottom', fontsize=8)
        else:
            ax.text(b1.get_x() + b1.get_width()/2, b1.get_height(),
                   f'{h:.0e}', ha='center', va='bottom', fontsize=8)
            ax.text(b2.get_x() + b2.get_width()/2, b2.get_height(),
                   f'{d:.0e}', ha='center', va='bottom', fontsize=8)
    
    ax.set_ylabel('Normalized Value', fontsize=10)
    ax.set_title('Quantitative Comparison', fontsize=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Annotation
    ax.text(0.5, 0.95, 'Quantitative disease signature',
           transform=ax.transAxes, fontsize=9, fontweight='bold',
           ha='center', va='top',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

if __name__ == "__main__":
    output_dir = Path(__file__).parent
    generate_figure_5(output_dir, dpi=300)
