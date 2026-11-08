"""
Figure 3: Multi-Modal Constraint Integration

Panel A: 12-Coordinate Phase Space (3D)
Panel B: Constraint Satisfaction Matrix
Panel C: Temporal Evolution
Panel D: Resolution Map (3D)
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from pathlib import Path
import seaborn as sns

def generate_figure_3(output_dir, dpi=300):
    """Generate Figure 3 with 4 panels."""
    fig = plt.figure(figsize=(16, 12))
    
    # Panel A: 12-Coordinate Phase Space (3D)
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    _panel_a_phase_space(ax1)
    
    # Panel B: Constraint Satisfaction Matrix
    ax2 = fig.add_subplot(2, 2, 2)
    _panel_b_constraint_matrix(ax2)
    
    # Panel C: Temporal Evolution
    ax3 = fig.add_subplot(2, 2, 3)
    _panel_c_temporal_evolution(ax3)
    
    # Panel D: Resolution Map (3D)
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    _panel_d_resolution_map(ax4)
    
    plt.tight_layout()
    output_path = output_dir / "figure_03_constraint_integration.png"
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

def _panel_a_phase_space(ax):
    """Panel A: 3D projection of 12D phase space."""
    # Generate molecular positions in 12D space
    n_molecules = 500
    data_12d = np.random.randn(n_molecules, 12)
    
    # PCA to 3D for visualization
    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        data_3d = pca.fit_transform(data_12d)
    except ImportError:
        # Fallback: use first 3 dimensions
        data_3d = data_12d[:, :3]
    
    # Color by molecule type (simulated)
    molecule_types = np.random.randint(0, 5, n_molecules)
    colors = plt.cm.Set3(molecule_types / 5)
    
    # Plot points
    scatter = ax.scatter(*data_3d.T, c=molecule_types, cmap='Set3', 
                        s=30, alpha=0.6, edgecolors='black', linewidths=0.5)
    
    # Constraint surfaces (semi-transparent)
    # O₂ surface
    u = np.linspace(-3, 3, 20)
    v = np.linspace(-3, 3, 20)
    U, V = np.meshgrid(u, v)
    W = 0.5 * U + 0.3 * V  # Example surface
    ax.plot_surface(U, V, W, alpha=0.3, color='red', label='O₂ constraint')
    
    # ΔΨ surface
    W2 = -0.3 * U + 0.5 * V
    ax.plot_surface(U, V, W2, alpha=0.3, color='blue', label='ΔΨ constraint')
    
    # pH surface
    W3 = 0.2 * U - 0.4 * V
    ax.plot_surface(U, V, W3, alpha=0.3, color='green', label='pH constraint')
    
    # ATP surface
    W4 = 0.4 * U + 0.2 * V
    ax.plot_surface(U, V, W4, alpha=0.3, color='yellow', label='ATP constraint')
    
    # Intersection points (valid positions)
    # Find points near all surfaces (simplified)
    valid_mask = (np.abs(data_3d[:, 2] - (0.5*data_3d[:, 0] + 0.3*data_3d[:, 1])) < 0.5) & \
                 (np.abs(data_3d[:, 2] - (-0.3*data_3d[:, 0] + 0.5*data_3d[:, 1])) < 0.5)
    valid_points = data_3d[valid_mask]
    if len(valid_points) > 0:
        ax.scatter(*valid_points.T, s=100, c='yellow', alpha=1.0, 
                  edgecolors='black', linewidths=2, label='Valid positions')
    
    ax.set_xlabel('PC1', fontsize=10)
    ax.set_ylabel('PC2', fontsize=10)
    ax.set_zlabel('PC3', fontsize=10)
    ax.set_title('Panel A: 12-Coordinate Phase Space (3D)', fontsize=12, fontweight='bold')
    ax.view_init(elev=20, azim=45)
    ax.legend(loc='upper left', fontsize=7)

def _panel_b_constraint_matrix(ax):
    """Panel B: Constraint satisfaction matrix heatmap."""
    # 12 coordinate systems
    coord_systems = ['O₂', 'ΔΨ', 'pH', 'ATP', 'ρ_protein', 'T', 'P', 'V', 
                    'S_k', 'S_t', 'S_e', 'Network']
    
    # Generate coupling matrix
    n = len(coord_systems)
    coupling_matrix = np.random.rand(n, n)
    
    # Make symmetric
    coupling_matrix = (coupling_matrix + coupling_matrix.T) / 2
    
    # Strong diagonal (self-consistency)
    np.fill_diagonal(coupling_matrix, 1.0)
    
    # Strong cross-couplings (example)
    coupling_matrix[0, 4] = 0.9  # O₂ - ρ_protein
    coupling_matrix[4, 0] = 0.9
    coupling_matrix[1, 2] = 0.85  # ΔΨ - pH
    coupling_matrix[2, 1] = 0.85
    coupling_matrix[3, 5] = 0.8  # ATP - T
    coupling_matrix[5, 3] = 0.8
    
    # Heatmap
    im = ax.imshow(coupling_matrix, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
    
    # Add text annotations for strongest couplings
    threshold = 0.7
    for i in range(n):
        for j in range(n):
            if coupling_matrix[i, j] > threshold:
                ax.text(j, i, f'{coupling_matrix[i, j]:.2f}',
                       ha='center', va='center', fontsize=7, 
                       color='white' if coupling_matrix[i, j] > 0.8 else 'black',
                       fontweight='bold')
    
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(coord_systems, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(coord_systems, fontsize=8)
    ax.set_title('Panel B: Constraint Satisfaction Matrix', fontsize=12, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Coupling Strength', fontsize=10)

def _panel_c_temporal_evolution(ax):
    """Panel C: Temporal evolution of constraint satisfaction."""
    # Time array (ms)
    t = np.linspace(0, 2, 1000)
    
    # 12 coordinate systems
    coord_systems = ['O₂', 'ΔΨ', 'pH', 'ATP', 'ρ_protein', 'T', 'P', 'V', 
                    'S_k', 'S_t', 'S_e', 'Network']
    
    # Generate convergence curves (all converge to 1)
    colors = plt.cm.tab20(np.linspace(0, 1, len(coord_systems)))
    
    for i, (coord, color) in enumerate(zip(coord_systems, colors)):
        # Exponential convergence
        tau = 0.3 + 0.2 * np.random.rand()  # Convergence time constant
        satisfaction = 1 - np.exp(-t / tau)
        satisfaction += 0.05 * np.random.randn(len(t)) * np.exp(-t / tau)  # Noise decreases
        
        ax.plot(t, satisfaction, linewidth=2, label=coord, color=color, alpha=0.8)
    
    # Shaded convergence region
    ax.axvspan(0, 1, alpha=0.2, color='gray', label='Convergence time')
    
    ax.axhline(1, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(1, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Time (ms)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Constraint Satisfaction Score', fontsize=11, fontweight='bold')
    ax.set_title('Panel C: Temporal Evolution', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.set_xlim(0, 2)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3, fontsize=7, loc='lower right')
    
    # Annotation
    ax.annotate('Real-time state determination', 
               xy=(1, 1), xytext=(1.5, 0.5),
               arrowprops=dict(arrowstyle='->', lw=2),
               fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

def _panel_d_resolution_map(ax):
    """Panel D: 3D resolution map of cell."""
    # Create 3D cellular volume
    x = np.linspace(0, 10, 30)
    y = np.linspace(0, 10, 30)
    z = np.linspace(0, 10, 30)
    X, Y, Z = np.meshgrid(x, y, z)
    
    # Resolution map: high near membranes (edges), lower in center
    center = np.array([5, 5, 5])
    distances = np.sqrt((X - center[0])**2 + (Y - center[1])**2 + (Z - center[2])**2)
    
    # Resolution: 0.1 nm near edges, 1 nm in center
    resolution = 0.1 + 0.9 * (1 - np.exp(-distances / 2))
    
    # Create isosurface at 0.5 nm
    try:
        from skimage import measure
        verts, faces, normals, values = measure.marching_cubes(
            resolution, level=0.5, spacing=(x[1]-x[0], y[1]-y[0], z[1]-z[0]))
        
        # Plot isosurface
        ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], 
                       triangles=faces, alpha=0.3, color='cyan', edgecolor='none')
    except (ImportError, Exception):
        # Fallback: plot slices
        mid_z = len(z) // 2
        ax.contourf(X[:, :, mid_z], Y[:, :, mid_z], resolution[:, :, mid_z], 
                   levels=20, cmap='viridis', alpha=0.6)
    
    # Slice planes
    mid_y = len(y) // 2
    ax.contourf(X[:, mid_y, :], Z[:, mid_y, :], resolution[:, mid_y, :], 
               levels=20, cmap='viridis', alpha=0.4, zdir='y', offset=5)
    
    # Color by resolution
    scatter = ax.scatter(X[::3, ::3, ::3], Y[::3, ::3, ::3], Z[::3, ::3, ::3],
                        c=resolution[::3, ::3, ::3], cmap='viridis_r', 
                        s=20, alpha=0.6, vmin=0.1, vmax=1.0)
    
    ax.set_xlabel('X (μm)', fontsize=10)
    ax.set_ylabel('Y (μm)', fontsize=10)
    ax.set_zlabel('Z (μm)', fontsize=10)
    ax.set_title('Panel D: Resolution Map (3D)', fontsize=12, fontweight='bold')
    ax.view_init(elev=20, azim=45)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5)
    cbar.set_label('Resolution (nm)', fontsize=9)

if __name__ == "__main__":
    output_dir = Path(__file__).parent
    generate_figure_3(output_dir, dpi=300)
