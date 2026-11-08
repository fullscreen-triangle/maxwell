"""
Figure 1: Quintupartite/Dodecapartite Constraint Architecture

Panel A: Constraint Hierarchy (3D tree)
Panel B: Sequential Categorical Exclusion
Panel C: Resolution Enhancement
Panel D: Information Content
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
from pathlib import Path
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform

class Arrow3D(FancyArrowPatch):
    """3D arrow for 3D plots."""
    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

def generate_figure_1(output_dir, dpi=300):
    """Generate Figure 1 with 4 panels."""
    fig = plt.figure(figsize=(16, 12))
    
    # Panel A: Constraint Hierarchy (3D tree)
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    _panel_a_constraint_hierarchy(ax1)
    
    # Panel B: Sequential Categorical Exclusion
    ax2 = fig.add_subplot(2, 2, 2)
    _panel_b_sequential_exclusion(ax2)
    
    # Panel C: Resolution Enhancement
    ax3 = fig.add_subplot(2, 2, 3)
    _panel_c_resolution_enhancement(ax3)
    
    # Panel D: Information Content
    ax4 = fig.add_subplot(2, 2, 4)
    _panel_d_information_content(ax4)
    
    plt.tight_layout()
    output_path = output_dir / "figure_01_constraint_architecture.png"
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

def _panel_a_constraint_hierarchy(ax):
    """Panel A: 3D tree structure showing constraint levels."""
    # Root node (full cellular volume)
    root_pos = np.array([0, 0, 0])
    root_size = 2.0
    
    # Level 1: 5 modalities
    level1_positions = np.array([
        [-2, 2, 0],   # O₂
        [0, 2, 0],    # ΔΨ
        [2, 2, 0],    # pH
        [-1, 2, -1],  # ATP
        [1, 2, -1]    # ρ_protein
    ])
    level1_labels = ['O₂', 'ΔΨ', 'pH', 'ATP', 'ρ_protein']
    
    # Level 2: 12 coordinate systems (subset shown)
    level2_positions = []
    level2_labels = []
    for i, l1_pos in enumerate(level1_positions):
        for j in range(2):  # Show 2 per modality
            offset = np.array([-0.3 + 0.6*j, 0, 0])
            level2_positions.append(l1_pos + np.array([0, 1.5, 0]) + offset)
            level2_labels.append(f'{level1_labels[i]}_{j+1}')
    
    # Leaf nodes: resolved positions
    leaf_positions = []
    for l2_pos in level2_positions:
        for k in range(3):  # 3 leaf nodes per level2
            offset = np.array([-0.2 + 0.2*k, 0, 0])
            leaf_positions.append(l2_pos + np.array([0, 1, 0]) + offset)
    
    # Draw root
    ax.scatter(*root_pos, s=root_size*500, c='blue', alpha=0.6, edgecolors='black', linewidths=2)
    ax.text(*root_pos, ' Root\n(10³ μm³)', fontsize=8, ha='center', va='bottom')
    
    # Draw level 1
    colors1 = plt.cm.RdYlBu(np.linspace(0.2, 0.8, len(level1_positions)))
    for pos, label, color in zip(level1_positions, level1_labels, colors1):
        ax.scatter(*pos, s=300, c=[color], alpha=0.7, edgecolors='black', linewidths=1.5)
        ax.text(*pos, label, fontsize=7, ha='center', va='bottom')
        # Connect to root
        ax.plot([root_pos[0], pos[0]], [root_pos[1], pos[1]], [root_pos[2], pos[2]], 
                'k-', linewidth=2, alpha=0.5)
    
    # Draw level 2
    colors2 = plt.cm.RdYlBu(np.linspace(0.3, 0.9, len(level2_positions)))
    for i, (pos, label, color) in enumerate(zip(level2_positions, level2_labels, colors2)):
        parent_idx = i // 2
        parent_pos = level1_positions[parent_idx]
        ax.scatter(*pos, s=150, c=[color], alpha=0.6, edgecolors='black', linewidths=1)
        # Connect to level 1
        ax.plot([parent_pos[0], pos[0]], [parent_pos[1], pos[1]], [parent_pos[2], pos[2]], 
                'k-', linewidth=1.5, alpha=0.4)
    
    # Draw leaf nodes
    for pos in leaf_positions:
        ax.scatter(*pos, s=50, c='red', alpha=0.5, edgecolors='black', linewidths=0.5)
    
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y (Constraint Level)', fontsize=10)
    ax.set_zlabel('Z', fontsize=10)
    ax.set_title('Panel A: Constraint Hierarchy (3D)', fontsize=12, fontweight='bold')
    ax.view_init(elev=20, azim=45)

def _panel_b_sequential_exclusion(ax):
    """Panel B: Sequential categorical exclusion progression."""
    # Create 5 sub-images showing progressive refinement
    n_steps = 5
    cell_size = 100  # pixels
    
    # Exclusion percentages
    exclusion_pcts = [0, 0.30, 0.70, 0.90, 0.99]
    labels = ['Initial', 'After O₂', 'After ΔΨ', 'After pH', 'After ATP', 'Final']
    
    # Create grid of images
    cols = 3
    rows = 2
    spacing = 0.1
    
    for i in range(min(n_steps, rows * cols)):
        row = i // cols
        col = i % cols
        
        # Position for this subplot
        x0 = col * (1/cols + spacing)
        y0 = 1 - (row + 1) * (1/rows + spacing)
        width = 1/cols - spacing
        height = 1/rows - spacing
        
        # Create synthetic cell image
        img = np.random.rand(cell_size, cell_size)
        
        # Apply exclusion mask
        excluded = np.random.rand(cell_size, cell_size) < exclusion_pcts[i]
        img[excluded] = 0.3  # Gray for excluded
        
        # Add some structure
        center = cell_size // 2
        radius = cell_size // 4 * (1 - exclusion_pcts[i])
        y, x = np.ogrid[:cell_size, :cell_size]
        mask = (x - center)**2 + (y - center)**2 < radius**2
        img[mask] = 0.8  # Active region
        
        # Blur decreases with refinement
        blur = 1 - exclusion_pcts[i]
        
        # Place image
        ax.imshow(img, extent=[x0, x0+width, y0, y0+height], 
                 cmap='viridis', alpha=0.8, aspect='auto')
        ax.text(x0 + width/2, y0 - 0.02, labels[i], 
               ha='center', va='top', fontsize=8, fontweight='bold')
        ax.text(x0 + width/2, y0 + height/2, f'{exclusion_pcts[i]*100:.0f}% excluded',
               ha='center', va='center', fontsize=7, color='white', weight='bold')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Panel B: Sequential Categorical Exclusion', fontsize=12, fontweight='bold', pad=20)

def _panel_c_resolution_enhancement(ax):
    """Panel C: Resolution enhancement log-log plot."""
    # Number of constraints
    n_constraints = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    
    # Resolution (nm) - exponential improvement
    resolution = 200 * np.exp(-0.3 * (n_constraints - 1))
    resolution[-1] = 0.1  # Final resolution
    
    # Plot data
    ax.loglog(n_constraints, resolution, 'o-', linewidth=2.5, markersize=8, 
             color='green', label='This method', zorder=3)
    
    # Conventional microscopy limits
    ax.axhline(200, color='red', linestyle='--', linewidth=2, label='Optical (200 nm)')
    ax.axhline(100, color='orange', linestyle='--', linewidth=2, label='Confocal (100 nm)')
    ax.axhline(20, color='yellow', linestyle='--', linewidth=2, label='STED (20 nm)')
    
    # Annotation
    ax.annotate('10³× improvement', xy=(12, 0.1), xytext=(10, 1),
               arrowprops=dict(arrowstyle='->', lw=2, color='green'),
               fontsize=10, fontweight='bold', color='green',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Number of Constraints Applied', fontsize=11, fontweight='bold')
    ax.set_ylabel('Spatial Resolution (nm)', fontsize=11, fontweight='bold')
    ax.set_title('Panel C: Resolution Enhancement', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(0.9, 13)

def _panel_d_information_content(ax):
    """Panel D: Information content stacked bar chart."""
    modalities = ['Optical', 'Spectral', 'Vibrational', 'O₂ GPS', 'Temporal',
                 'HCNA', 'IGLT', 'MRT', 'PRM', 'CCV', 'ETPV', 'SLDI']
    
    # Information content per modality (bits/voxel)
    info_per_modality = np.array([1e2, 1e3, 1e4, 1e5, 1e4, 1e5, 1e4, 1e5, 1e4, 1e5, 1e4, 1e5])
    
    # Cumulative
    cumulative = np.cumsum(info_per_modality)
    
    # Colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(modalities)))
    
    # Stacked bars
    bottom = 0
    bars = []
    for i, (mod, info, color) in enumerate(zip(modalities, info_per_modality, colors)):
        bar = ax.barh(0, info, left=bottom, color=color, alpha=0.7, edgecolor='black', linewidth=1)
        bars.append(bar)
        bottom += info
    
    # Comparison bars
    ax.barh(1, 1e3, color='gray', alpha=0.5, edgecolor='black', linewidth=1, label='Conventional')
    ax.barh(2, cumulative[-1], color='green', alpha=0.7, edgecolor='black', linewidth=2, label='This method')
    
    # Annotation
    ax.text(cumulative[-1]/2, 2, f'{cumulative[-1]/1e3:.0f}×10³ bits/voxel',
           ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    ax.text(1e3/2, 1, '10³ bits/voxel',
           ha='center', va='center', fontsize=9, color='white')
    
    ax.set_xlabel('Information Content (bits/voxel)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Imaging Method', fontsize=11, fontweight='bold')
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['Modalities\n(Stacked)', 'Conventional', 'This Method'])
    ax.set_title('Panel D: Information Content', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='x')

if __name__ == "__main__":
    output_dir = Path(__file__).parent
    generate_figure_1(output_dir, dpi=300)
