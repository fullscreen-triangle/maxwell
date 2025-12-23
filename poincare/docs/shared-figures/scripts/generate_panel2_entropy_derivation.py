"""
Panel 2: Entropy Derivation - S = k_B M ln n

Shows the step-by-step derivation of entropy from each framework:
- Oscillatory: counting quantum states
- Categorical: counting distinguishable states  
- Partition: counting paths through tree

Demonstrates convergence to the unified formula.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle, Wedge, Arc
from matplotlib.collections import LineCollection
import matplotlib.gridspec as gridspec

# Set up style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['figure.facecolor'] = 'white'

def create_entropy_derivation_panel():
    """Create a 2x3 panel showing entropy derivation from three perspectives."""
    
    fig = plt.figure(figsize=(16, 11))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1], hspace=0.35, wspace=0.3)
    
    # Color scheme
    osc_color = '#E74C3C'
    cat_color = '#27AE60'
    part_color = '#3498DB'
    formula_color = '#8E44AD'
    
    # =========================================================================
    # Panel A: Oscillatory State Counting
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(-0.5, 5.5)
    ax1.set_ylim(-0.5, 4.5)
    ax1.axis('off')
    ax1.set_title('(A) Oscillatory: Counting Quantum States', fontsize=12, 
                  fontweight='bold', pad=10, color=osc_color)
    
    # Draw M=3 modes, each with n=4 states
    M, n = 3, 4
    
    for mode in range(M):
        # Mode label
        ax1.text(-0.3, 3.5 - mode * 1.3, f'Mode {mode+1}:', fontsize=10, va='center')
        
        # Draw n quantum state boxes
        for state in range(n):
            x = 1.0 + state * 1.1
            y = 3.5 - mode * 1.3
            
            box = FancyBboxPatch((x - 0.4, y - 0.35), 0.8, 0.7,
                                  boxstyle="round,pad=0.05",
                                  facecolor=plt.cm.Reds(0.2 + state * 0.2),
                                  edgecolor='black', linewidth=1)
            ax1.add_patch(box)
            ax1.text(x, y, f'$|{state}\\rangle$', ha='center', va='center', 
                    fontsize=9, color='white' if state > 1 else 'black')
    
    # Formula derivation
    ax1.text(2.5, 0.8, r'$W_{osc} = n \times n \times ... \times n = n^M$', 
             ha='center', fontsize=11, color=osc_color)
    ax1.text(2.5, 0.3, r'$W_{osc} = 4^3 = 64$ states', ha='center', fontsize=10)
    
    # Brace showing n choices per mode
    ax1.annotate('', xy=(4.8, 3.85), xytext=(0.6, 3.85),
                arrowprops=dict(arrowstyle='-', color='#2C3E50', lw=1.5,
                               connectionstyle='bar,angle=180,fraction=0.2'))
    ax1.text(2.7, 4.2, f'$n = {n}$ states per mode', ha='center', fontsize=9)
    
    # =========================================================================
    # Panel B: Categorical State Counting
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(-0.5, 5.5)
    ax2.set_ylim(-0.5, 4.5)
    ax2.axis('off')
    ax2.set_title('(B) Categorical: Counting Distinguishable States', fontsize=12,
                  fontweight='bold', pad=10, color=cat_color)
    
    # Draw 3D grid representing categorical space
    # Show 4x4 grid for M=2 dimensions with n=4 levels each
    grid_size = 4
    
    for i in range(grid_size):
        for j in range(grid_size):
            x = 0.8 + i * 1.0
            y = 3.2 - j * 0.8
            
            circle = Circle((x, y), 0.25, 
                            facecolor=plt.cm.Greens(0.2 + (i+j)*0.08),
                            edgecolor='black', linewidth=1)
            ax2.add_patch(circle)
            ax2.text(x, y, f'$C_{{{i+1},{j+1}}}$', ha='center', va='center', fontsize=6)
    
    # Axis labels
    ax2.annotate('', xy=(4.5, 3.5), xytext=(0.5, 3.5),
                arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=2))
    ax2.text(5.0, 3.5, r'$\mathcal{C}_1$', fontsize=10, va='center')
    
    ax2.annotate('', xy=(0.5, 0.5), xytext=(0.5, 3.5),
                arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=2))
    ax2.text(0.5, 0.1, r'$\mathcal{C}_2$', fontsize=10, ha='center')
    
    # Formula
    ax2.text(2.7, 0.6, r'$|\mathcal{C}| = n_1 \times n_2 \times ... = n^M$', 
             ha='center', fontsize=11, color=cat_color)
    ax2.text(2.7, 0.2, r'$|\mathcal{C}| = 4 \times 4 = 16$ states (for $M=2$)', 
             ha='center', fontsize=9)
    
    # =========================================================================
    # Panel C: Partition Path Counting
    # =========================================================================
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_xlim(-0.5, 5.5)
    ax3.set_ylim(-0.5, 4.5)
    ax3.axis('off')
    ax3.set_title('(C) Partition: Counting Paths Through Tree', fontsize=12,
                  fontweight='bold', pad=10, color=part_color)
    
    # Draw partition tree with M=2 levels, n=3 branches
    M_tree, n_tree = 2, 3
    
    # Root
    root_x, root_y = 2.5, 4.0
    root = Circle((root_x, root_y), 0.2, facecolor=part_color, edgecolor='black', linewidth=2)
    ax3.add_patch(root)
    ax3.text(root_x, root_y, 'R', ha='center', va='center', fontsize=10, 
             fontweight='bold', color='white')
    
    # Level 1
    level1_x = [1.0, 2.5, 4.0]
    level1_y = 2.8
    for i, x in enumerate(level1_x):
        ax3.plot([root_x, x], [root_y - 0.2, level1_y + 0.15], 'k-', linewidth=1.5)
        node = Circle((x, level1_y), 0.15, facecolor=plt.cm.Blues(0.4 + i*0.15),
                       edgecolor='black', linewidth=1.5)
        ax3.add_patch(node)
    
    # Level 2 (leaves)
    level2_y = 1.6
    leaf_positions = []
    for i, parent_x in enumerate(level1_x):
        for j in range(n_tree):
            offset = (j - 1) * 0.4
            x = parent_x + offset
            leaf_positions.append(x)
            ax3.plot([parent_x, x], [level1_y - 0.15, level2_y + 0.1], 'k-', linewidth=1)
            leaf = Circle((x, level2_y), 0.1, facecolor=plt.cm.Blues(0.2 + (i*3+j)*0.05),
                          edgecolor='black', linewidth=1)
            ax3.add_patch(leaf)
    
    # Highlight one path
    path_color = '#E74C3C'
    ax3.plot([root_x, level1_x[1]], [root_y - 0.2, level1_y + 0.15], 
             color=path_color, linewidth=3, alpha=0.7)
    ax3.plot([level1_x[1], level1_x[1] + 0.4], [level1_y - 0.15, level2_y + 0.1],
             color=path_color, linewidth=3, alpha=0.7)
    
    # Labels
    ax3.text(-0.3, root_y, 'Level 0', fontsize=8, va='center')
    ax3.text(-0.3, level1_y, 'Level 1', fontsize=8, va='center')
    ax3.text(-0.3, level2_y, 'Level 2', fontsize=8, va='center')
    
    # Count leaves
    total_leaves = n_tree ** M_tree
    ax3.text(2.5, 0.8, r'Paths = $n \times n \times ... = n^M$', 
             ha='center', fontsize=11, color=part_color)
    ax3.text(2.5, 0.4, f'Leaves = $3^2 = {total_leaves}$ paths', ha='center', fontsize=10)
    ax3.text(2.5, 0.0, '(One path highlighted in red)', ha='center', fontsize=8, 
             style='italic', color=path_color)
    
    # =========================================================================
    # Panel D: Boltzmann's Relation
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_xlim(-1, 11)
    ax4.set_ylim(-1, 6)
    ax4.axis('off')
    ax4.set_title('(D) Boltzmann\'s Fundamental Relation', fontsize=12, 
                  fontweight='bold', pad=10)
    
    # Central formula box
    formula_box = FancyBboxPatch((2, 3.5), 6, 1.5, boxstyle="round,pad=0.1",
                                  facecolor='#F8F9FA', edgecolor='#2C3E50', linewidth=2)
    ax4.add_patch(formula_box)
    ax4.text(5, 4.25, r'$S = k_B \ln W$', ha='center', va='center', 
             fontsize=18, fontweight='bold', color='#2C3E50')
    
    # Arrow pointing down
    ax4.annotate('', xy=(5, 2.8), xytext=(5, 3.4),
                arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=2))
    
    # Substitution
    ax4.text(5, 2.4, r'Substitute $W = n^M$:', ha='center', fontsize=11)
    
    # Derivation steps
    steps = [
        r'$S = k_B \ln(n^M)$',
        r'$S = k_B \cdot M \ln n$',
    ]
    
    for i, step in enumerate(steps):
        y = 1.6 - i * 0.6
        ax4.text(5, y, step, ha='center', fontsize=12,
                color='#2C3E50', fontweight='normal')
    
    # Final boxed formula
    final_box = FancyBboxPatch((2.5, -0.1), 5, 0.6, boxstyle="round,pad=0.05",
                                facecolor=formula_color, edgecolor='black', linewidth=2, alpha=0.9)
    ax4.add_patch(final_box)
    ax4.text(5, 0.2, r'$S = k_B M \ln n$', ha='center', fontsize=14,
            color='white', fontweight='bold')
    
    # Side annotations
    ax4.text(0.5, 4.25, 'Entropy', ha='center', fontsize=10, color='gray')
    ax4.text(9.5, 4.25, 'Microstates', ha='center', fontsize=10, color='gray')
    
    # =========================================================================
    # Panel E: Three Derivations Converge
    # =========================================================================
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_xlim(-1, 11)
    ax5.set_ylim(-1, 6)
    ax5.axis('off')
    ax5.set_title('(E) Three Derivations, One Formula', fontsize=12, 
                  fontweight='bold', pad=10)
    
    # Three source boxes
    sources = [
        (1.5, 5, 'Oscillatory', osc_color, r'$W_{osc} = n^M$'),
        (5, 5, 'Categorical', cat_color, r'$|\mathcal{C}| = n^M$'),
        (8.5, 5, 'Partition', part_color, r'$P = n^M$'),
    ]
    
    for x, y, label, color, formula in sources:
        box = FancyBboxPatch((x - 1.2, y - 0.6), 2.4, 1.2, boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor='black', linewidth=2, alpha=0.8)
        ax5.add_patch(box)
        ax5.text(x, y + 0.2, label, ha='center', va='center', fontsize=10, 
                fontweight='bold', color='white')
        ax5.text(x, y - 0.2, formula, ha='center', va='center', fontsize=9, color='white')
        
        # Arrow down
        ax5.annotate('', xy=(x, 3.3), xytext=(x, 4.3),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2))
    
    # Convergence point
    convergence_box = FancyBboxPatch((2.5, 2.4), 5, 1.2, boxstyle="round,pad=0.1",
                                      facecolor='#ECF0F1', edgecolor='#2C3E50', linewidth=3)
    ax5.add_patch(convergence_box)
    ax5.text(5, 3.0, r'All give: $W = n^M$', ha='center', va='center', fontsize=12)
    
    # Final arrow and formula
    ax5.annotate('', xy=(5, 1.4), xytext=(5, 2.3),
                arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=2))
    
    final_box = FancyBboxPatch((2, 0.3), 6, 1.2, boxstyle="round,pad=0.1",
                                facecolor=formula_color, edgecolor='black', linewidth=2)
    ax5.add_patch(final_box)
    ax5.text(5, 0.9, r'$S = k_B M \ln n$', ha='center', va='center', 
             fontsize=16, fontweight='bold', color='white')
    
    # =========================================================================
    # Panel F: Entropy vs Parameters
    # =========================================================================
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_title('(F) Entropy Scaling: $S = k_B M \\ln n$', fontsize=12, 
                  fontweight='bold', pad=10)
    
    # Plot entropy surface
    M_vals = np.linspace(1, 10, 50)
    n_vals = np.linspace(2, 10, 50)
    M_grid, n_grid = np.meshgrid(M_vals, n_vals)
    S_grid = M_grid * np.log(n_grid)  # In units of k_B
    
    # Contour plot
    contour = ax6.contourf(M_grid, n_grid, S_grid, levels=20, cmap='viridis')
    cbar = plt.colorbar(contour, ax=ax6, label=r'$S/k_B$')
    
    # Contour lines
    ax6.contour(M_grid, n_grid, S_grid, levels=[5, 10, 15, 20], colors='white', 
                linewidths=1, linestyles='--')
    
    ax6.set_xlabel(r'Degrees of Freedom $M$', fontsize=11)
    ax6.set_ylabel(r'States per DOF $n$', fontsize=11)
    
    # Annotations
    ax6.annotate(r'$S \propto M$' + '\n(linear)', xy=(8, 3), fontsize=9, 
                color='white', ha='center')
    ax6.annotate(r'$S \propto \ln n$' + '\n(logarithmic)', xy=(3, 8), fontsize=9,
                color='white', ha='center')
    
    # Mark specific points
    ax6.plot(3, 4, 'r*', markersize=15)
    ax6.text(3.5, 4.5, 'Pendulum:\n$M=1, n=4$', fontsize=8, color='red')
    
    ax6.plot(6, 3, 'y*', markersize=15)
    ax6.text(6.5, 3.5, 'Gas:\n$M=6, n=3$', fontsize=8, color='yellow')
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    import os
    
    # Get the script directory and construct output path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, '..', 'figures')
    os.makedirs(output_dir, exist_ok=True)
    
    fig = create_entropy_derivation_panel()
    
    # Save in multiple formats
    fig.savefig(os.path.join(output_dir, 'panel2_entropy_derivation.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(os.path.join(output_dir, 'panel2_entropy_derivation.pdf'), 
                bbox_inches='tight', facecolor='white')
    
    print("Panel 2 (Entropy Derivation) saved successfully!")
    plt.close()

